#!/usr/bin/env python3
"""Score one spiral fit and add its row to the leaderboard. See README.md.

    tools/spiralfiteval/leaderboard/add_row.sh --id <id-in-fits.json>
    tools/spiralfiteval/leaderboard/add_row.sh --fit PATH --id my-id --author "Me" \
        -- --references PHercParis4 --z 8400 9400 --floors 5

Everything after ``--`` goes to ``spiralfiteval score`` unchanged. For a fit listed in
fits.json the arguments and the download URL come from the registry. The row records the
command, the SHA-256 of the fit input, the tool commit and the measured cost, then the table
and the CSV are rebuilt. Standard library plus the spiralfiteval install; no network unless
something is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "fits.json"
ROWS = Path(os.environ.get("LEADERBOARD_ROWS", HERE / "rows"))   # overridden by selftest.sh


def repo_root() -> Path:
    for p in HERE.parents:
        if (p / "tools" / "spiralfiteval").is_dir():
            return p
    raise SystemExit("cannot find the repository root above " + str(HERE))


REPO = repo_root()

# the scoring child reports its own peak RSS, so the number is the row's, not this driver's
RUNNER = """
import json, resource, sys, time
from spiralfiteval.cli import main
t0 = time.time()
rc = main(sys.argv[1:])
r = resource.getrusage(resource.RUSAGE_SELF)
print("__ROW_COST__ " + json.dumps({"seconds": round(time.time() - t0, 1),
      "peak_rss_gb": round(r.ru_maxrss / 2**20, 2)}), flush=True)
sys.exit(rc)
"""


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_hash(path: Path) -> dict:
    """SHA-256 of a fit input: the file itself, or a sorted manifest of a mesh folder
    (``relpath  sha256`` per file, newline separated, hashed again)."""
    if path.is_file():
        return {"kind": "file", "sha256": sha256_file(path), "files": 1,
                "bytes": path.stat().st_size}
    files = sorted(f for f in path.rglob("*") if f.is_file())
    man = "".join(f"{f.relative_to(path).as_posix()}  {sha256_file(f)}\n" for f in files)
    return {"kind": "manifest", "sha256": hashlib.sha256(man.encode()).hexdigest(),
            "files": len(files), "bytes": sum(f.stat().st_size for f in files)}


def rel(p: Path) -> str:
    """Repository-relative path when the input is inside the clone, else absolute.
    ``absolute()`` first, so a gitignored ``data/`` that is a symlink elsewhere still prints
    as ``data/...`` (which is where a fresh clone puts it)."""
    for cand in (Path(p).absolute(), Path(p).resolve()):
        try:
            return str(cand.relative_to(REPO))
        except ValueError:
            pass
    return str(p)


def tool_commit() -> dict:
    def git(*a):
        r = subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain"))}


def fetch_fit(entry: dict, dest: Path) -> None:
    """Download a registry fit to its ``local`` path if it is not there yet."""
    src = entry["source"]
    if src["kind"] == "curl":
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"+ curl {src['url']}", flush=True)
        subprocess.run(["curl", "-sfL", "-o", str(dest), src["url"]], check=True)
    elif src["kind"] == "git":
        # the fit is one folder of a repository: sparse, blobless, pinned to its commit
        root = dest
        while root.name != Path(src["sparse"]).name and root.parent != root:
            root = root.parent
        root = root.parent
        print(f"+ git clone --sparse {src['url']} {root}", flush=True)
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--sparse",
                        src["url"], str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "sparse-checkout", "set", src["sparse"]], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", "-q", src["commit"]], check=True)
    else:
        raise SystemExit(f"unknown source kind {src['kind']!r} for {entry['id']}")


def ensure_references(score_args: list[str]) -> None:
    if "--references" not in score_args:
        return
    scroll = score_args[score_args.index("--references") + 1]
    from spiralfiteval.data import scroll_dir
    if (scroll_dir(scroll) / "relative_windings.json").exists() and \
            any((scroll_dir(scroll) / "eval_fibers").glob("*.json")):
        return
    print(f"+ spiralfiteval fetch --scroll {scroll}  (~290 MB, once)", flush=True)
    subprocess.run([sys.executable, "-m", "spiralfiteval.cli", "fetch", "--scroll", scroll],
                   check=True)


def run_score(fit: Path, score_args: list[str], out: Path) -> tuple[dict, dict, str]:
    """Run ``spiralfiteval score`` in a child; return (score, cost, console)."""
    argv = ["score", str(fit), *score_args, "--out", str(out)]
    cmd = [sys.executable, "-c", RUNNER, *argv]
    print("+ spiralfiteval " + shlex.join(argv), flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    sys.stdout.write(p.stdout)
    sys.stderr.write(p.stderr)
    if p.returncode != 0:
        raise SystemExit(f"spiralfiteval score failed (rc={p.returncode})")
    cost = {"seconds": round(time.time() - t0, 1), "peak_rss_gb": None}
    m = re.search(r"__ROW_COST__ (\{.*\})", p.stdout)
    if m:
        cost = json.loads(m.group(1))
        cost["wall_seconds_with_startup"] = round(time.time() - t0, 1)
    console = "\n".join(l for l in p.stdout.splitlines() if not l.startswith("__ROW_COST__"))
    return json.loads(out.read_text())[0], cost, console


FLOOR_KEYS = {"H1": "slips per wrap", "H1_ladders": "slips per wrap on the ladder set",
              "H2": "switches per 1,000 vx", "H3": "median distance"}


def floor_verdicts(score: dict) -> dict:
    """README's rule: a metric is at floor when margin < 2 x floor_sd over the phase-jitter
    floor (for H3 the smaller of the jitter and random-point margins)."""
    fl = score.get("floors")
    if not fl:
        # no --floors: say so for the reference metrics the row does have, nothing for H5-only rows
        return {k: "not measured" for k in FLOOR_KEYS if k in score}
    out = {}
    for key in FLOOR_KEYS:
        b = fl.get(key)
        if not b or "jitter" not in b:
            continue
        cands = [b["jitter"]] + ([b["random"]] if key == "H3" and "random" in b else [])
        worst = min(cands, key=lambda c: (c["margin"] / c["floor_sd"]) if c.get("floor_sd") else -1e9)
        sd, margin = worst.get("floor_sd"), worst["margin"]
        usable = sd is not None and sd == sd and sd > 0   # a floor with no spread cannot be tested
        z = (margin / sd) if usable else float("nan")
        out[key] = {"value": b["jitter"]["value"], "floor_mean": worst["floor_mean"],
                    "floor_sd": sd, "margin": margin, "margin_over_sd": z,
                    "at_floor": bool(z < 2.0) if usable else None,
                    "floor_used": "random" if worst is not cands[0] else "jitter"}
    return out


def headline(score: dict) -> dict:
    h1, h2, h3, h5 = score.get("H1"), score.get("H2"), score.get("H3"), score["H5"]
    out = {"vertices": score["vertices"], "windings": score["windings"],
           "grid_step_vox": score["grid_step_vox"],
           "H5_duplicate": h5["duplicate"], "H5_median_gap_vox": h5.get("median"),
           "H5_no_neighbour": h5["no_neighbour"]}
    if h1:
        out.update(H1_slips_per_wrap=h1["slips_per_wrap"], H1_slips_ci95=h1.get("slips_ci95"),
                   H1_exact=h1.get("exact"), H1_adjacent_tests=h1["adjacent_tests"],
                   H1_chains=h1.get("chains"), H1_covered=h1.get("human_pairs_covered"))
    if h2:
        out.update(H2_per_1000_vox=h2["per_1000_vox"], H2_traces_with_switch=h2["traces_with_switch"])
    if h3:
        out.update(H3_median_vox=h3.get("median"), H3_p90_vox=h3.get("p90"),
                   H3_within_4=h3["within_4"], H3_not_on_fit=h3["not_on_fit"])
    if "H5_by_winding" in score:
        w = score["H5_by_winding"]
        out.update(H5_mean_over_windings=w["mean_over_windings"], H5_area_weighted=w["area_weighted"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True, help="row id; if it is in fits.json nothing else is needed")
    ap.add_argument("--fit", default=None, help="fit folder or .npz (default: the registry's)")
    ap.add_argument("--author", default=None)
    ap.add_argument("--scroll-name", default=None, help="free-text scroll name for the table")
    ap.add_argument("--fit-url", default=None, help="public URL of the fit (required for a pull request)")
    ap.add_argument("--no-table", action="store_true", help="skip rebuilding TABLE.md / the CSV")
    ap.add_argument("score_args", nargs="*", help="after --: arguments for spiralfiteval score")
    args = ap.parse_args(argv)

    reg = {f["id"]: f for f in json.loads(REGISTRY.read_text())["fits"]} if REGISTRY.exists() else {}
    entry = dict(reg.get(args.id, {}))
    if not entry and not (args.fit and args.author):
        raise SystemExit(f"{args.id!r} is not in fits.json: pass --fit and --author too")
    score_args = list(args.score_args) or list(entry.get("score_args", []))
    fit = Path(args.fit) if args.fit else (REPO / entry["local"])
    if not args.fit and not fit.exists():
        fetch_fit(entry, fit)
    if not fit.exists():
        raise SystemExit(f"no fit at {fit}")
    ensure_references(score_args)

    ROWS.mkdir(parents=True, exist_ok=True)
    raw = ROWS / f".raw_{args.id}.json"
    score, cost, console = run_score(fit, score_args, raw)
    raw.unlink(missing_ok=True)

    shown = f"tools/spiralfiteval/leaderboard/add_row.sh --id {args.id}"
    if args.fit or args.author:
        shown = ("tools/spiralfiteval/leaderboard/add_row.sh"
                 f" --fit {shlex.quote(rel(fit))} --id {args.id}"
                 + (f" --author {shlex.quote(args.author)}" if args.author else "")
                 + (" -- " + shlex.join(score_args) if score_args else ""))
    row = {
        "id": args.id,
        "author": args.author or entry.get("author"),
        "author_url": entry.get("author_url"),
        "release": entry.get("release"),
        "scroll": args.scroll_name or entry.get("scroll"),
        "frame": entry.get("frame"),
        "fitter": entry.get("fitter"),
        "run": entry.get("run"),
        "band_z": entry.get("band_z"),
        "fit_inputs": entry.get("fit_inputs"),
        "held_out_H1": entry.get("held_out_H1"),
        "published_slips_per_wrap": entry.get("published_slips_per_wrap"),
        "published_slips_ci95": entry.get("published_slips_ci95"),
        "format": entry.get("format"),
        "fit_url": args.fit_url or (entry.get("source") or {}).get("url"),
        "fit_commit": (entry.get("source") or {}).get("commit"),
        "note": entry.get("note"),
        "command": shown,
        "score_command": "spiralfiteval " + shlex.join(["score", rel(fit), *score_args]),
        "input_hash": input_hash(fit),
        "tool": tool_commit(),
        "python": sys.version.split()[0],
        "scored_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cost": cost,
        "headline": headline(score),
        "floors": floor_verdicts(score),
        "console": console,
        "score": score,
    }
    out = ROWS / f"{args.id}.json"
    out.write_text(json.dumps(row, indent=1, default=float) + "\n")
    print(f"row -> {rel(out)}  ({cost['seconds']} s, peak {cost['peak_rss_gb']} GB)")
    if not args.no_table:
        subprocess.run([sys.executable, str(HERE / "build_table.py")], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
