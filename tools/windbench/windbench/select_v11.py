"""v1.1 crop selection (README, "v1.1" item 2): compressed boxes by voxel pitch, box-filling coverage.

Candidates come from the GT meshes only. They are built in rank order and kept when the built
crop's median voxel pitch <= 14.4 and coverage >= 40 % (GT-only design properties).
Nothing else about the crop is looked at here. Stops at 4 kept or 10 tried. Idempotent: kept
crops are recorded in manifest/crops_v11.json, tried ones in manifest/select_v11_log.json.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from .build import build_crop, crop_dir
from .rasterise import axis_centre, load_meshes
from .select import candidates, eligible_v11, label_surfaces, overlaps, vertex_pitch

MAN = Path(__file__).resolve().parent.parent / "manifest"
V1_KEEP = ["c04", "c05", "c06", "c07", "c08"]
MAX_PITCH = 14.4
MIN_COVERAGE = 0.40
MAX_KEEP, MAX_TRY = 4, 10


def crop_gt_stats(cid: str) -> dict:
    import zarr
    g = zarr.open_group(str(crop_dir(cid) / "crop.zarr"), mode="r")
    f, ct, pitch = g["f"][...], g["ct"][...], g["pitch"][...]
    known = np.isfinite(f)
    inscroll = ct > 0
    pk = pitch[known & np.isfinite(pitch)]
    return {"coverage": float((known & inscroll).sum() / max(1, inscroll.sum())),
            "median_voxel_pitch": float(np.median(pk)) if len(pk) else float("nan")}


def run():
    v1 = {c["id"]: c for c in json.loads((MAN / "crops.json").read_text())["crops"]}
    taken = [v1[c]["origin"] for c in V1_KEEP]
    meshes = load_meshes()
    centre = axis_centre(meshes)
    pts, lab, pit = vertex_pitch(label_surfaces(meshes))
    ranked = eligible_v11(candidates(pts, lab, pit), centre, taken)
    print(f"{len(ranked)} eligible boxes; best proxies:",
          [(r["origin"], r["voxel_pitch_proxy"]) for r in ranked[:12]], flush=True)
    log_p = MAN / "select_v11_log.json"
    log = json.loads(log_p.read_text()) if log_p.exists() else []
    tried = {tuple(t["origin"]): t for t in log}
    kept = [t for t in log if t["kept"]]
    for r in ranked:
        if len(kept) >= MAX_KEEP or len(tried) >= MAX_TRY:
            break
        if tuple(r["origin"]) in tried or any(overlaps(r["origin"], k["origin"]) for k in kept):
            continue
        cid = f"p{len(kept) + 1:02d}"
        t0 = time.time()
        build_crop(meshes, {"id": cid, "origin": r["origin"]}, force=True)
        st = crop_gt_stats(cid)
        ok = st["median_voxel_pitch"] <= MAX_PITCH and st["coverage"] >= MIN_COVERAGE
        rec = {"origin": r["origin"], "id": cid if ok else None, "kept": ok, **st,
               "voxel_pitch_proxy": r["voxel_pitch_proxy"], "vertex_median_pitch": r["median_pitch"],
               "wrong_side_vertices": round(r["wrong_side"], 4), "chunk_occupancy": r["chunk_occupancy"],
               "seam_distance_deg": r["seam_distance_deg"], "labels": [min(r["pairs"]), r["max_label"]],
               "seconds": round(time.time() - t0, 1)}
        print(json.dumps(rec), flush=True)
        if not ok:
            shutil.rmtree(crop_dir(cid))
        else:
            kept.append(rec)
        tried[tuple(r["origin"])] = rec
        log.append(rec)
        log_p.write_text(json.dumps(log, indent=1))
    crops = [{**v1[c], "set": "v1"} for c in V1_KEEP]
    crops += [{"id": k["id"], "origin": k["origin"], "why": "compressed (v1.1 rule)", "set": "v1.1",
               "labels": k["labels"], "voxel_pitch_proxy": k["voxel_pitch_proxy"],
               "vertex_median_pitch": k["vertex_median_pitch"], "wrong_side_vertices": k["wrong_side_vertices"],
               "chunk_occupancy": k["chunk_occupancy"], "seam_distance_deg": k["seam_distance_deg"]} for k in kept]
    (MAN / "crops_v11.json").write_text(json.dumps(
        {"version": "v1.1", "level": 2, "volume": "20251217075048", "size": 512,
         "selection": "tools/windbench/windbench/select_v11.py + experiments/2026-10-13-w0b-windbench/README.md (v1.1)",
         "crops": crops}, indent=1))
    print(f"kept {len(kept)} of {len(tried)} tried")


if __name__ == "__main__":
    run()
