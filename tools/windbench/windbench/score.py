"""Score a prediction directory: one `<crop_id>.npz` per crop with any of
  k        int16 (512,512,512), -1 = no prediction           -> wrap accuracy, switches along GT traces
  defects  bool (512,512,512) or (M,3) float points           -> defect precision/recall
  trace_points + trace_offsets  predicted surface polylines   -> switches along the prediction in GT k
Coordinates are crop voxels (z, y, x) at level 2 of volume 20251217075048.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .build import load_crop
from .metrics import defect_pr, switch_rate, wrap_accuracy


def score_crop(pred: dict, crop: dict) -> dict:
    out = {}
    f, k = crop["f"], crop["k"]
    if "k" in pred:
        kp = pred["k"].astype(np.int16)
        out["acc_known"] = wrap_accuracy(kp, k)
        out["acc_sheet"] = wrap_accuracy(kp, k, mask=np.abs(f - k) <= 0.1)
        out["switches_gt_traces"] = switch_rate(crop["traces"], kp)
    if "trace_points" in pred:
        off = pred["trace_offsets"]
        tr = [pred["trace_points"][a:b] for a, b in zip(off[:-1], off[1:])]
        out["switches_pred_traces"] = switch_rate(tr, k)
    if "defects" in pred:
        types = {s["id"]: s["type"] for s in crop["sites_meta"]}
        out["defects"] = defect_pr(pred["defects"], crop["sites"], types, r=20)
    return out


def score_dir(pred_dir: Path, crop_ids: list[str]) -> dict:
    per = {}
    for cid in crop_ids:
        p = pred_dir / f"{cid}.npz"
        if not p.exists():
            per[cid] = {"missing": True}
            continue
        per[cid] = score_crop(dict(np.load(p)), load_crop(cid))
    summ = {}
    for key, sub in (("acc_known", "acc"), ("acc_sheet", "acc"), ("switches_gt_traces", "per_1000"),
                     ("switches_pred_traces", "per_1000"), ("defects", "recall"), ("defects", "precision")):
        vals = [r[key][sub] for r in per.values() if key in r]
        if vals:
            summ[f"{key}.{sub}.mean"] = float(np.nanmean(vals))
    return {"summary": summ, "crops": per}
