"""Frozen point-vs-reference update-mismatch diagnostic for the paper intro."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .core import group_standardized_coefficients
from .microbatch import microbatch_row_from_transaction

def _vec(row, names):
    for name in names:
        value = row.get(name)
        if value is not None:
            return np.asarray(value, dtype=float)
    return None


def _scores(row):
    estimated = _vec(row, ("estimated_scores", "scores", "point_scores"))
    reference = _vec(row, ("reference_scores", "reference", "gold_scores"))
    if estimated is not None and reference is not None:
        return estimated, reference
    try:
        converted = microbatch_row_from_transaction(row, estimated_record="new", reference_record="old")
    except Exception:
        return None, None
    metadata = converted["metadata"]
    return np.asarray(metadata["estimated_scores"], dtype=float), np.asarray(metadata["reference_scores"], dtype=float)


def _row_keys(row):
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return {
        str(value)
        for value in (
            row.get("transaction_id"), row.get("microbatch_id"), row.get("id"),
            metadata.get("transaction_id"), metadata.get("microbatch_id"),
        )
        if value not in (None, "")
    }


def _load_geometry(paths):
    geometry = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get("jacobian")
            if value is None and isinstance(row.get("geometry"), dict):
                value = row["geometry"].get("G")
            if value is None:
                continue
            for key in _row_keys(row):
                geometry[key] = value
    return geometry

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transactions", required=True)
    ap.add_argument("--geometry", action="append", default=[])
    ap.add_argument("--output", required=True)
    ap.add_argument("--bootstrap-samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rows=[]
    for line in Path(args.transactions).read_text(encoding="utf-8").splitlines():
        if line.strip(): rows.append(json.loads(line))
    geometry_by_key = _load_geometry(args.geometry)
    records=[]
    for row in rows:
        est, ref = _scores(row)
        if est is None or ref is None or est.shape != ref.shape or est.ndim != 1: continue
        try: ce=group_standardized_coefficients(est); cr=group_standardized_coefficients(ref)
        except Exception: continue
        drift=float(np.linalg.norm(ce-cr))
        coefficient_denom=float(np.linalg.norm(ce)*np.linalg.norm(cr))
        coefficient_cosine=float(np.dot(ce,cr)/coefficient_denom) if coefficient_denom else 0.0
        jac = row.get("jacobian")
        if jac is None and isinstance(row.get("geometry"), dict):
            jac = row["geometry"].get("G")
        if jac is None:
            for key in _row_keys(row):
                if key in geometry_by_key:
                    jac = geometry_by_key[key]
                    break
        sensitivity = float("nan")
        update_cosine = float("nan")
        if jac is not None:
            geometry = np.asarray(jac, dtype=float)
            if geometry.ndim == 2:
                if geometry.shape[0] != ce.size and geometry.shape[1] == ce.size:
                    geometry = geometry.T
                if geometry.shape[0] == ce.size:
                    sensitivity = float(np.linalg.norm(geometry, ord="fro"))
                    point_update = ce @ geometry
                    reference_update = cr @ geometry
                    update_denom = float(np.linalg.norm(point_update) * np.linalg.norm(reference_update))
                    if update_denom:
                        update_cosine = float(np.dot(point_update, reference_update) / update_denom)
        records.append({
            "transaction_id":str(row.get("transaction_id",row.get("id",""))),
            "score_error_l2":float(np.linalg.norm(est-ref)),
            "estimated_reward_std":float(np.std(est)),
            "coefficient_drift_l2":drift,
            "coefficient_cosine":coefficient_cosine,
            "update_cosine":update_cosine,
            "geometry_effect_delta":update_cosine-coefficient_cosine if np.isfinite(update_cosine) else float("nan"),
            "reversal":bool(update_cosine<0) if np.isfinite(update_cosine) else None,
            "jacobian_sensitivity":sensitivity,
        })
    rng = np.random.default_rng(args.seed)
    def ci(values):
        values = np.asarray(values, dtype=float)
        if values.size == 0: return [0.0, 0.0, 0.0]
        draws = rng.choice(values, size=(max(1, args.bootstrap_samples), values.size), replace=True).mean(axis=1)
        return [float(values.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]
    def summarize(subset):
        update_rows = [r for r in subset if np.isfinite(r["update_cosine"])]
        return {
            "rows": len(subset),
            "update_rows": len(update_rows),
            "score_error_ci": ci([r["score_error_l2"] for r in subset]),
            "coefficient_drift_ci": ci([r["coefficient_drift_l2"] for r in subset]),
            "reversal_rate_ci": ci([float(r["reversal"]) for r in update_rows]),
        }

    def median_strata(value_key):
        eligible = [r for r in records if np.isfinite(r[value_key])]
        if not eligible:
            return []
        edges = np.quantile([r[value_key] for r in eligible], [0, .5, 1])
        result = []
        for label, lo, hi in (("low", edges[0], edges[1]), ("high", edges[1], edges[2])):
            subset = [
                r for r in eligible
                if r[value_key] >= lo and (r[value_key] <= hi if label == "high" else r[value_key] < hi)
            ]
            result.append({"stratum": label, "value": value_key, **summarize(subset)})
        return result

    sensitivity_strata = median_strata("jacobian_sensitivity")
    reward_std_strata = median_strata("estimated_reward_std")
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    finite = [r for r in records if np.isfinite(r["jacobian_sensitivity"])]
    update_records = [r for r in records if np.isfinite(r["update_cosine"])]
    geometry_rows = len(finite)
    out.write_text(json.dumps({"rows":len(records),"geometry_rows":geometry_rows,"update_rows":len(update_records),"reversal_rate_ci":ci([float(r["reversal"]) for r in update_records]),"score_error_ci":ci([r["score_error_l2"] for r in records]),"coefficient_drift_ci":ci([r["coefficient_drift_l2"] for r in records]),"jacobian_sensitivity_strata":sensitivity_strata,"reward_std_strata":reward_std_strata,"records":records},indent=2),encoding="utf-8")
    print(f"intro_diagnostic_rows={len(records)} geometry_rows={geometry_rows} output={out}")
if __name__ == "__main__": main()
