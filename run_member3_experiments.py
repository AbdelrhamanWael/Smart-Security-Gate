"""
run_member3_experiments.py - Member 3: Early Fusion + Hierarchy + Evaluation Suite
=================================================================================
Runs, for every enrolled member that has face + voice + fingerprint data:

  E4  Early fusion (Mode C): SVM vs Random Forest      vs. face-only (A) / voice-only (B)
  E5  Hierarchical pipeline (Mode D) with / without fingerprint,
      T_low policy ablation, processing-time comparison hierarchy vs flat
  E6  Fusion design choices: block weights, early vs late fusion (score average)
  E7  Robustness: low light / over-exposure on the face, white noise on the voice

Protocol (see src/multimodal_dataset.py for the leakage guarantees)
  DEV  = genuine sessions 1-2 + SEEN impostor identities   -> training, tuning, threshold freezing
  TEST = genuine session 3 (raw) + UNSEEN impostor identities (open set) -> reported numbers

Outputs -> results/m3/   (tables .md/.csv/.json, ROC + confusion-matrix figures)
Models  -> models/fusion/, models/hierarchy/   (used by hierarchical_verify / demo_member3.py)

Usage:  python run_member3_experiments.py [--users member1 member2] [--refit-all] [--skip-robustness]
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
import zlib
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")

import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.evaluation import (compute_eer, confusion_from_metrics, evaluate, plot_confusion_matrix,
                            plot_roc_comparison, plot_roc_curve, plot_score_distributions,
                            threshold_at_eer, threshold_at_far, threshold_at_frr)
from src.face_models import build_face_pipeline
from src.face_pipeline import crop_face, extract_face_features
from src.fingerprint import build_fingerprint_pipeline, extract_fingerprint_features, preprocess_fingerprint
from src.fusion import FUSION_PARAM_GRID, build_fusion_pipeline, save_fusion_model
from src.hierarchy import (GateBundle, Thresholds, cascade_levels, cascade_score, expected_time, save_bundle)
from src.multimodal_dataset import build_rows, gather, load_pools, users_with_all_modalities
from src.voice_features import extract_voice_features, load_audio
from src.voice_verification import VoiceVerifier

OUT = ROOT / "results" / "m3"
NAMES = {
    "face_svm": "Face-only SVM", "face_knn": "Face-only KNN",
    "voice_svm": "Voice-only SVM", "voice_rf": "Voice-only RF",
    "fusion_svm": "Early Fusion SVM", "fusion_rf": "Early Fusion RF",
    "fp_svm": "Fingerprint-only SVM", "fp_knn": "Fingerprint-only KNN",
}
METRIC_COLS = ["accuracy", "precision", "recall", "f1", "FAR", "FRR", "EER", "AUC"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def make_splits(y, g, n_splits=2):
    """Session/identity grouped CV folds in which every fold has both classes."""
    for seed in range(42, 120):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        sp = list(cv.split(np.zeros(len(y)), y, g))
        if all(len(np.unique(y[a])) == 2 and len(np.unique(y[b])) == 2 for a, b in sp):
            return sp
    raise RuntimeError("Could not build grouped folds with both classes in every fold")


def oof_scores(factory, X, y, splits):
    oof = np.zeros(len(y))
    for tr, va in splits:
        oof[va] = factory().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    return oof


def finite_or(t, default=0.5):
    return float(t) if np.isfinite(t) else default


def tune_fusion(kind, X, y, splits):
    """Pick fusion hyper-parameters / block weights by grouped-CV AUC on DEV only."""
    rows = []
    for params in FUSION_PARAM_GRID[kind]:
        aucs, eers = [], []
        for tr, va in splits:
            s = build_fusion_pipeline(kind, **params).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
            aucs.append(roc_auc_score(y[va], s))
            eers.append(compute_eer(y[va], s)[0])
        rows.append({"params": params, "auc": float(np.mean(aucs)), "eer": float(np.mean(eers))})
    best = max(rows, key=lambda r: (round(r["auc"], 6), -r["eer"]))
    return best["params"], rows


def timed(fn, n=1):
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
    return out, (time.perf_counter() - t0) / n


# ---------------------------------------------------------------------------
# Per-user experiment
# ---------------------------------------------------------------------------
def run_user(user, pools, args):
    print(f"\n{'=' * 78}\n  USER: {user}\n{'=' * 78}")
    dev, test, info = build_rows(user, pools, seed=args.seed)
    Fd, Vd, Pd = gather(dev, pools)
    Ft, Vt, Pt = gather(test, pools)
    yd, gd, yt = dev.y, dev.groups, test.y
    print(f"  DEV : {len(dev)} rows ({yd.sum()} genuine / {(yd == 0).sum()} impostor), groups={sorted(set(gd))}")
    print(f"  TEST: {len(test)} rows ({yt.sum()} genuine / {(yt == 0).sum()} impostor), "
          f"unseen impostors={[v['name'] for v in info['unseen_vids']]}")
    splits = make_splits(yd, gd)

    Xd = {"face": Fd, "voice": Vd, "fp": Pd, "fusion": np.hstack([Fd, Vd])}
    Xt = {"face": Ft, "voice": Vt, "fp": Pt, "fusion": np.hstack([Ft, Vt])}
    vv_svm, vv_rf = VoiceVerifier("svm"), VoiceVerifier("rf")

    systems = {}

    def add(name, modality, factory):
        oof = oof_scores(factory, Xd[modality], yd, splits)
        model = factory().fit(Xd[modality], yd)
        s_test = model.predict_proba(Xt[modality])[:, 1]
        d_eer, d_thr = compute_eer(yd, oof)[:2]
        systems[name] = dict(modality=modality, factory=factory, model=model, oof=oof, test=s_test,
                             thr=finite_or(d_thr), dev_eer=float(d_eer), dev_auc=float(roc_auc_score(yd, oof)))
        print(f"  [{NAMES[name]:22s}] dev EER={d_eer:6.2%}  dev AUC={systems[name]['dev_auc']:.3f}")

    add("face_svm", "face", lambda: build_face_pipeline("svm", n_components=30))
    add("face_knn", "face", lambda: build_face_pipeline("knn", n_components=30))
    add("voice_svm", "voice", lambda: vv_svm._create_base_estimator({}))
    add("voice_rf", "voice", lambda: vv_rf._create_base_estimator({}))
    fusion_grids = {}
    for kind in ("svm", "rf"):
        params, grid = tune_fusion(kind, Xd["fusion"], yd, splits)
        fusion_grids[kind] = {"best": params, "grid": grid}
        print(f"  [fusion {kind}] best params (dev CV): {params}")
        add(f"fusion_{kind}", "fusion", lambda p=params, k=kind: build_fusion_pipeline(k, **p))
    add("fp_svm", "fp", lambda: build_fingerprint_pipeline("svm", n_components=30))
    add("fp_knn", "fp", lambda: build_fingerprint_pipeline("knn", n_components=30))

    # --- stand-alone metrics at the threshold frozen on DEV (EER point) --------------------
    for name, s in systems.items():
        s["metrics"] = evaluate(yt, s["test"], threshold=s["thr"])

    def best_of(names):
        return min(names, key=lambda n: (round(systems[n]["dev_eer"], 6), -systems[n]["dev_auc"]))

    best_face, best_voice = best_of(["face_svm", "face_knn"]), best_of(["voice_svm", "voice_rf"])
    best_fusion, best_fp = best_of(["fusion_svm", "fusion_rf"]), best_of(["fp_svm", "fp_knn"])
    print(f"  Selected on DEV -> face: {best_face}, voice: {best_voice}, fusion: {best_fusion}, fingerprint: {best_fp}")

    # --- Hierarchical pipeline (Mode D) --------------------------------------------------------
    def make_thresholds(t_low_policy="eer"):
        f, fu, fp = systems[best_face], systems[best_fusion], systems[best_fp]
        if t_low_policy == "eer":
            t_low = threshold_at_eer(yd, f["oof"])
        elif t_low_policy == "far1":
            t_low = threshold_at_far(yd, f["oof"], 0.01)
        elif t_low_policy == "lenient":
            t_low = threshold_at_frr(yd, f["oof"], 0.02)
        else:
            raise ValueError(t_low_policy)
        return Thresholds(finite_or(t_low), finite_or(threshold_at_eer(yd, fu["oof"])),
                          finite_or(threshold_at_eer(yd, fp["oof"])), policy=f"T_low:{t_low_policy}, others:eer")

    sf, sfu, sfp = (systems[best_face]["test"], systems[best_fusion]["test"], systems[best_fp]["test"])

    def hier_eval(th):
        lv = cascade_levels(sf, sfu, sfp, th)
        z_d = cascade_score(sf, sfu, None, th)
        z_f = cascade_score(sf, sfu, sfp, th)
        return {"levels": lv,
                "D": evaluate(yt, z_d, threshold=0.5), "D_score": z_d,
                "DFP": evaluate(yt, z_f, threshold=0.5), "DFP_score": z_f}

    th_main = make_thresholds("eer")
    H = hier_eval(th_main)
    print(f"  Frozen thresholds (dev): T_low={th_main.t_low:.3f}  T_fusion={th_main.t_fusion:.3f}  T_fp={th_main.t_fp:.3f}")

    def stage_stats(levels):
        imp, gen = levels[yt == 0], levels[yt == 1]
        n_i, n_g = max(len(imp), 1), max(len(gen), 1)
        return {"imp_stopped_L1": float(np.mean(imp == 0)), "imp_stopped_L2": float(np.mean(imp == 1)),
                "imp_stopped_L3": float(np.mean(imp == 2)), "imp_accepted_L3": float(np.mean(imp == 3)),
                "gen_rejected_L1": float(np.mean(gen == 0)), "gen_rejected_L2": float(np.mean(gen == 1)),
                "gen_rejected_L3": float(np.mean(gen == 2)), "gen_reach_L3": float(np.mean(gen == 3))}

    stats = stage_stats(H["levels"])

    # --- Processing time (measured) hierarchy vs flat ----------------------------------------
    rng = np.random.default_rng(args.seed)

    def sample_idx(pool_idx, k=12):
        return rng.choice(pool_idx, size=min(k, len(pool_idx)), replace=False)

    face_model, fusion_model, fp_model = systems[best_face]["model"], systems[best_fusion]["model"], systems[best_fp]["model"]
    imgs = [cv2.imread(str(pools.face.paths[i])) for i in sample_idx(test.face_idx)]
    wavs = [str(pools.voice.paths[i]) for i in sample_idx(test.voice_idx)]
    fps = [str(pools.fp.paths[i]) for i in sample_idx(test.fp_idx)]

    def t_face_once(img):
        f = crop_face(img)
        x = extract_face_features(f, mode="both") if f is not None else np.zeros(Ft.shape[1])
        face_model.predict_proba(x[None, :])
    fv0 = Ft[0]

    def t_voice_once(w):
        v = extract_voice_features(w)
        fusion_model.predict_proba(np.concatenate([fv0, v])[None, :])

    def t_fp_once(p):
        g = preprocess_fingerprint(p)
        fp_model.predict_proba(extract_fingerprint_features(g, mode="both")[None, :])

    t_face = np.mean([timed(lambda im=im: t_face_once(im))[1] for im in imgs])
    t_voice = np.mean([timed(lambda w=w: t_voice_once(w))[1] for w in wavs])
    t_fp = np.mean([timed(lambda p=p: t_fp_once(p))[1] for p in fps])
    print(f"  Measured compute per attempt: face={t_face * 1000:.0f} ms, voice+fusion={t_voice * 1000:.0f} ms, fingerprint={t_fp * 1000:.0f} ms")

    def times_for(levels, use_fp, cap_v, cap_f):
        imp, gen = levels[yt == 0], levels[yt == 1]
        ti = expected_time(imp, t_face, t_voice, t_fp, use_fp, cap_v, cap_f)
        tg = expected_time(gen, t_face, t_voice, t_fp, use_fp, cap_v, cap_f)
        return {"intruder": ti, "genuine": tg, "mixed_50_50": 0.5 * (ti + tg)}

    flat_v = t_face + t_voice
    flat_all = t_face + t_voice + t_fp
    timing = {
        "measured_ms": {"face": t_face * 1000, "voice_fusion": t_voice * 1000, "fingerprint": t_fp * 1000},
        "compute_only": {
            "flat_face_voice": flat_v, "flat_face_voice_fp": flat_all,
            "hier_D": times_for(H["levels"], False, 0.0, 0.0), "hier_DFP": times_for(H["levels"], True, 0.0, 0.0)},
        "with_capture": {
            "flat_face_voice": flat_v + args.voice_capture_s, "flat_face_voice_fp": flat_all + args.voice_capture_s + args.fp_capture_s,
            "hier_D": times_for(H["levels"], False, args.voice_capture_s, 0.0),
            "hier_DFP": times_for(H["levels"], True, args.voice_capture_s, args.fp_capture_s)},
    }

    # --- T_low policy ablation (E5b) ---------------------------------------------------------
    policy_rows = []
    for pol in ("lenient", "eer", "far1"):
        th = make_thresholds(pol)
        h = hier_eval(th)
        st = stage_stats(h["levels"])
        policy_rows.append({"policy": pol, "T_low": th.t_low, "FAR": h["D"]["FAR"], "FRR": h["D"]["FRR"],
                            "accuracy": h["D"]["accuracy"], "imp_stopped_L1": st["imp_stopped_L1"],
                            "t_intruder_s": times_for(h["levels"], False, args.voice_capture_s, 0.0)["intruder"]})

    # --- E6: late fusion + block weights ------------------------------------------------------
    s_late_oof = 0.5 * (systems[best_face]["oof"] + systems[best_voice]["oof"])
    s_late_test = 0.5 * (systems[best_face]["test"] + systems[best_voice]["test"])
    late_thr = finite_or(threshold_at_eer(yd, s_late_oof))
    late = evaluate(yt, s_late_test, threshold=late_thr)
    weight_rows = {}
    for r in fusion_grids["svm"]["grid"]:
        w = r["params"]["w_voice"]
        if w not in weight_rows or r["auc"] > weight_rows[w]["dev_cv_auc"]:
            weight_rows[w] = {"w_voice": w, "dev_cv_auc": r["auc"], "dev_cv_eer": r["eer"], "params": r["params"]}

    # --- E7: robustness (frozen models + thresholds, perturbed TEST inputs) --------------------
    robustness = None
    if not args.skip_robustness:
        robustness = run_robustness(pools, test, yt, systems, best_face, best_voice, best_fusion, th_main)

    # --- Save production artefacts -----------------------------------------------------------
    for kind in ("svm", "rf"):
        mdl = systems[f"fusion_{kind}"]["model"]
        if args.refit_all:
            mdl = systems[f"fusion_{kind}"]["factory"]().fit(np.vstack([Xd["fusion"], Xt["fusion"]]),
                                                             np.r_[yd, yt])
        save_fusion_model(mdl, user, kind)

    def final(name):
        s = systems[name]
        if not args.refit_all:
            return s["model"]
        return s["factory"]().fit(np.vstack([Xd[s["modality"]], Xt[s["modality"]]]), np.r_[yd, yt])

    save_bundle(GateBundle(user, final(best_face), final(best_fusion), final(best_fp), th_main,
                           meta={"face_model": best_face, "fusion_model": best_fusion, "fp_model": best_fp,
                                 "refit_all": args.refit_all, "dev_rows": len(dev), "test_rows": len(test)}))

    # --- Figures ---------------------------------------------------------------------------------
    figs = OUT / "figures"
    for name, s in systems.items():
        plot_confusion_matrix(confusion_from_metrics(s["metrics"]), ["Impostor", "Genuine"],
                              f"{user}: {NAMES[name]}", figs / f"cm_{user}_{name}.png")
        plt.close("all")
    for key, label in (("D", "Hierarchical (face to fusion)"), ("DFP", "Hierarchical + Fingerprint")):
        plot_confusion_matrix(confusion_from_metrics(H[key]), ["Impostor", "Genuine"], f"{user}: {label}",
                              figs / f"cm_{user}_hier_{key}.png")
        plt.close("all")
    curves = {f"Face ({best_face.split('_')[1].upper()})": (yt, systems[best_face]["test"]),
              f"Voice ({best_voice.split('_')[1].upper()})": (yt, systems[best_voice]["test"]),
              f"Early fusion ({best_fusion.split('_')[1].upper()})": (yt, systems[best_fusion]["test"]),
              "Hierarchical": (yt, H["D_score"]), "Hierarchical + FP": (yt, H["DFP_score"])}
    plot_roc_comparison(curves, f"{user}: ROC on held-out test (session 3 + unseen impostors)", figs / f"roc_{user}.png")
    plt.close("all")
    for key, score, thr, title in (("fusion", systems[best_fusion]["test"], systems[best_fusion]["thr"], "Early fusion"),
                                   ("hier", H["D_score"].clip(0, 1), 0.5, "Hierarchical")):
        plot_score_distributions(score[yt == 1], score[yt == 0], thr, f"{user}: {title} score distribution",
                                 figs / f"dist_{user}_{key}.png")
        plt.close("all")

    return {
        "user": user, "info": {k: v for k, v in info.items() if k in ("n_dev", "n_test", "dev_genuine", "test_genuine")},
        "unseen": [v["name"] for v in info["unseen_vids"]], "seen": [v["name"] for v in info["seen_vids"]],
        "selected": {"face": best_face, "voice": best_voice, "fusion": best_fusion, "fp": best_fp},
        "fusion_params": {k: v["best"] for k, v in fusion_grids.items()},
        "systems": {n: {"metrics": s["metrics"], "dev_eer": s["dev_eer"], "dev_auc": s["dev_auc"], "thr": s["thr"]}
                    for n, s in systems.items()},
        "thresholds": {"T_low": th_main.t_low, "T_fusion": th_main.t_fusion, "T_fp": th_main.t_fp},
        "hier": {"D": H["D"], "DFP": H["DFP"], "stats": stats},
        "late": {"metrics": late, "thr": late_thr}, "weights": list(weight_rows.values()),
        "policy": policy_rows, "timing": timing, "robustness": robustness,
        "_scores": {"y": yt, "face": systems[best_face]["test"], "voice": systems[best_voice]["test"],
                    "fusion": systems[best_fusion]["test"], "D": H["D_score"], "DFP": H["DFP_score"]},
        "_curves_cm": {n: confusion_from_metrics(s["metrics"]) for n, s in systems.items()},
    }


# ---------------------------------------------------------------------------
# E7 robustness
# ---------------------------------------------------------------------------
CONDITIONS = [("nominal", None, None), ("low light (x0.35)", "dark", None), ("over-exposed (x1.8)", "bright", None),
              ("noise 20 dB SNR", None, 20), ("noise 10 dB SNR", None, 10), ("low light + noise 10 dB", "dark", 10)]
_PERT_CACHE = {}


def _face_pert(path, light):
    key = ("f", path, light)
    if key not in _PERT_CACHE:
        img = cv2.imread(path)
        if light == "dark":
            img = np.clip(img.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)
        elif light == "bright":
            img = np.clip(img.astype(np.float32) * 1.8, 0, 255).astype(np.uint8)
        f = crop_face(img)
        _PERT_CACHE[key] = None if f is None else extract_face_features(f, mode="both")
    return _PERT_CACHE[key]


def _voice_pert(path, snr_db):
    key = ("v", path, snr_db)
    if key not in _PERT_CACHE:
        audio, _ = load_audio(path)
        rng = np.random.default_rng(zlib.crc32(path.encode()))
        p_sig = float(np.mean(audio ** 2)) + 1e-12
        noisy = audio + rng.normal(0, np.sqrt(p_sig / 10 ** (snr_db / 10)), size=audio.shape).astype(np.float32)
        _PERT_CACHE[key] = extract_voice_features(noisy)
    return _PERT_CACHE[key]


def run_robustness(pools, test, yt, systems, best_face, best_voice, best_fusion, th):
    out = []
    face_model, voice_model, fusion_model = (systems[best_face]["model"], systems[best_voice]["model"],
                                              systems[best_fusion]["model"])
    for label, light, snr in CONDITIONS:
        fx, f_ok = [], []
        for i in test.face_idx:
            f = pools.face.X[i] if light is None else _face_pert(str(pools.face.paths[i]), light)
            f_ok.append(f is not None)
            fx.append(f if f is not None else np.zeros(pools.face.X.shape[1], dtype=np.float32))
        vx = [pools.voice.X[i] if snr is None else _voice_pert(str(pools.voice.paths[i]), snr) for i in test.voice_idx]
        fx, vx, f_ok = np.array(fx), np.array(vx), np.array(f_ok)
        s_face = np.where(f_ok, face_model.predict_proba(fx)[:, 1], 0.0)   # no face found -> score 0 (fail closed)
        s_voice = voice_model.predict_proba(vx)[:, 1]
        s_fus = np.where(f_ok, fusion_model.predict_proba(np.hstack([fx, vx]))[:, 1], 0.0)
        lv = cascade_levels(s_face, s_fus, None, th)
        row = {"condition": label, "face_not_detected": float(1 - f_ok.mean())}
        for key, s, thr in (("face", s_face, systems[best_face]["thr"]), ("voice", s_voice, systems[best_voice]["thr"]),
                            ("fusion", s_fus, systems[best_fusion]["thr"])):
            m = evaluate(yt, s, threshold=thr)
            row[key] = {"FAR": m["FAR"], "FRR": m["FRR"], "accuracy": m["accuracy"], "AUC": m["AUC"]}
        m = evaluate(yt, cascade_score(s_face, s_fus, None, th), threshold=0.5)
        row["hier"] = {"FAR": m["FAR"], "FRR": m["FRR"], "accuracy": m["accuracy"], "AUC": m["AUC"]}
        out.append(row)
        print(f"    E7 {label:26s} face FRR={row['face']['FRR']:.0%}  voice FRR={row['voice']['FRR']:.0%}  "
              f"fusion FRR/FAR={row['fusion']['FRR']:.0%}/{row['fusion']['FAR']:.0%}  hier FRR/FAR={row['hier']['FRR']:.0%}/{row['hier']['FAR']:.0%}")
    return out


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------
def macro(rows, key_path):
    vals = []
    for r in rows:
        v = r
        for k in key_path:
            v = v[k]
        vals.append(v)
    return float(np.mean(vals)), float(np.std(vals))


def fmt(m, s=None):
    return f"{m * 100:.1f}%" if s is None else f"{m * 100:.1f}% ± {s * 100:.1f}"


def main_table(results):
    lines = ["| System | Model | Accuracy | Precision | Recall | F1 | FAR | FRR | EER | AUC |", "|---|---|---|---|---|---|---|---|---|---|"]
    spec = [("Face-only (Mode A)", "SVM", ("systems", "face_svm", "metrics")), ("Face-only (Mode A)", "KNN", ("systems", "face_knn", "metrics")),
            ("Voice-only (Mode B)", "SVM", ("systems", "voice_svm", "metrics")), ("Voice-only (Mode B)", "RF", ("systems", "voice_rf", "metrics")),
            ("Early Fusion (Mode C)", "SVM", ("systems", "fusion_svm", "metrics")), ("Early Fusion (Mode C)", "RF", ("systems", "fusion_rf", "metrics")),
            ("Hierarchical (Mode D: face→fusion)", "Best pipeline", ("hier", "D")), ("Hierarchical + Fingerprint", "Full system", ("hier", "DFP")),
            ("Fingerprint-only (bonus)", "SVM", ("systems", "fp_svm", "metrics")), ("Fingerprint-only (bonus)", "KNN", ("systems", "fp_knn", "metrics"))]
    csv_rows = []
    for sysname, model, path in spec:
        cells = []
        for c in METRIC_COLS:
            m, s = macro(results, path + (c,))
            cells.append(f"{m * 100:.1f}% ± {s * 100:.1f}" if len(results) > 1 else f"{m * 100:.1f}%")
            csv_rows.append((sysname, model, c, m, s))
        lines.append(f"| {sysname} | {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines), csv_rows


def per_user_table(results):
    lines = ["| User | System | Acc | Prec | Rec | F1 | FAR | FRR | EER | AUC | TP/FP/TN/FN |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        entries = [(NAMES[n], r["systems"][n]["metrics"]) for n in NAMES] + [("Hierarchical (face→fusion)", r["hier"]["D"]),
                                                                              ("Hierarchical + Fingerprint", r["hier"]["DFP"])]
        for name, m in entries:
            lines.append(f"| {r['user']} | {name} | " + " | ".join(f"{m[c] * 100:.1f}%" for c in METRIC_COLS)
                         + f" | {m['tp']}/{m['fp']}/{m['tn']}/{m['fn']} |")
    return "\n".join(lines)


METRIC_EXPLANATION = """| Metric | Definition | Why it matters for biometric verification |
|---|---|---|
| Confusion matrix | counts of TP, FP, TN, FN (genuine = positive) | shows *which* error happens, not just how many |
| Accuracy | (TP+TN)/all | easy overview but misleading when impostor attempts outnumber genuine ones (ours: 5:1), so it is never used alone |
| Precision | TP/(TP+FP) | of everyone the gate let in, how many were really the claimed user |
| Recall = TAR | TP/(TP+FN) | share of genuine attempts that are accepted |
| F1 | harmonic mean of precision and recall | one number that stays honest under class imbalance |
| **FAR** | FP/(FP+TN) | an intruder is accepted: **the dangerous (security) error** |
| **FRR** | FN/(FN+TP) | the real admin is rejected: the usability error |
| ROC / AUC | TAR vs FAR for every possible threshold | compares systems independently of one threshold; AUC = probability a genuine attempt scores above an impostor one |
| **EER** | operating point where FAR = FRR (lower is better) | standard single-number summary for comparing biometric systems; threshold-free |
"""


def write_reports(results, skipped, args):
    OUT.mkdir(parents=True, exist_ok=True)
    table, csv_rows = main_table(results)
    with open(OUT / "m3_main_comparison.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "model", "metric", "mean_over_users", "std_over_users"])
        w.writerows(csv_rows)

    md = []
    md.append("# Member 3: Early Fusion, Hierarchy & Evaluation Results\n")
    md.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  |  users evaluated: {', '.join(r['user'] for r in results)}"
              + (f"  |  **skipped (missing modality data): {', '.join(skipped)}**" if skipped else "") + "\n")
    md.append("## Protocol\n")
    r0 = results[0]
    md.append("- **DEV** (training, tuning, threshold freezing): genuine sessions 1-2 + *seen* impostor identities.\n"
              "- **TEST** (all numbers below): genuine session 3 (raw images only) + *unseen* impostor identities (open set).\n"
              "- Impostor pairs are *virtual identities* (face / voice / fingerprint come from different public sources); every person belongs to exactly one virtual identity, so no person or session crosses DEV/TEST (asserted in code).\n"
              "- Scalers / PCA / block-normalisation sit inside sklearn Pipelines and are fitted on training folds only. Grouped CV = `StratifiedGroupKFold` by session / identity.\n"
              "- Thresholds are frozen on DEV out-of-fold scores (EER point) and applied unchanged on TEST. EER and AUC are threshold-free and computed on TEST scores.\n"
              + "".join(f"- {r['user']}: DEV {r['info']['n_dev']} rows ({r['info']['dev_genuine']} genuine), TEST {r['info']['n_test']} rows ({r['info']['test_genuine']} genuine); unseen impostors: {', '.join(r['unseen'])}\n" for r in results))
    md.append("\n> ⚠ **Small-sample warning.** TEST contains only a handful of genuine attempts per user (see above). One error moves FRR by several points, and EER/AUC of 0% / 100% mean *no error was observed on this small test*, not that the system is perfect. Report them with this caveat.\n")
    md.append("\n## 1. Main comparison table (mean ± std over enrolled users)\n")
    md.append(table + "\n")
    md.append("\n## 2. Per-user results\n")
    md.append(per_user_table(results) + "\n")

    md.append("\n## 3. E4 - Early fusion (Mode C): selected models and tuned parameters\n")
    md.append("| User | Face model | Voice model | Fusion model | Fusion params (tuned on DEV) |\n|---|---|---|---|---|\n")
    for r in results:
        md.append(f"| {r['user']} | {r['selected']['face']} | {r['selected']['voice']} | {r['selected']['fusion']} | `{r['fusion_params'][r['selected']['fusion'].split('_')[1]]}` |\n")

    md.append("\n## 4. E5 - Hierarchical pipeline (Mode D)\n")
    md.append("| User | T_low | T_fusion | T_fp | Impostors stopped L1 / L2 / L3 / accepted | Genuine rejected L1 / L2 / L3 / reach L3 |\n|---|---|---|---|---|---|\n")
    for r in results:
        s, t = r["hier"]["stats"], r["thresholds"]
        md.append(f"| {r['user']} | {t['T_low']:.3f} | {t['T_fusion']:.3f} | {t['T_fp']:.3f} | "
                  f"{s['imp_stopped_L1']:.0%} / {s['imp_stopped_L2']:.0%} / {s['imp_stopped_L3']:.0%} / {s['imp_accepted_L3']:.0%} | "
                  f"{s['gen_rejected_L1']:.0%} / {s['gen_rejected_L2']:.0%} / {s['gen_rejected_L3']:.0%} / {s['gen_reach_L3']:.0%} |\n")
    md.append("\n### 4.1 Processing time: hierarchy vs flat (average per attempt, seconds)\n")
    md.append(f"`compute` = measured feature extraction + inference on this machine; `with capture` adds the human time assumed for speaking the passphrase ({args.voice_capture_s:.1f} s) and presenting a fingerprint ({args.fp_capture_s:.1f} s) - **these two are assumptions, not measurements**.\n\n")
    md.append("| User | Measured ms (face / voice+fusion / fp) | Flat face+voice | Hier. D (intruder / genuine / 50-50) | Flat all 3 | Hier. D+FP (intruder / genuine / 50-50) |\n|---|---|---|---|---|---|\n")
    for r in results:
        t = r["timing"]
        for tag in ("compute_only", "with_capture"):
            md.append(f"| {r['user']} ({'compute' if tag == 'compute_only' else 'with capture'}) | "
                      f"{t['measured_ms']['face']:.0f} / {t['measured_ms']['voice_fusion']:.0f} / {t['measured_ms']['fingerprint']:.0f} | "
                      f"{t[tag]['flat_face_voice']:.3f} | {t[tag]['hier_D']['intruder']:.3f} / {t[tag]['hier_D']['genuine']:.3f} / {t[tag]['hier_D']['mixed_50_50']:.3f} | "
                      f"{t[tag]['flat_face_voice_fp']:.3f} | {t[tag]['hier_DFP']['intruder']:.3f} / {t[tag]['hier_DFP']['genuine']:.3f} / {t[tag]['hier_DFP']['mixed_50_50']:.3f} |\n")
    md.append("\n### 4.2 Level-1 threshold policy ablation (frozen on DEV)\n")
    md.append("| User | T_low policy | T_low | FAR | FRR | Accuracy | Impostors stopped at L1 | Avg intruder time with capture (s) |\n|---|---|---|---|---|---|---|---|\n")
    for r in results:
        for p in r["policy"]:
            md.append(f"| {r['user']} | {p['policy']} | {p['T_low']:.3f} | {p['FAR']:.1%} | {p['FRR']:.1%} | {p['accuracy']:.1%} | {p['imp_stopped_L1']:.0%} | {p['t_intruder_s']:.2f} |\n")
    md.append("\n`lenient` = highest threshold with DEV FRR <= 2% (almost no genuine user rejected early), `eer` = EER point, `far1` = lowest threshold with DEV FAR <= 1%.\n")

    md.append("\n## 5. E6 - Fusion design choices\n")
    md.append("### 5.1 Block weight on the voice block (SVM fusion, DEV grouped-CV, best of the other params)\n")
    md.append("| User | w_voice | CV AUC | CV EER |\n|---|---|---|---|\n")
    for r in results:
        for w in sorted(r["weights"], key=lambda x: x["w_voice"]):
            md.append(f"| {r['user']} | {w['w_voice']} | {w['dev_cv_auc']:.3f} | {w['dev_cv_eer']:.1%} |\n")
    md.append("\n### 5.2 Early fusion vs late fusion (mean of best face score and best voice score)\n")
    md.append("| User | Early fusion FAR / FRR / EER / AUC | Late fusion FAR / FRR / EER / AUC |\n|---|---|---|\n")
    for r in results:
        e, l = r["systems"][r["selected"]["fusion"]]["metrics"], r["late"]["metrics"]
        md.append(f"| {r['user']} | {e['FAR']:.1%} / {e['FRR']:.1%} / {e['EER']:.1%} / {e['AUC']:.3f} | {l['FAR']:.1%} / {l['FRR']:.1%} / {l['EER']:.1%} / {l['AUC']:.3f} |\n")

    if all(r["robustness"] for r in results):
        md.append("\n## 6. E7 - Robustness (frozen models and thresholds, perturbed TEST inputs)\n")
        md.append("Low light = pixel values x0.35, over-exposed = x1.8 (before Haar detection + CLAHE); noise = white Gaussian noise at the given SNR added to the passphrase. Face not detected => score 0 (fail closed). Mean over users.\n\n")
        md.append("| Condition | Face not detected | Face FAR / FRR | Voice FAR / FRR | Fusion FAR / FRR | Hierarchical FAR / FRR |\n|---|---|---|---|---|---|\n")
        for i, (label, _, _) in enumerate(CONDITIONS):
            def cell(k):
                far = np.mean([r["robustness"][i][k]["FAR"] for r in results]); frr = np.mean([r["robustness"][i][k]["FRR"] for r in results])
                return f"{far:.1%} / {frr:.1%}"
            nd = np.mean([r["robustness"][i]["face_not_detected"] for r in results])
            md.append(f"| {label} | {nd:.0%} | {cell('face')} | {cell('voice')} | {cell('fusion')} | {cell('hier')} |\n")

    md.append("\n## 7. Metrics: what they mean and why they suit biometrics\n")
    md.append(METRIC_EXPLANATION)
    md.append("\n## 8. Figures\n`results/m3/figures/`: `roc_<user>.png`, `roc_pooled.png`, `cm_<user>_<system>.png`, `cm_pooled_*.png`, `dist_<user>_*.png`.\n")
    (OUT / "m3_results_summary.md").write_text("".join(md), encoding="utf-8")

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if not k.startswith("_")}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    (OUT / "m3_results.json").write_text(json.dumps(clean(results), indent=2), encoding="utf-8")
    return table


def pooled_figures(results):
    figs = OUT / "figures"
    y = np.concatenate([r["_scores"]["y"] for r in results])
    curves = {"Face-only": (y, np.concatenate([r["_scores"]["face"] for r in results])),
              "Voice-only": (y, np.concatenate([r["_scores"]["voice"] for r in results])),
              "Early fusion": (y, np.concatenate([r["_scores"]["fusion"] for r in results])),
              "Hierarchical": (y, np.concatenate([r["_scores"]["D"] for r in results])),
              "Hierarchical + Fingerprint": (y, np.concatenate([r["_scores"]["DFP"] for r in results]))}
    plot_roc_comparison(curves, "Pooled ROC over enrolled users (held-out test)", figs / "roc_pooled.png")
    plt.close("all")
    for name in ("face_svm", "voice_rf", "fusion_svm", "fusion_rf"):
        cm = sum(r["_curves_cm"][name] for r in results)
        plot_confusion_matrix(cm, ["Impostor", "Genuine"], f"Pooled: {NAMES[name]}", figs / f"cm_pooled_{name}.png")
        plt.close("all")
    for key, label in (("D", "Hierarchical"), ("DFP", "Hierarchical + FP")):
        cm = sum(confusion_from_metrics(r["hier"][key]) for r in results)
        plot_confusion_matrix(cm, ["Impostor", "Genuine"], f"Pooled: {label}", figs / f"cm_pooled_hier_{key}.png")
        plt.close("all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", nargs="*", default=None, help="default: every member with face+voice+fingerprint data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rebuild-cache", action="store_true", help="re-extract features from raw files")
    ap.add_argument("--refit-all", action="store_true", help="refit saved production models on DEV+TEST (thresholds stay frozen)")
    ap.add_argument("--skip-robustness", action="store_true")
    ap.add_argument("--voice-capture-s", type=float, default=3.0, help="assumed time to speak the passphrase")
    ap.add_argument("--fp-capture-s", type=float, default=2.0, help="assumed time to present/upload a fingerprint")
    args = ap.parse_args()

    print("Loading / extracting features (cached in cache/) ...")
    pools = load_pools(rebuild=args.rebuild_cache)
    usable, skipped = users_with_all_modalities(pools)
    users = [u for u in (args.users or usable) if u in usable]
    for u in skipped:
        print(f"[SKIP] {u}: missing data in at least one modality (face / voice / fingerprint) -> cannot build Modes C/D for this user.")
    if not users:
        raise SystemExit("No user has face + voice + fingerprint data.")

    results = [run_user(u, pools, args) for u in users]
    pooled_figures(results)
    table = write_reports(results, skipped, args)
    print("\n" + "=" * 78 + "\nMAIN COMPARISON TABLE (mean ± std over users, held-out test)\n" + "=" * 78)
    print(table)
    print(f"\nFull report: {OUT / 'm3_results_summary.md'}")


if __name__ == "__main__":
    main()
