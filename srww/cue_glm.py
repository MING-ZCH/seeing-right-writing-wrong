"""Probit GLM of the cue-factorial answers: says-subject-front ~ phrasing + occlusion + vertical (+ size) codes."""
from __future__ import annotations

import re

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from srww import records as BB
from srww.records import pe

QRX = re.compile(r"Is the (\w+ \w+) (in front of|behind) the (\w+ \w+)\?")


def geom_meta(mt: dict, x: str) -> dict:
    front, back = mt["front_name"].lower(), mt["back_name"].lower()
    cyf = (mt["bbox_front"][1] + mt["bbox_front"][3]) / 2
    cyb = (mt["bbox_back"][1] + mt["bbox_back"][3]) / 2
    lower = front if cyf > cyb else back
    larger = front if mt["area_front_px"] > mt["area_back_px"] else back
    return {"front": front, "lower": lower, "upper": back if lower == front else front, "larger": larger,
            "x_front": x == front, "x_lower": x == lower, "x_larger": x == larger}


def probit_fit(X: np.ndarray, y: np.ndarray, ridge: float = 0.1) -> np.ndarray:
    """Probit MLE with a weak N(0, 1/ridge) prior on non-intercept terms (finite under quasi-separation)."""
    def nll(b):
        z = X @ b
        return -(y * norm.logcdf(z) + (1 - y) * norm.logcdf(-z)).sum() + 0.5 * ridge * (b[1:] ** 2).sum()

    def grad(b):
        z = X @ b
        lam1 = np.exp(norm.logpdf(z) - norm.logcdf(z))
        lam0 = np.exp(norm.logpdf(z) - norm.logcdf(-z))
        g = -(X.T @ (y * lam1 - (1 - y) * lam0))
        g[1:] += ridge * b[1:]
        return g
    return minimize(nll, np.zeros(X.shape[1]), jac=grad, method="BFGS", options={"gtol": 1e-7, "maxiter": 500}).x


def resp_prob(r: dict) -> float:
    p = BB.probs(r.get("lp_sum") or r["top20"])
    return p.get("A", 0.0)


def glm_rows(recs: list[dict], use_size: bool) -> dict:
    """Arrays for the probit: says subject front; phrasing, occlusion, vertical, size codes (+-1); pair; score."""
    rows = []
    for r in recs:
        qm = QRX.search(r["question"]) if "question" in r else None
        mt = r["meta"]
        x = (qm.group(1) if qm else mt["x_name"]).lower()
        g = geom_meta(mt, x)
        behind = (r.get("phrasing") == "behind") or (qm is not None and qm.group(2) == "behind")
        pa = resp_prob(r)
        says = (pe(r) == "A") != behind
        score = (pa if not behind else 1 - pa)
        rows.append((float(says), -1.0 if behind else 1.0, 1.0 if g["x_front"] else -1.0, 1.0 if g["x_lower"] else -1.0,
                     1.0 if g["x_larger"] else -1.0, r["image_id"], score))
    a = np.array([x[:5] for x in rows])
    cols = [np.ones(len(a)), a[:, 1], a[:, 2], a[:, 3]] + ([a[:, 4]] if use_size else [])
    return {"y": a[:, 0], "X": np.column_stack(cols), "pair": [x[5] for x in rows], "score": np.array([x[6] for x in rows]),
            "truth": a[:, 2] > 0}
