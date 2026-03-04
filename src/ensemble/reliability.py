# src/ensemble/reliability.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class ReliabilityResult:
    p_final: float
    pred_final: int
    reliability: float
    scenario: str

def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def _safe_float(x, default=0.5) -> float:
    try:
        if x is None:
            return default
        x = float(x)
        if np.isnan(x):
            return default
        return x
    except Exception:
        return default

def compute_reliability_and_fusion(
    A_mean: float,
    B_mean: float,
    C_mean: float,
    E_mean: float | None = None,          # DCT frequency model
    alignment_AB: float | None = None,
    agree_AB: bool | None = None,
) -> ReliabilityResult:
    """
    Scenario-aware reliability fusion.
    - C (CNN) is strong primary
    - A/B provide consistency (agreement + alignment)
    - E (DCT frequency) provides orthogonal evidence (frequency artifacts)

    Returns:
      p_final, pred_final, reliability [0..1], scenario label
    """

    A = _safe_float(A_mean, 0.5)
    B = _safe_float(B_mean, 0.5)
    C = _safe_float(C_mean, 0.5)
    E = _safe_float(E_mean, 0.5) if E_mean is not None else 0.5

    align = _safe_float(alignment_AB, 0.5)   # expected [0..1]
    agree = bool(agree_AB) if agree_AB is not None else False

    # --- scenario logic ---
    # Main idea:
    #  S3: strong-consensus -> C strong + A/B agree + E supports
    #  S2: C strong but weak-consensus -> check explanations (Grad-CAM + DCT)
    #  S1: A/B override -> possible domain shift (C uncertain)
    #  S4: general disagreement/uncertainty

    # "support" = model on same side of 0.5 as C
    def same_side(x, y):
        return (x >= 0.5) == (y >= 0.5)

    E_supports_C = same_side(E, C)

    vals = {"M1(A)": A, "M2(B)": B, "M3(C)": C, "M4(E-DCT)": E}
    top_model = max(vals, key=vals.get)

    c_conf = _clip01(abs(C - 0.5) * 2.0)     # 0 at 0.5, 1 at extremes
    e_conf = _clip01(abs(E - 0.5) * 2.0)

    if c_conf >= 0.7 and agree and align >= 0.8 and E_supports_C and e_conf >= 0.4:
        scenario = "S3: strong-consensus (C + AB + DCT agree)"
    elif c_conf >= 0.7 and (not agree or align < 0.8 or not E_supports_C):
        scenario = "S2: C-strong but weak-consensus (check Grad-CAM + DCT)"
    elif c_conf < 0.55 and agree and align >= 0.7:
        scenario = "S1: A/B consensus overrides (possible domain shift)"
    else:
        scenario = "S4: disagreement/uncertain"

    # --- reliability score ---
    agree_score = 1.0 if agree else 0.0
    support_score = 1.0 if E_supports_C else 0.0

    reliability = _clip01(
        0.45 * c_conf +
        0.25 * align +
        0.15 * agree_score +
        0.15 * (0.5 * e_conf + 0.5 * support_score)
    )

    # --- fusion ---
    # Use C as anchor, reinforced by DCT when it supports.
    mean_ABCE = (A + B + C + E) / 4.0
    # If DCT supports C, slightly pull towards (C+E)/2
    ce_mean = (C + E) / 2.0 if E_mean is not None else C

    p_anchor = 0.7 * C + 0.3 * ce_mean
    p_final = _clip01(reliability * p_anchor + (1.0 - reliability) * mean_ABCE)

    pred_final = 1 if p_final >= 0.5 else 0

    return ReliabilityResult(
        p_final=p_final,
        pred_final=pred_final,
        reliability=reliability,
        scenario=scenario
    )