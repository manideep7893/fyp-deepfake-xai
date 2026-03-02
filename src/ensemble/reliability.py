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

def compute_reliability_and_fusion(
    A_mean: float,
    B_mean: float,
    C_mean: float,
    alignment_AB: float | None = None,
    agree_AB: bool | None = None,
) -> ReliabilityResult:
    """
    Reliability-aware fusion (no training).
    - C is the strongest model → primary signal
    - A/B provide a consistency signal (agreement + alignment)
    Returns: p_final, pred_final, reliability [0..1], scenario label
    """

    # --- sanitize NaNs ---
    def safe(x, default):
        try:
            if x is None: 
                return default
            x = float(x)
            if np.isnan(x):
                return default
            return x
        except Exception:
            return default

    A = safe(A_mean, 0.5)
    B = safe(B_mean, 0.5)
    C = safe(C_mean, 0.5)

    align = safe(alignment_AB, 0.5)     # expected in [0..1]
    agree = bool(agree_AB) if agree_AB is not None else False

    # --- scenario logic (S1..S4) ---
    # We use which model is strongest + A/B alignment as scenario explanation
    # (You can refine later once Rifaie confirms scenario meaning.)
    vals = {"M1(A)": A, "M2(B)": B, "M3(C)": C}
    top_model = max(vals, key=vals.get)

    if top_model == "M3(C)" and agree and align >= 0.8:
        scenario = "S3: strong-consensus (C dominant, A/B agree)"
    elif top_model == "M3(C)" and (not agree or align < 0.8):
        scenario = "S2: C-dominant but weak-consensus (check XAI)"
    elif top_model != "M3(C)" and agree:
        scenario = "S1: A/B consensus overrides (possible domain shift)"
    else:
        scenario = "S4: disagreement/uncertain (needs review)"

    # --- reliability score ---
    # Reliability = agreement signal + alignment signal + C confidence signal
    # C confidence: how far from 0.5
    c_conf = _clip01(abs(C - 0.5) * 2.0)      # 0 at 0.5, 1 at 0 or 1
    agree_score = 1.0 if agree else 0.0

    # Weighted reliability (tunable)
    reliability = _clip01(0.50 * c_conf + 0.30 * align + 0.20 * agree_score)

    # --- reliability-weighted fusion ---
    # Base is C. If reliability is low, blend towards average(A,B,C) to reduce overconfidence.
    mean_ABC = (A + B + C) / 3.0
    p_final = _clip01(reliability * C + (1.0 - reliability) * mean_ABC)

    pred_final = 1 if p_final >= 0.5 else 0

    return ReliabilityResult(
        p_final=p_final,
        pred_final=pred_final,
        reliability=reliability,
        scenario=scenario
    )