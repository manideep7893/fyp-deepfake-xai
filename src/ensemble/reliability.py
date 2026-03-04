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
    D_mean: float | None = None,          # ✅ NEW (M4 FFT)
    alignment_AB: float | None = None,
    agree_AB: bool | None = None,
) -> ReliabilityResult:
    """
    Reliability-aware, scenario-aware fusion (no training).
    - C is the strongest model → primary signal
    - A/B provide consistency signal (agreement + alignment)
    - D (FFT) supports decisions mainly under uncertainty/disagreement
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
    D = safe(D_mean, 0.5)                 # ✅ NEW

    align = safe(alignment_AB, 0.5)       # expected in [0..1]
    agree = bool(agree_AB) if agree_AB is not None else False

    # --- scenario logic (S1..S4) ---
    vals = {"M1(A)": A, "M2(B)": B, "M3(C)": C, "M4(D)": D}
    top_model = max(vals, key=vals.get)

    if top_model == "M3(C)" and agree and align >= 0.8:
        scenario = "S3: strong-consensus (C dominant, A/B agree)"
    elif top_model == "M3(C)" and (not agree or align < 0.8):
        scenario = "S2: C-dominant but weak-consensus (check XAI + FFT)"
    elif top_model != "M3(C)" and agree:
        scenario = "S1: A/B consensus overrides (possible domain shift)"
    else:
        scenario = "S4: disagreement/uncertain (needs review)"

    # --- reliability score ---
    c_conf = _clip01(abs(C - 0.5) * 2.0)      # 0 at 0.5, 1 at 0 or 1
    agree_score = 1.0 if agree else 0.0

    # Add D confidence lightly (FFT should not dominate)
    d_conf = _clip01(abs(D - 0.5) * 2.0)

    reliability = _clip01(
        0.45 * c_conf +
        0.25 * align +
        0.15 * agree_score +
        0.15 * d_conf
    )

    # --- scenario-aware fusion ---
    mean_ABC = (A + B + C) / 3.0
    mean_ABCD = (A + B + C + D) / 4.0

    # Base decision depends on scenario:
    # S3: trust C
    # S2: trust C but let FFT help a bit
    # S4: increase FFT influence (uncertainty)
    # S1: if A/B override, keep your conservative blend (FFT low weight)
    if scenario.startswith("S3"):
        p_base = C
    elif scenario.startswith("S2"):
        p_base = _clip01(0.75 * C + 0.25 * D)
    elif scenario.startswith("S4"):
        p_base = _clip01(0.60 * C + 0.40 * D)
    else:  # S1
        p_base = _clip01(0.70 * mean_ABC + 0.30 * D)

    # Reliability shrinkage towards a safe mean to avoid overconfidence
    p_final = _clip01(reliability * p_base + (1.0 - reliability) * mean_ABCD)

    pred_final = 1 if p_final >= 0.5 else 0

    return ReliabilityResult(
        p_final=p_final,
        pred_final=pred_final,
        reliability=reliability,
        scenario=scenario
    )