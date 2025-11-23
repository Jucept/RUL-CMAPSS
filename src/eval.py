# eval.py
import numpy as np

def score_phm08(y_true: np.ndarray, y_pred: np.ndarray, a1: float=10.0, a2: float=13.0) -> float:
    d = y_pred - y_true
    # vectorized stable computation
    loss = np.where(d < 0, np.exp(-d / a1) - 1.0, np.exp(d / a2) - 1.0)
    return float(np.sum(loss))

if __name__ == "__main__":
    # small synthetic checks
    import numpy as np
    d_early = -5.0
    d_late = 5.0
    # compute expected
    s_early = np.exp(-d_early / 10.0) - 1.0
    s_late = np.exp(d_late / 13.0) - 1.0
    print("s_early:", s_early, "s_late:", s_late, "s_late < s_early?", s_late < s_early)