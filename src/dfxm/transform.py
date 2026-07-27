import numpy as np

def adaptive_log(img) -> np.ndarray:
    c = 1.0 / np.log1p(np.max(img))
    return c * np.log1p(img)