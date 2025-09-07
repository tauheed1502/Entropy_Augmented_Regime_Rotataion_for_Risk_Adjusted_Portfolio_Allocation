import os
os.system('pip install antropy')

import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy
from antropy import perm_entropy, sample_entropy


def rolling_shannon_entropy(series: pd.Series, window: int = 21, bins: int = 10) -> pd.Series:
    """
    Computes Shannon entropy over a rolling window using histogram binning.

    Parameters:
    - series: pd.Series of returns or price
    - window: int, size of rolling window
    - bins: int, number of bins for histogram

    Returns:
    - pd.Series of rolling entropy values
    """
    def _entropy_window(x):
        hist, _ = np.histogram(x, bins=bins, density=True)
        hist = hist[hist > 0]  # remove zero entries
        return -np.sum(hist * np.log(hist))

    return series.rolling(window).apply(_entropy_window, raw=True)


from antropy import perm_entropy
import pandas as pd
import numpy as np

def rolling_permutation_entropy(series: pd.Series, window: int = 21, order: int = 3) -> pd.Series:
    """
    Computes permutation entropy over a rolling window.

    Parameters:
    - series: pd.Series
    - window: int: Rolling window length
    - order: int: Permutation entropy embedding dimension

    Returns:
    - pd.Series: Rolling permutation entropy
    """
    def safe_perm_entropy(x):
        try:
            return perm_entropy(x, order=order, normalize=True)
        except:
            return np.nan

    return series.rolling(window).apply(safe_perm_entropy, raw=True)



# def rolling_sample_entropy(series: pd.Series, window: int = 21) -> pd.Series:
#     """
#     Computes sample entropy over a rolling window.

#     Returns:
#     - pd.Series
#     """
#     return series.rolling(window).apply(lambda x: sample_entropy(x), raw=True)

def rolling_sample_entropy(series: pd.Series, window: int = 50, m: int = 2, r: float = 0.2) -> pd.Series:
    """
    Computes Sample Entropy over a rolling window for a pandas Series.

    Parameters:
    - series: pd.Series of float values (e.g., returns)
    - window: int, rolling window size (should be >= 50 for reliable results)
    - m: int, embedding dimension (default = 2)
    - r: float, tolerance multiplier (default = 0.2)

    Returns:
    - pd.Series of sample entropy values (NaN where undefined)
    """
    import numpy as np

    def sample_entropy(x):
        x = np.asarray(x)
        N = len(x)
        if N <= m + 1 or np.std(x) == 0:
            return np.nan

        r_scaled = r * np.std(x)

        def count_matches(template_len):
            count = 0
            for i in range(N - template_len):
                for j in range(i + 1, N - template_len):
                    if np.all(np.abs(x[i:i+template_len] - x[j:j+template_len]) <= r_scaled):
                        count += 1
            return count

        A = count_matches(m + 1)
        B = count_matches(m)

        if B == 0 or A == 0:
            return np.nan
        return -np.log(A / B)

    return series.rolling(window=window).apply(sample_entropy, raw=True)
