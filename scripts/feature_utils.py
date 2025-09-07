import numpy as np
import pandas as pd

def realized_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """Rolling standard deviation as realized volatility."""
    return returns.rolling(window).std()

def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 21) -> pd.Series:
    """Parkinson volatility based on high-low price range."""
    factor = 1 / (4 * np.log(2))
    return (factor * (np.log(high / low) ** 2).rolling(window).mean()) ** 0.5

def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 21) -> pd.Series:
    """Computes ATR (Average True Range)."""
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def z_score(series: pd.Series, window: int = 21) -> pd.Series:
    """Rolling Z-score."""
    return (series - series.rolling(window).mean()) / series.rolling(window).std()


from scipy.stats import skew, kurtosis
import pandas as pd

def rolling_skew(series: pd.Series, window: int = 21) -> pd.Series:
    """Rolling skewness."""
    return series.rolling(window).apply(lambda x: skew(x), raw=True)

def rolling_kurtosis(series: pd.Series, window: int = 21) -> pd.Series:
    """Rolling kurtosis."""
    return series.rolling(window).apply(lambda x: kurtosis(x), raw=True)
