"""Atributos com relógio explícito: nunca deslocar por linha sem grade horária."""
import numpy as np
import pandas as pd


def historical_features(frame, wind="vento_ms", gust="rajada_ms", delay_hours=1):
    """Índice horário único. t-1 significa hora de medição, não latência garantida."""
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.has_duplicates:
        raise ValueError("É necessário um índice temporal único")
    if not isinstance(delay_hours, int) or delay_hours < 1:
        raise ValueError("A defasagem mínima deve ser um inteiro positivo")
    if not frame.index.equals(frame.index.floor('h')):
        raise ValueError("As medições devem estar em horas inteiras")
    grid = pd.date_range(frame.index.min(), frame.index.max(), freq="h")
    x = frame.reindex(grid).copy()
    out = pd.DataFrame(index=grid)
    for lag in [1, 3, 6, 24]:
        out[f"wind_lag{lag}"] = x[wind].shift(lag + delay_hours - 1)
    for lag in [1, 24]:
        out[f"gust_lag{lag}"] = x[gust].shift(lag + delay_hours - 1)
    out["wind_mean24"] = x[wind].shift(delay_hours).rolling(24, min_periods=24).mean()
    out["gust_max24"] = x[gust].shift(delay_hours).rolling(24, min_periods=24).max()
    return out


def calendar_features(index):
    out = pd.DataFrame(index=index)
    for name, values, period in [("hour", index.hour, 24), ("weekday", index.dayofweek, 7), ("month", index.month-1, 12)]:
        out[name] = values
        out[name+"_sin"] = np.sin(2*np.pi*values/period)
        out[name+"_cos"] = np.cos(2*np.pi*values/period)
    out["weekend"] = (index.dayofweek >= 5).astype(int)
    return out
