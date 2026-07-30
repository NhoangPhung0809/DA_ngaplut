from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChronosForecastResult:
    context_dates: list[pd.Timestamp]
    context_values: list[float]
    forecast_dates: list[pd.Timestamp]
    forecast_values: list[float]
    forecast_labels: list[int]
    thresholds: dict[str, float]
    model_id: str


def _to_daily_series(
    historical_data: pd.DataFrame,
    value_column: str,
    time_column: str = "Thời_gian",
) -> pd.Series:
    if historical_data is None or historical_data.empty:
        raise ValueError("historical_data is empty.")
    if time_column not in historical_data.columns:
        raise ValueError(f"Missing time column: {time_column}")
    if value_column not in historical_data.columns:
        raise ValueError(f"Missing value column: {value_column}")

    df = historical_data[[time_column, value_column]].copy()
    df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
    df = df.dropna(subset=[time_column]).sort_values(time_column)
    df["__date__"] = df[time_column].dt.floor("D")
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce").fillna(0.0)
    daily = df.groupby("__date__", as_index=True)[value_column].sum().sort_index()
    daily.index = pd.to_datetime(daily.index)
    return daily


def _ensure_context_length(values: np.ndarray, context_length: int) -> np.ndarray:
    if len(values) == 0:
        raise ValueError("No values available for context.")
    if len(values) >= context_length:
        return values[-context_length:]
    pad_count = context_length - len(values)
    pad_value = float(values[0])
    pad = np.full((pad_count,), pad_value, dtype=np.float32)
    return np.concatenate([pad, values], axis=0).astype(np.float32)


def _labels_from_thresholds(values: np.ndarray, light_threshold: float, heavy_threshold: float) -> list[int]:
    labels: list[int] = []
    for v in values.tolist():
        if v >= heavy_threshold:
            labels.append(2)
        elif v >= light_threshold:
            labels.append(1)
        else:
            labels.append(0)
    return labels


@lru_cache(maxsize=1)
def _load_chronos_pipeline(model_id: str):
    import torch

    try:
        from chronos import ChronosPipeline
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "Missing dependency for Chronos. Install `chronos-forecasting`."
        ) from exc

    torch.set_num_threads(max(1, min(4, int(torch.get_num_threads()))))
    pipeline = ChronosPipeline.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    return pipeline


def run_chronos_forecast(
    historical_data: pd.DataFrame,
    prediction_length: int = 7,
    context_length: int = 30,
    value_column: str = "Lượng_mưa_mm",
    time_column: str = "Thời_gian",
    model_id: str = "amazon/chronos-t5-mini",
    light_threshold: float = 25.0,
    heavy_threshold: float = 50.0,
    num_samples: int = 20,
) -> ChronosForecastResult:
    import torch

    daily_series = _to_daily_series(historical_data, value_column=value_column, time_column=time_column)
    if len(daily_series) < 2:
        raise ValueError("Not enough daily history to run Chronos forecast.")

    context_dates = daily_series.index[-context_length:].tolist()
    context_values_raw = daily_series.to_numpy(dtype=np.float32)
    context_values = _ensure_context_length(context_values_raw, context_length=context_length)

    pipeline = _load_chronos_pipeline(model_id)
    with torch.no_grad():
        context_tensor = torch.tensor(context_values, dtype=torch.float32)
        samples = pipeline.predict(
            context_tensor,
            prediction_length=prediction_length,
            num_samples=num_samples,
        )
        if hasattr(samples, "detach"):
            samples_np = samples.detach().cpu().numpy()
        else:
            samples_np = np.asarray(samples)

    if samples_np.ndim == 1:
        forecast_values = samples_np.astype(np.float32)
    else:
        forecast_values = np.median(samples_np, axis=0).astype(np.float32)

    last_date = pd.to_datetime(daily_series.index.max())
    forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=prediction_length, freq="D").tolist()
    forecast_labels = _labels_from_thresholds(
        forecast_values,
        light_threshold=light_threshold,
        heavy_threshold=heavy_threshold,
    )

    return ChronosForecastResult(
        context_dates=[pd.to_datetime(d) for d in context_dates][-context_length:],
        context_values=[float(v) for v in context_values.tolist()],
        forecast_dates=[pd.to_datetime(d) for d in forecast_dates],
        forecast_values=[float(v) for v in forecast_values.tolist()],
        forecast_labels=forecast_labels,
        thresholds={"light": float(light_threshold), "heavy": float(heavy_threshold)},
        model_id=model_id,
    )


def chronos_result_to_plotly_frame(result: ChronosForecastResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dt, val in zip(result.context_dates, result.context_values):
        rows.append({"date": dt, "value": float(val), "segment": "history"})
    for dt, val, label in zip(result.forecast_dates, result.forecast_values, result.forecast_labels):
        rows.append({"date": dt, "value": float(val), "segment": "forecast", "label": int(label)})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

