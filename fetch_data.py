import math
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
HISTORICAL_DIR = BASE_DIR / "data" / "historical"
CACHE_DIR = BASE_DIR / "cache"
INCREMENTAL_NEW_ROWS_PATH = CACHE_DIR / "new_historical_rows.csv"
TRAINING_WORKER_PATH = BASE_DIR / "training_worker.py"

# Danh sách các địa điểm cần tải dữ liệu.
LOCATIONS = [
    {"name": "TP_Hue", "lat": 16.4637, "lon": 107.5909, "coast_lat": 16.4300, "coast_lon": 107.7600},
    {"name": "Huong_Thuy", "lat": 16.3382, "lon": 107.6742, "coast_lat": 16.3300, "coast_lon": 107.7800},
    {"name": "Phu_Vang", "lat": 16.4706, "lon": 107.7148, "coast_lat": 16.4400, "coast_lon": 107.8400},
    {"name": "Huong_Tra", "lat": 16.5181, "lon": 107.4747, "coast_lat": 16.5200, "coast_lon": 107.6200},
    {"name": "Quang_Dien", "lat": 16.5798, "lon": 107.4930, "coast_lat": 16.6100, "coast_lon": 107.5600},
]

DEFAULT_LOOKBACK_DAYS = 365 * 10
API_DATE_FORMAT = "%Y-%m-%d"

HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def format_api_date(dt_value: datetime) -> str:
    return dt_value.strftime(API_DATE_FORMAT)


def get_last_csv_timestamp(save_path: Path) -> pd.Timestamp | None:
    if not save_path.exists():
        return None
    try:
        existing_times = pd.read_csv(save_path, usecols=["Thời_gian"], encoding="utf-8-sig")
    except Exception as exc:
        raise ValueError(f"API Error: Failed to read existing CSV {save_path.resolve()}: {exc}") from exc

    existing_times["Thời_gian"] = pd.to_datetime(existing_times["Thời_gian"], errors="coerce")
    existing_times = existing_times.dropna(subset=["Thời_gian"]).reset_index(drop=True)
    if existing_times.empty:
        return None
    return pd.Timestamp(existing_times["Thời_gian"].iloc[-1])


def determine_date_window(save_path: Path) -> tuple[str, str]:
    end_date = datetime.now(timezone.utc).date() - timedelta(days=1)
    start_date = end_date - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    last_timestamp = get_last_csv_timestamp(save_path)
    if last_timestamp is not None:
        start_date = last_timestamp.date() + timedelta(days=1)
        print(f"[DEBUG] Last timestamp from final CSV row: {last_timestamp}")

    start_date_str = pd.Timestamp(start_date).strftime(API_DATE_FORMAT)
    end_date_str = pd.Timestamp(end_date).strftime(API_DATE_FORMAT)
    if start_date_str > end_date_str:
        start_date_str = end_date_str
    return start_date_str, end_date_str


def fetch_openmeteo_weather(lat, lon, start_date_str: str, end_date_str: str):
    """Tải dữ liệu thời tiết lịch sử từ Open-Meteo."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "hourly": "temperature_2m,relative_humidity_2m,rain,soil_moisture_0_to_7cm",
        "timezone": "auto",
    }
    try:
        response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=30)
    except requests.RequestException as exc:
        raise ValueError(f"API Error: {exc}") from exc
    print(f"API Status Code: {response.status_code}")
    print(f"Raw API Response: {response.text[:200]}")
    if response.status_code != 200:
        raise ValueError(f"API Error: {response.text}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError(f"API Error: {response.text}") from exc
    hourly = payload.get("hourly")
    if not hourly:
        raise ValueError(f"API Error: {response.text}")

    time_values = hourly.get("time", [])
    df = pd.DataFrame(
        {
            "Thời_gian": pd.to_datetime(time_values, errors="coerce"),
            "Nhiệt_độ_C": hourly.get("temperature_2m", []),
            "Độ_ẩm_%": hourly.get("relative_humidity_2m", []),
            "Lượng_mưa_mm": hourly.get("rain", []),
            "Độ_ẩm_đất": hourly.get("soil_moisture_0_to_7cm", []),
        }
    )
    return df


def fetch_openmeteo_tide(lat, lon, start_date_str: str, end_date_str: str):
    """Tải dữ liệu biển dùng làm xấp xỉ cho chiều cao triều."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "hourly": "wave_height",
        "timezone": "auto",
    }
    try:
        response = requests.get("https://marine-api.open-meteo.com/v1/marine", params=params, timeout=30)
    except requests.RequestException as exc:
        raise ValueError(f"API Error: {exc}") from exc
    print(f"API Status Code: {response.status_code}")
    print(f"Raw API Response: {response.text[:200]}")
    if response.status_code != 200:
        raise ValueError(f"API Error: {response.text}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError(f"API Error: {response.text}") from exc
    hourly = payload.get("hourly")
    if not hourly:
        raise ValueError(f"API Error: {response.text}")

    df = pd.DataFrame(
        {
            "Thời_gian": pd.to_datetime(hourly.get("time", []), errors="coerce"),
            "Chiều_cao_triều_m": hourly.get("wave_height", []),
        }
    )
    df["Chiều_cao_triều_m"] = pd.to_numeric(df["Chiều_cao_triều_m"], errors="coerce").clip(lower=0.0, upper=5.0)
    return df


def calculate_synthetic_tide(dates):
    """Sinh chuỗi triều tổng hợp theo chu kỳ bán nhật triều và chu kỳ mặt trăng."""
    def tide_height(current_time):
        lunar_cycle_days = 29.53
        semi_daily_hours = 12.42
        elapsed_seconds = (current_time - datetime(2000, 1, 1)).total_seconds()
        lunar_phase = 2 * math.pi * elapsed_seconds / (lunar_cycle_days * 86400)
        semi_daily_phase = 2 * math.pi * elapsed_seconds / (semi_daily_hours * 3600)
        value = 1.0 + 0.5 * math.sin(lunar_phase) + 0.8 * math.sin(semi_daily_phase)
        return max(0.1, min(4.0, value))

    tide_df = pd.DataFrame({"Thời_gian": dates})
    tide_df["Chiều_cao_triều_m"] = tide_df["Thời_gian"].map(tide_height)
    return tide_df


def add_rule_based_label(df):
    """Tạo nhãn nguy cơ ngập theo luật chuyên gia."""
    labeled = df.copy()
    rain = labeled["Lượng_mưa_mm"].fillna(0)
    soil_moisture = labeled["Độ_ẩm_đất"].fillna(0)
    tide = labeled["Chiều_cao_triều_m"].fillna(0)

    labeled["Nguy_cơ_ngập"] = (
        (rain > 25)
        | ((rain > 15) & (soil_moisture > 0.3))
        | (tide > 2.5)
    ).astype(int)
    return labeled


def finalize_location_dataframe(weather_df, tide_df):
    """Gộp và làm sạch dữ liệu của một địa điểm."""
    final_df = weather_df.merge(tide_df, on="Thời_gian", how="left").sort_values("Thời_gian")

    # Dữ liệu thời tiết liên tục theo giờ nên có thể nội suy các biến trơn.
    smooth_columns = ["Nhiệt_độ_C", "Độ_ẩm_%", "Độ_ẩm_đất", "Chiều_cao_triều_m"]
    for column in smooth_columns:
        final_df[column] = final_df[column].interpolate(limit_direction="both")

    # Lượng mưa thiếu thì coi như không ghi nhận mưa ở mốc giờ đó.
    final_df["Lượng_mưa_mm"] = final_df["Lượng_mưa_mm"].fillna(0)

    # Chặn giá trị bất hợp lý cơ bản.
    final_df["Độ_ẩm_%"] = final_df["Độ_ẩm_%"].clip(lower=0, upper=100)
    final_df["Độ_ẩm_đất"] = final_df["Độ_ẩm_đất"].clip(lower=0, upper=1)
    final_df["Lượng_mưa_mm"] = final_df["Lượng_mưa_mm"].clip(lower=0)
    final_df["Chiều_cao_triều_m"] = final_df["Chiều_cao_triều_m"].clip(lower=0, upper=5)

    return add_rule_based_label(final_df)


def append_or_create_historical_csv(save_path: Path, existing_df: pd.DataFrame, new_rows: pd.DataFrame) -> int:
    resolved_path = save_path.resolve()
    rows_to_save = new_rows.copy()
    rows_to_save["Thời_gian"] = rows_to_save["Thời_gian"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Cột "Địa phương" chỉ cần thiết cho cache/new_historical_rows.csv (gộp nhiều địa phương trong
    # 1 file, dùng cho incremental_train trong analyze_and_train.py) - KHÔNG được lưu vào file lịch sử
    # theo từng địa phương, vì địa phương đã được suy ra từ TÊN FILE ở mọi nơi đọc lại dữ liệu này
    # (load_and_concatenate_csvs trong analyze_and_train.py, load_and_combine_historical_data trong
    # eda_analysis.py). Nếu vẫn ghi cột này vào đây, file sẽ có số cột không khớp với header gốc
    # (7 cột) mỗi khi append, khiến pandas.read_csv() báo lỗi "Expected N fields, saw N+1" ở các lần
    # đọc sau - đây chính là nguyên nhân gây lỗi ParserError đã gặp trong `analyze_and_train.py`.
    if "Địa phương" in rows_to_save.columns:
        rows_to_save = rows_to_save.drop(columns=["Địa phương"])

    print(f"[DEBUG] Writing {len(rows_to_save)} rows to historical CSV: {resolved_path}")

    try:
        if save_path.exists() and not existing_df.empty:
            rows_to_save.to_csv(resolved_path, mode="a", header=False, index=False, encoding="utf-8-sig")
            return len(existing_df) + len(rows_to_save)

        rows_to_save.to_csv(resolved_path, index=False, encoding="utf-8-sig")
        return len(rows_to_save)
    except Exception as exc:
        raise ValueError(f"API Error: Failed to save CSV {resolved_path}: {exc}") from exc


def fetch_location_data(location, start_date_str: str, end_date_str: str):
    """Tải toàn bộ dữ liệu lịch sử cho một địa điểm."""
    print(f"\n--- Đang xử lý {location['name']} ---")
    print(f"[DEBUG] Requested date window: start={start_date_str} end={end_date_str}")

    weather_df = fetch_openmeteo_weather(location["lat"], location["lon"], start_date_str, end_date_str)
    if weather_df.empty:
        raise ValueError(f"API Error: Weather API returned no data for {location['name']}")

    tide_df = fetch_openmeteo_tide(location["coast_lat"], location["coast_lon"], start_date_str, end_date_str)
    if tide_df.empty:
        raise ValueError(f"API Error: Tide API returned no data for {location['name']}")

    return finalize_location_dataframe(weather_df, tide_df)


def main():
    print("=== BẮT ĐẦU TẢI DỮ LIỆU LỊCH SỬ ===")
    new_rows_frames = []
    for location in LOCATIONS:
        save_path = HISTORICAL_DIR / f"{location['name']}_10years.csv"
        print(f"[DEBUG] Target historical CSV: {save_path}")
        start_date_str, end_date_str = determine_date_window(save_path)
        print(f"Thời gian: {start_date_str} đến {end_date_str}\n")
        df = fetch_location_data(location, start_date_str, end_date_str)
        if df.empty:
            raise ValueError(f"API Error: Empty dataframe for {location['name']}")

        location_name_for_training = save_path.stem.replace("_10years", "").replace("_", " ").strip()
        df["Địa phương"] = location_name_for_training

        existing_df = pd.DataFrame()
        last_existing_timestamp = None
        if save_path.exists():
            try:
                existing_df = pd.read_csv(save_path, encoding="utf-8-sig")
                existing_df["Thời_gian"] = pd.to_datetime(existing_df["Thời_gian"], errors="coerce")
                existing_df = existing_df.dropna(subset=["Thời_gian"]).reset_index(drop=True)
                if not existing_df.empty:
                    last_existing_timestamp = pd.Timestamp(existing_df["Thời_gian"].iloc[-1])
                    print(f"[DEBUG] Existing last timestamp in CSV final row: {last_existing_timestamp}")
            except Exception as exc:
                raise ValueError(f"API Error: Failed to read existing CSV {save_path}: {exc}") from exc

        df["Thời_gian"] = pd.to_datetime(df["Thời_gian"], errors="coerce")
        df = df.dropna(subset=["Thời_gian"]).sort_values("Thời_gian").reset_index(drop=True)
        numeric_columns = ["Nhiệt_độ_C", "Độ_ẩm_%", "Lượng_mưa_mm", "Độ_ẩm_đất", "Chiều_cao_triều_m", "Nguy_cơ_ngập"]
        for column in numeric_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        if last_existing_timestamp is not None:
            new_rows = df.loc[df["Thời_gian"] > last_existing_timestamp].copy()
        else:
            new_rows = df.copy()

        if not new_rows.empty:
            new_rows_frames.append(new_rows)
        print(f"[DEBUG] New rows identified for append: {len(new_rows)}")

        if new_rows.empty:
            print(f"✅ Đã đồng bộ: {save_path} (total={len(existing_df)} | new=0)")
            continue

        total_rows = append_or_create_historical_csv(save_path, existing_df, new_rows)

        print(f"✅ Đã đồng bộ: {save_path} (total~={total_rows} | new={len(new_rows)})")

    print("\n=== HOÀN THÀNH TẢI DỮ LIỆU! ===")
    if new_rows_frames:
        new_rows_all = pd.concat(new_rows_frames, ignore_index=True)
        try:
            new_rows_all.to_csv(INCREMENTAL_NEW_ROWS_PATH, index=False, encoding="utf-8-sig")
        except Exception as exc:
            raise ValueError(f"API Error: Failed to save CSV {INCREMENTAL_NEW_ROWS_PATH.resolve()}: {exc}") from exc
        print(f"✅ Export new rows for incremental training: {INCREMENTAL_NEW_ROWS_PATH} ({len(new_rows_all)} dòng)")

        if str(os.environ.get("AUTO_INCREMENTAL_TRAINING", "1")).strip() == "1":
            command = [
                sys.executable,
                str(TRAINING_WORKER_PATH),
                "--mode",
                "incremental",
                "--models-json",
                json.dumps([], ensure_ascii=False),
                "--balancing-method",
                "auto",
                "--new-data-csv",
                str(INCREMENTAL_NEW_ROWS_PATH),
            ]
            subprocess.Popen(command, cwd=str(BASE_DIR))
            print("✅ Đã khởi chạy incremental training worker.")
    else:
        print("ℹ️ Không có bản ghi mới để incremental training.")


if __name__ == "__main__":
    main()
