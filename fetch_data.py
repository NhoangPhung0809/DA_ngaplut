import math
from datetime import datetime, timedelta
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

BASE_DIR = Path(__file__).resolve().parent
HISTORICAL_DIR = BASE_DIR / "data" / "historical"

# Cấu hình cache và retry cho các yêu cầu HTTP.
cache_session = requests_cache.CachedSession(str(BASE_DIR / ".cache"), expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Danh sách các địa điểm cần tải dữ liệu.
LOCATIONS = [
    {"name": "TP_Hue", "lat": 16.4637, "lon": 107.5909, "coast_lat": 16.4300, "coast_lon": 107.7600},
    {"name": "Huong_Thuy", "lat": 16.3382, "lon": 107.6742, "coast_lat": 16.3300, "coast_lon": 107.7800},
    {"name": "Phu_Vang", "lat": 16.4706, "lon": 107.7148, "coast_lat": 16.4400, "coast_lon": 107.8400},
    {"name": "Huong_Tra", "lat": 16.5181, "lon": 107.4747, "coast_lat": 16.5200, "coast_lon": 107.6200},
    {"name": "Quang_Dien", "lat": 16.5798, "lon": 107.4930, "coast_lat": 16.6100, "coast_lon": 107.5600},
]

# Thời gian tải dữ liệu: 10 năm gần nhất, lùi 7 ngày để tránh ngày hiện tại chưa đủ dữ liệu.
END_DATE = datetime.now() - timedelta(days=7)
START_DATE = END_DATE - timedelta(days=365 * 10)
START_DATE_STR = START_DATE.strftime("%Y-%m-%d")
END_DATE_STR = END_DATE.strftime("%Y-%m-%d")

HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)


def build_hourly_index(hourly_block):
    """Tạo trục thời gian từ metadata của Open-Meteo."""
    timestamps = pd.date_range(
        start=pd.to_datetime(hourly_block.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly_block.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly_block.Interval()),
        inclusive="left",
    )
    return timestamps.tz_localize(None)


def fetch_openmeteo_weather(lat, lon):
    """Tải dữ liệu thời tiết lịch sử từ Open-Meteo."""
    try:
        responses = openmeteo.weather_api(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": START_DATE_STR,
                "end_date": END_DATE_STR,
                "hourly": [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "rain",
                    "soil_moisture_0_to_7cm",
                ],
                "timezone": "auto",
            },
        )
        hourly = responses[0].Hourly()
        df = pd.DataFrame(
            {
                "Thời_gian": build_hourly_index(hourly),
                "Nhiệt_độ_C": hourly.Variables(0).ValuesAsNumpy(),
                "Độ_ẩm_%": hourly.Variables(1).ValuesAsNumpy(),
                "Lượng_mưa_mm": hourly.Variables(2).ValuesAsNumpy(),
                "Độ_ẩm_đất": hourly.Variables(3).ValuesAsNumpy(),
            }
        )
        return df
    except Exception as exc:
        print(f"⚠️ Lỗi Open-Meteo Weather ({lat}, {lon}): {exc}")
        return pd.DataFrame()


def fetch_openmeteo_tide(lat, lon):
    """Tải dữ liệu biển dùng làm xấp xỉ cho chiều cao triều."""
    try:
        responses = openmeteo.weather_api(
            "https://marine-api.open-meteo.com/v1/marine",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": START_DATE_STR,
                "end_date": END_DATE_STR,
                "hourly": "wave_height",
                "timezone": "auto",
            },
        )
        hourly = responses[0].Hourly()
        df = pd.DataFrame(
            {
                "Thời_gian": build_hourly_index(hourly),
                "Chiều_cao_triều_m": hourly.Variables(0).ValuesAsNumpy(),
            }
        )
        # Giới hạn giá trị vật lý hợp lý để tránh outlier từ API.
        df["Chiều_cao_triều_m"] = df["Chiều_cao_triều_m"].clip(lower=0.0, upper=5.0)
        return df
    except Exception as exc:
        print(f"⚠️ Lỗi Open-Meteo Tide ({lat}, {lon}): {exc}")
        return pd.DataFrame()


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


def fetch_location_data(location):
    """Tải toàn bộ dữ liệu lịch sử cho một địa điểm."""
    print(f"\n--- Đang xử lý {location['name']} ---")

    weather_df = fetch_openmeteo_weather(location["lat"], location["lon"])
    if weather_df.empty:
        print(f"❌ Không tải được dữ liệu thời tiết cho {location['name']}")
        return pd.DataFrame()

    tide_df = fetch_openmeteo_tide(location["coast_lat"], location["coast_lon"])
    if tide_df.empty:
        print("⚠️ Không lấy được dữ liệu triều từ API, dùng dữ liệu tổng hợp.")
        tide_df = calculate_synthetic_tide(weather_df["Thời_gian"])

    return finalize_location_dataframe(weather_df, tide_df)


def main():
    print("=== BẮT ĐẦU TẢI DỮ LIỆU LỊCH SỬ ===")
    print(f"Thời gian: {START_DATE_STR} đến {END_DATE_STR}\n")

    for location in LOCATIONS:
        df = fetch_location_data(location)
        if df.empty:
            continue

        save_path = HISTORICAL_DIR / f"{location['name']}_10years.csv"
        df.to_csv(save_path, index=False)
        print(f"✅ Đã lưu thành công: {save_path} ({len(df)} dòng)")

    print("\n=== HOÀN THÀNH TẢI DỮ LIỆU! ===")


if __name__ == "__main__":
    main()
