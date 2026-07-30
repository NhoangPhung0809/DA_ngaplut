import json
import importlib.util
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import folium
import joblib
import openmeteo_requests
import pandas as pd
import plotly.express as px
import requests
import requests_cache
import streamlit as st
from branca.colormap import LinearColormap
from dotenv import load_dotenv
from retry_requests import retry
from routing_engine import DEFAULT_CENTER, build_visual_line_map, download_or_load_graph, get_safe_route
from streamlit_folium import st_folium

# Tải biến môi trường từ file .env nếu có.
load_dotenv()

# Cấu hình giao diện trang Streamlit.
st.set_page_config(
    page_title="Dự báo Ngập lụt TP Huế",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
HISTORICAL_DIR = BASE_DIR / "data" / "historical"
MODELS_DIR = BASE_DIR / "models"
LATEST_MODELS_DIR = MODELS_DIR / "latest"
PLOTS_DIR = BASE_DIR / "plots"
GEOJSON_PATH = BASE_DIR / "data" / "geo" / "thuathienhue_districts.geojson"
RAW_OSM_PATH = BASE_DIR / "data" / "geo" / "map"
STORMGLASS_CACHE_PATH = BASE_DIR / "data" / "stormglass_cache.json"
FETCH_SCRIPT_PATH = BASE_DIR / "fetch_data.py"
TRAIN_SCRIPT_PATH = BASE_DIR / "analyze_and_train.py"
EVALUATION_METRICS_PATH = LATEST_MODELS_DIR / "evaluation_metrics.json"
SCALER_PATH = LATEST_MODELS_DIR / "scaler.pkl"
PRIMARY_MODEL_PATH = LATEST_MODELS_DIR / "best_model.pkl"
EXPECTED_HISTORICAL_FILES = {
    "TP_Hue_10years.csv",
    "Huong_Thuy_10years.csv",
    "Phu_Vang_10years.csv",
    "Huong_Tra_10years.csv",
    "Quang_Dien_10years.csv",
}

# Khởi tạo client Open-Meteo có cache và retry để hạn chế lỗi mạng tạm thời.
cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Dữ liệu gốc cho các địa phương cần theo dõi.
LOCATIONS = [
    {
        "name": "TP Huế",
        "aliases": ["Huế", "Thành phố Huế", "TP Huế"],
        "lat": 16.4637,
        "lon": 107.5909,
        "coast_lat": 16.4300,
        "coast_lon": 107.7600,
    },
    {
        "name": "Hương Thủy",
        "aliases": ["Hương Thủy", "Phường Hương Thủy", "Thị xã Hương Thủy"],
        "lat": 16.3382,
        "lon": 107.6742,
        "coast_lat": 16.3300,
        "coast_lon": 107.7800,
    },
    {
        "name": "Phú Vang",
        "aliases": ["Phú Vang", "Huyện Phú Vang"],
        "lat": 16.4706,
        "lon": 107.7148,
        "coast_lat": 16.4400,
        "coast_lon": 107.8400,
    },
    {
        "name": "Hương Trà",
        "aliases": ["Hương Trà", "Thị xã Hương Trà"],
        "lat": 16.5181,
        "lon": 107.4747,
        "coast_lat": 16.5200,
        "coast_lon": 107.6200,
    },
    {
        "name": "Quảng Điền",
        "aliases": ["Quảng Điền", "Huyện Quảng Điền"],
        "lat": 16.5798,
        "lon": 107.4930,
        "coast_lat": 16.6100,
        "coast_lon": 107.5600,
    },
]

DEFAULT_FEATURE_ORDER = [
    "Nhiệt_độ_C",
    "Độ_ẩm_%",
    "Lượng_mưa_mm",
    "Độ_ẩm_đất",
    "Chiều_cao_triều_m",
]

MAP_NAME_NORMALIZATION = {
    "Huế": "TP Huế",
    "Thành phố Huế": "TP Huế",
    "TP Huế": "TP Huế",
    "Hương Thủy": "Hương Thủy",
    "Phường Hương Thủy": "Hương Thủy",
    "Thị xã Hương Thủy": "Hương Thủy",
    "Phú Vang": "Phú Vang",
    "Huyện Phú Vang": "Phú Vang",
    "Hương Trà": "Hương Trà",
    "Thị xã Hương Trà": "Hương Trà",
    "Quảng Điền": "Quảng Điền",
    "Huyện Quảng Điền": "Quảng Điền",
}

HUE_LAT_RANGE = (15.80, 16.85)
HUE_LON_RANGE = (107.35, 107.95)


def apply_global_ui_theme():
    """Tăng độ tương phản tổng thể cho giao diện Streamlit."""
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #020817 0%, #06101f 100%);
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #f8fafc !important;
            font-weight: 700 !important;
        }
        p, li, label, span, div {
            color: #e5eefc;
        }
        [data-testid="stCaptionContainer"] p {
            color: #cbd5e1 !important;
            font-size: 0.95rem !important;
        }
        [data-testid="stExpander"] {
            background: #0b1220;
            border: 1px solid #334155;
            border-radius: 10px;
        }
        [data-testid="stExpander"] summary {
            color: #f8fafc !important;
            font-weight: 600 !important;
        }
        [data-baseweb="tab-list"] {
            gap: 8px;
        }
        [data-baseweb="tab"] {
            background: #0f172a !important;
            color: #cbd5e1 !important;
            border-radius: 8px 8px 0 0 !important;
            border: 1px solid #334155 !important;
            padding: 10px 16px !important;
            font-weight: 600 !important;
        }
        [aria-selected="true"][data-baseweb="tab"] {
            background: #1d4ed8 !important;
            color: #eff6ff !important;
            border-color: #3b82f6 !important;
        }
        .stButton > button,
        .stDownloadButton > button {
            background: #0f172a !important;
            color: #f8fafc !important;
            border: 1px solid #475569 !important;
            font-weight: 600 !important;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: #60a5fa !important;
            color: #ffffff !important;
        }
        section[data-testid="stSidebar"] {
            background: #06101f;
        }
        section[data-testid="stSidebar"] * {
            color: #e5eefc !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def dynamically_import_module(module_name: str, module_path: Path):
    """Import module động từ đường dẫn file Python."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Không thể tạo spec cho module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def historical_data_missing() -> bool:
    """Kiểm tra bộ dữ liệu lịch sử 10 năm có đầy đủ hay chưa."""
    if not HISTORICAL_DIR.exists():
        return True

    existing_files = {file_path.name for file_path in HISTORICAL_DIR.glob("*.csv")}
    return not EXPECTED_HISTORICAL_FILES.issubset(existing_files)


def ensure_latest_models_dir() -> None:
    """Đảm bảo thư mục `models/latest/` luôn tồn tại cho frontend."""
    LATEST_MODELS_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource(show_spinner=False)
def get_train_module():
    """Nạp module huấn luyện để tái sử dụng cho init và thao tác từ sidebar."""
    return dynamically_import_module("analyze_and_train_runtime", TRAIN_SCRIPT_PATH)


def get_all_model_names() -> list[str]:
    """Lấy danh sách model từ script train, fallback về danh sách cố định nếu cần."""
    fallback_names = [
        "Linear Regression Threshold",
        "Polynomial Regression Threshold",
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "ARIMA",
        "SARIMA",
        "LSTM",
        "LSTM + XGBoost Hybrid",
    ]
    try:
        train_module = get_train_module()
        if hasattr(train_module, "list_available_models"):
            return list(train_module.list_available_models())
    except Exception:
        pass
    return fallback_names


def training_artifacts_missing() -> bool:
    """Kiểm tra các artifact tối thiểu trong `models/latest/`."""
    ensure_latest_models_dir()
    return not PRIMARY_MODEL_PATH.exists() or not SCALER_PATH.exists() or not EVALUATION_METRICS_PATH.exists()


@st.cache_resource(show_spinner=False)
def initialize_system():
    """
    Khởi tạo hệ thống đúng một lần khi app bắt đầu:
    - Tự tải dữ liệu lịch sử nếu thiếu
    - Tự huấn luyện mô hình nếu thiếu artifact
    """
    ensure_latest_models_dir()

    if historical_data_missing():
        with st.spinner("📥 Downloading 10-year historical data..."):
            fetch_module = dynamically_import_module("fetch_data_runtime", FETCH_SCRIPT_PATH)
            if not hasattr(fetch_module, "main"):
                raise AttributeError("`fetch_data.py` không có hàm `main()` để thực thi.")
            fetch_module.main()

    if training_artifacts_missing():
        with st.spinner("🧠 Training multi-class ML models (SMOTE applied)..."):
            train_module = get_train_module()
            if not hasattr(train_module, "run_training_pipeline"):
                raise AttributeError("`analyze_and_train.py` không có hàm `run_training_pipeline()`.")
            train_module.run_training_pipeline(get_all_model_names())

    if not PRIMARY_MODEL_PATH.exists():
        raise FileNotFoundError("Không tìm thấy `models/latest/best_model.pkl` sau khi huấn luyện.")
    if not SCALER_PATH.exists():
        raise FileNotFoundError("Không tìm thấy `models/latest/scaler.pkl` sau khi huấn luyện.")
    if not EVALUATION_METRICS_PATH.exists():
        raise FileNotFoundError("Không tìm thấy `models/latest/evaluation_metrics.json` sau khi huấn luyện.")

    return {
        "model_path": str(PRIMARY_MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
        "metrics_path": str(EVALUATION_METRICS_PATH),
        "latest_dir": str(LATEST_MODELS_DIR),
    }


@st.cache_resource(show_spinner=False)
def load_runtime_artifacts():
    """Nạp model, scaler và metrics sau khi hệ thống đã khởi tạo xong."""
    runtime_info = initialize_system()
    model = joblib.load(runtime_info["model_path"])
    scaler = joblib.load(runtime_info["scaler_path"])
    with open(runtime_info["metrics_path"], "r", encoding="utf-8") as file:
        evaluation_metrics = json.load(file)
    return model, scaler, evaluation_metrics, runtime_info


def calculate_synthetic_tide(target_time=None):
    """Sinh giá trị triều cường gần đúng khi API biển không trả dữ liệu."""
    if target_time is None:
        target_time = datetime.now()

    lunar_cycle_days = 29.53
    semi_daily_hours = 12.42
    elapsed_seconds = (target_time - datetime(2000, 1, 1)).total_seconds()

    lunar_phase = 2 * math.pi * elapsed_seconds / (lunar_cycle_days * 86400)
    semi_daily_phase = 2 * math.pi * elapsed_seconds / (semi_daily_hours * 3600)
    tide_value = 1.0 + 0.45 * math.sin(lunar_phase) + 0.75 * math.sin(semi_daily_phase)

    return round(max(0.1, min(4.0, tide_value)), 2)


def get_current_weather(location):
    """Lấy dữ liệu thời tiết hiện tại cho một địa phương."""
    weather_data = {
        "Nhiệt_độ_C": 25.0,
        "Độ_ẩm_%": 70.0,
        "Lượng_mưa_mm": 0.0,
        "Độ_ẩm_đất": 0.30,
        "Chiều_cao_triều_m": calculate_synthetic_tide(),
    }

    try:
        weather_responses = openmeteo.weather_api(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": location["lat"],
                "longitude": location["lon"],
                "current": [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "rain",
                    "soil_moisture_0_to_7cm",
                ],
                "timezone": "auto",
            },
        )
        current = weather_responses[0].Current()
        weather_data["Nhiệt_độ_C"] = current.Variables(0).Value()
        weather_data["Độ_ẩm_%"] = current.Variables(1).Value()
        weather_data["Lượng_mưa_mm"] = current.Variables(2).Value()
        weather_data["Độ_ẩm_đất"] = current.Variables(3).Value()
    except Exception:
        # Nếu API thời tiết lỗi thì giữ giá trị fallback đã khai báo phía trên.
        pass

    try:
        marine_responses = openmeteo.weather_api(
            "https://marine-api.open-meteo.com/v1/marine",
            params={
                "latitude": location["coast_lat"],
                "longitude": location["coast_lon"],
                "current": "wave_height",
                "timezone": "auto",
            },
        )
        current_marine = marine_responses[0].Current()
        marine_value = current_marine.Variables(0).Value()
        if marine_value is not None:
            weather_data["Chiều_cao_triều_m"] = max(0.1, round(float(marine_value), 2))
    except Exception:
        # Giữ giá trị tổng hợp nếu không lấy được từ nguồn biển.
        pass

    return weather_data


def build_hourly_time_index(hourly_block):
    """Tạo trục thời gian từ block dữ liệu theo giờ của Open-Meteo."""
    hourly_index = pd.date_range(
        start=pd.to_datetime(hourly_block.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly_block.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly_block.Interval()),
        inclusive="left",
    )
    return hourly_index.tz_localize(None)


def fetch_daily_forecast_features(location, forecast_days=14):
    """Lấy dữ liệu forecast theo ngày bằng cách tổng hợp từ dữ liệu theo giờ."""
    weather_columns = [
        "temperature_2m",
        "relative_humidity_2m",
        "rain",
        "soil_moisture_0_to_7cm",
    ]
    forecast_df = pd.DataFrame()

    try:
        weather_responses = openmeteo.weather_api(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": location["lat"],
                "longitude": location["lon"],
                "hourly": weather_columns,
                "forecast_days": forecast_days,
                "timezone": "auto",
            },
        )
        hourly = weather_responses[0].Hourly()
        forecast_df = pd.DataFrame(
            {
                "Thời_gian": build_hourly_time_index(hourly),
                "Nhiệt_độ_C": hourly.Variables(0).ValuesAsNumpy(),
                "Độ_ẩm_%": hourly.Variables(1).ValuesAsNumpy(),
                "Lượng_mưa_mm": hourly.Variables(2).ValuesAsNumpy(),
                "Độ_ẩm_đất": hourly.Variables(3).ValuesAsNumpy(),
            }
        )
    except Exception:
        return pd.DataFrame()

    marine_daily = None
    try:
        marine_responses = openmeteo.weather_api(
            "https://marine-api.open-meteo.com/v1/marine",
            params={
                "latitude": location["coast_lat"],
                "longitude": location["coast_lon"],
                "hourly": "wave_height",
                "forecast_days": forecast_days,
                "timezone": "auto",
            },
        )
        marine_hourly = marine_responses[0].Hourly()
        marine_df = pd.DataFrame(
            {
                "Thời_gian": build_hourly_time_index(marine_hourly),
                "Chiều_cao_triều_m": marine_hourly.Variables(0).ValuesAsNumpy(),
            }
        )
        marine_df["Ngày"] = marine_df["Thời_gian"].dt.date
        marine_daily = (
            marine_df.groupby("Ngày", as_index=False)["Chiều_cao_triều_m"]
            .mean()
            .assign(Chiều_cao_triều_m=lambda df: df["Chiều_cao_triều_m"].clip(0.1, 5.0))
        )
    except Exception:
        marine_daily = None

    forecast_df["Ngày"] = forecast_df["Thời_gian"].dt.date
    daily_weather = (
        forecast_df.groupby("Ngày", as_index=False)
        .agg(
            {
                "Nhiệt_độ_C": "mean",
                "Độ_ẩm_%": "mean",
                "Lượng_mưa_mm": "sum",
                "Độ_ẩm_đất": "mean",
            }
        )
    )

    if marine_daily is not None and not marine_daily.empty:
        daily_forecast = daily_weather.merge(marine_daily, on="Ngày", how="left")
    else:
        daily_forecast = daily_weather.copy()
        daily_forecast["Chiều_cao_triều_m"] = pd.NA

    daily_forecast["Chiều_cao_triều_m"] = daily_forecast.apply(
        lambda row: (
            float(row["Chiều_cao_triều_m"])
            if pd.notna(row["Chiều_cao_triều_m"])
            else calculate_synthetic_tide(pd.to_datetime(row["Ngày"]) + pd.Timedelta(hours=12))
        ),
        axis=1,
    )
    daily_forecast["Địa phương"] = location["name"]
    return daily_forecast


def get_future_predictions(model, scaler, forecast_days=14):
    """Dự báo nguy cơ ngập cho các ngày sắp tới bằng mô hình ML đã huấn luyện."""
    feature_columns = get_expected_feature_columns(model, scaler)
    all_predictions = []

    for location in LOCATIONS:
        location_forecast = fetch_daily_forecast_features(location, forecast_days=forecast_days)
        if location_forecast.empty:
            continue

        feature_frame = location_forecast[feature_columns].copy()
        scaled_features = scale_feature_frame_for_inference(scaler, feature_frame)
        predictions = model.predict(scaled_features)
        if hasattr(model, "predict_proba"):
            proba_matrix = model.predict_proba(scaled_features)
            if proba_matrix.shape[1] >= 3:
                probabilities = proba_matrix[:, 1:].sum(axis=1)
            elif proba_matrix.shape[1] == 2:
                probabilities = proba_matrix[:, 1]
            else:
                probabilities = proba_matrix[:, 0]
        else:
            probabilities = (predictions > 0).astype(float)

        location_forecast["Xác suất ngập (%)"] = (probabilities * 100).round(2)
        location_forecast["Nguy cơ"] = [format_risk_label(int(value)) for value in predictions]
        location_forecast["Mưa dự báo (mm)"] = location_forecast["Lượng_mưa_mm"]
        location_forecast["Ngày"] = pd.to_datetime(location_forecast["Ngày"])
        all_predictions.append(location_forecast)

    if not all_predictions:
        return pd.DataFrame()

    future_df = pd.concat(all_predictions, ignore_index=True)
    future_df = future_df.rename(
        columns={
            "Nhiệt_độ_C": "Nhiệt độ (°C)",
            "Độ_ẩm_%": "Độ ẩm (%)",
            "Độ_ẩm_đất": "Độ ẩm đất",
            "Chiều_cao_triều_m": "Chiều cao triều (m)",
        }
    )
    future_df = future_df.sort_values(["Ngày", "Địa phương"]).reset_index(drop=True)
    future_df["Ngày"] = future_df["Ngày"].dt.strftime("%Y-%m-%d")
    return future_df[
        [
            "Ngày",
            "Địa phương",
            "Nhiệt độ (°C)",
            "Độ ẩm (%)",
            "Mưa dự báo (mm)",
            "Độ ẩm đất",
            "Chiều cao triều (m)",
            "Xác suất ngập (%)",
            "Nguy cơ",
        ]
    ]


def get_expected_feature_columns(model, scaler):
    """Xác định đúng thứ tự feature mà scaler/mô hình đang mong đợi."""
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return DEFAULT_FEATURE_ORDER


def build_feature_frame(weather, feature_columns):
    """Tạo DataFrame đầu vào đúng tên cột và đúng thứ tự."""
    row = {column: weather.get(column, 0.0) for column in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)


def scale_feature_frame_for_inference(scaler, feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Scale dữ liệu nhưng vẫn giữ nguyên DataFrame và tên cột cho sklearn."""
    scaled_values = scaler.transform(feature_frame)
    return pd.DataFrame(
        scaled_values,
        columns=feature_frame.columns,
        index=feature_frame.index,
    )


def get_prediction_and_flood_probability(model, features_scaled: pd.DataFrame):
    """
    Trả về:
    - nhãn dự đoán
    - xác suất có nguy cơ ngập

    Hỗ trợ cả mô hình nhị phân lẫn đa lớp.
    Với bài toán đa lớp 0/1/2:
    - Xác suất ngập = P(class 1) + P(class 2)
    """
    prediction = int(model.predict(features_scaled)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_scaled)[0]
        if len(probabilities) >= 3:
            flood_probability = float(probabilities[1:].sum())
        elif len(probabilities) == 2:
            flood_probability = float(probabilities[1])
        else:
            flood_probability = float(probabilities[0])
    else:
        flood_probability = float(prediction > 0)

    return prediction, flood_probability


def format_risk_label(prediction: int) -> str:
    """Chuẩn hóa nhãn hiển thị cho dashboard."""
    if prediction <= 0:
        return "🟢 An toàn"
    if prediction == 1:
        return "🟠 Ngập nhẹ"
    return "🔴 Ngập nặng"


def get_processing_timestamp_display() -> str:
    """Trả về thời gian xử lý hiện tại theo định dạng dễ đọc cho dashboard."""
    return datetime.now().strftime("%H:%M - %d/%m/%Y")


def get_realtime_prediction(model, scaler):
    """Dự báo realtime cho tất cả địa phương trong danh sách."""
    feature_columns = get_expected_feature_columns(model, scaler)
    predictions = []
    processing_time_display = get_processing_timestamp_display()

    for location in LOCATIONS:
        weather = get_current_weather(location)
        feature_frame = build_feature_frame(weather, feature_columns)
        scaled_features = scale_feature_frame_for_inference(scaler, feature_frame)

        prediction, probability = get_prediction_and_flood_probability(model, scaled_features)

        predictions.append(
            {
                "Thời gian": processing_time_display,
                "Địa phương": location["name"],
                "Vĩ độ": location["lat"],
                "Kinh độ": location["lon"],
                "Nhiệt độ (°C)": round(float(weather["Nhiệt_độ_C"]), 2),
                "Độ ẩm (%)": round(float(weather["Độ_ẩm_%"]), 2),
                "Lượng mưa (mm)": round(float(weather["Lượng_mưa_mm"]), 2),
                "Độ ẩm đất": round(float(weather["Độ_ẩm_đất"]), 2),
                "Chiều cao triều (m)": round(float(weather["Chiều_cao_triều_m"]), 2),
                "Xác suất ngập (%)": round(probability * 100, 2),
                "Nguy cơ": format_risk_label(prediction),
            }
        )

    return pd.DataFrame(predictions)


def build_noaa_placeholder() -> dict:
    """Giá trị mặc định khi NOAA chưa đồng bộ hoặc không tìm thấy dữ liệu."""
    return {
        "Nhiệt độ (NOAA)": "N/A (Chờ vệ tinh)",
        "Mưa (NOAA)": "N/A",
    }


def is_numeric_value(value) -> bool:
    """Kiểm tra giá trị có phải số hợp lệ hay không."""
    return isinstance(value, (int, float)) and not pd.isna(value)


def format_table_value(value):
    """Chuẩn hóa giá trị hiển thị trong bảng đối chiếu."""
    if is_numeric_value(value):
        return round(float(value), 2)
    return value


def derive_confidence_label(meteo_temp, noaa_temp, meteo_rain, noaa_rain) -> str:
    """Đánh giá mức độ tin cậy dựa trên sai lệch giữa Open-Meteo và NOAA."""
    temp_diff = None
    rain_diff = None

    if is_numeric_value(meteo_temp) and is_numeric_value(noaa_temp):
        temp_diff = abs(float(meteo_temp) - float(noaa_temp))
    if is_numeric_value(meteo_rain) and is_numeric_value(noaa_rain):
        rain_diff = abs(float(meteo_rain) - float(noaa_rain))

    if temp_diff is None and rain_diff is None:
        return "⚪ Chờ đồng bộ"

    if temp_diff is not None and rain_diff is not None:
        return "🟢 Cao" if temp_diff <= 2.0 and rain_diff <= 10.0 else "🔴 Thấp"

    if temp_diff is not None:
        return "🟢 Cao" if temp_diff <= 2.0 else "🔴 Thấp"

    return "🟢 Cao" if rain_diff is not None and rain_diff <= 10.0 else "🔴 Thấp"


@st.cache_data(show_spinner=False, ttl=21600)
def fetch_noaa_cdo_data(lat, lon):
    """
    Lấy daily summary từ NOAA CDO cho 2 ngày gần nhất.

    NOAA có thể trễ đồng bộ nên nếu không có dữ liệu phù hợp sẽ trả về placeholder.
    """
    token = os.getenv("NOAA_CDO_TOKEN")
    headers = {
        "token": token,
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        ),
    }
    placeholder = build_noaa_placeholder()
    if not token:
        return placeholder

    try:
        end_date = datetime.now(timezone.utc).date() - timedelta(days=1)
        start_date = end_date - timedelta(days=1)
        bbox_delta = 0.75

        stations_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/stations"
        station_params = {
            "datasetid": "GHCND",
            "extent": f"{lat - bbox_delta},{lon - bbox_delta},{lat + bbox_delta},{lon + bbox_delta}",
            "startdate": start_date.isoformat(),
            "enddate": end_date.isoformat(),
            "limit": 10,
            "sortfield": "maxdate",
            "sortorder": "desc",
        }
        station_response = requests.get(
            stations_url,
            headers=headers,
            params=station_params,
            timeout=20,
        )
        station_response.raise_for_status()
        stations = station_response.json().get("results", [])
        if not stations:
            return placeholder

        selected_station = min(
            stations,
            key=lambda station: (
                (float(station.get("latitude", lat)) - lat) ** 2
                + (float(station.get("longitude", lon)) - lon) ** 2
            ),
        )
        station_id = selected_station.get("id")
        if not station_id:
            return placeholder

        data_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
        data_params = {
            "datasetid": "GHCND",
            "stationid": station_id,
            "startdate": start_date.isoformat(),
            "enddate": end_date.isoformat(),
            "datatypeid": ["TAVG", "TMAX", "TMIN", "PRCP"],
            "units": "metric",
            "limit": 1000,
            "sortfield": "date",
            "sortorder": "desc",
        }
        data_response = requests.get(
            data_url,
            headers=headers,
            params=data_params,
            timeout=20,
        )
        data_response.raise_for_status()
        results = data_response.json().get("results", [])
        if not results:
            return placeholder

        latest_date = max(str(item.get("date", ""))[:10] for item in results if item.get("date"))
        latest_results = [
            item for item in results if str(item.get("date", "")).startswith(latest_date)
        ]
        if not latest_results:
            return placeholder

        datatype_map = {}
        for item in latest_results:
            datatype = item.get("datatype")
            value = item.get("value")
            if datatype and value is not None:
                datatype_map[datatype] = float(value)

        noaa_temp = datatype_map.get("TAVG")
        if noaa_temp is None:
            tmax = datatype_map.get("TMAX")
            tmin = datatype_map.get("TMIN")
            if tmax is not None and tmin is not None:
                noaa_temp = (tmax + tmin) / 2
            elif tmax is not None:
                noaa_temp = tmax
            elif tmin is not None:
                noaa_temp = tmin

        noaa_rain = datatype_map.get("PRCP")
        if noaa_temp is None and noaa_rain is None:
            return placeholder

        return {
            "Nhiệt độ (NOAA)": format_table_value(noaa_temp)
            if noaa_temp is not None
            else placeholder["Nhiệt độ (NOAA)"],
            "Mưa (NOAA)": format_table_value(noaa_rain)
            if noaa_rain is not None
            else placeholder["Mưa (NOAA)"],
        }
    except Exception:
        return placeholder


@st.cache_data(show_spinner=False, ttl=60)
def fetch_tomorrow_realtime(lat: float, lon: float):
    api_key = os.getenv("TOMORROW_API_KEY")
    if not api_key:
        return None

    url = "https://api.tomorrow.io/v4/weather/realtime"
    headers = {"accept-encoding": "deflate, gzip, br", "accept": "application/json"}
    params = {
        "location": f"{lat},{lon}",
        "apikey": api_key,
        "units": "metric",
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        if response.status_code in {401, 403, 429}:
            return None
        response.raise_for_status()
        payload = response.json()
        values = payload.get("data", {}).get("values", {})
        if not isinstance(values, dict) or not values:
            return None

        return {
            "time": payload.get("data", {}).get("time"),
            "temperature": values.get("temperature"),
            "humidity": values.get("humidity"),
            "precipitationIntensity": values.get("precipitationIntensity"),
            "windSpeed": values.get("windSpeed"),
        }
    except Exception:
        return None


def render_tomorrow_nowcasting():
    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 Thời tiết Hiện tại (Nowcasting - Tomorrow.io)")

    location = LOCATIONS[0] if LOCATIONS else None
    if location is None:
        st.sidebar.info("Không có tọa độ để hiển thị Nowcasting.")
        return

    nowcast = fetch_tomorrow_realtime(location["lat"], location["lon"])
    if nowcast is None:
        st.sidebar.info("Chưa cấu hình TOMORROW_API_KEY hoặc không lấy được dữ liệu Tomorrow.io.")
        return

    time_value = nowcast.get("time")
    if time_value:
        st.sidebar.caption(f"Vị trí: {location['name']} | {time_value}")
    else:
        st.sidebar.caption(f"Vị trí: {location['name']}")

    def format_metric(value, suffix: str, decimals: int = 2) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "N/A"
        try:
            return f"{float(value):.{decimals}f}{suffix}"
        except Exception:
            return "N/A"

    col_1, col_2 = st.sidebar.columns(2)
    col_1.metric("Nhiệt độ", format_metric(nowcast.get("temperature"), " °C"))
    col_2.metric("Độ ẩm", format_metric(nowcast.get("humidity"), " %"))

    col_3, col_4 = st.sidebar.columns(2)
    col_3.metric("Mưa (cường độ)", format_metric(nowcast.get("precipitationIntensity"), " mm/h"))
    col_4.metric("Gió", format_metric(nowcast.get("windSpeed"), " m/s"))


def load_stormglass_cache():
    """Đọc cache JSON Stormglass từ đĩa, lỗi thì trả về None."""
    try:
        if not STORMGLASS_CACHE_PATH.exists():
            return None
        with STORMGLASS_CACHE_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def extract_next_high_tide(stormglass_payload):
    """Lấy đợt triều cao kế tiếp từ JSON Stormglass."""
    if not isinstance(stormglass_payload, dict):
        return None

    tide_items = stormglass_payload.get("data", [])
    if not isinstance(tide_items, list):
        return None

    now_utc = datetime.now(timezone.utc)
    next_high_tide = None

    for item in tide_items:
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "")).lower() != "high":
            continue

        raw_time = item.get("time")
        raw_height = item.get("height")
        if raw_time is None or raw_height is None:
            continue

        try:
            tide_time = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
            tide_height = float(raw_height)
        except Exception:
            continue

        if tide_time < now_utc:
            continue

        if next_high_tide is None or tide_time < next_high_tide["time"]:
            next_high_tide = {"time": tide_time, "height": tide_height}

    return next_high_tide


def render_stormglass_tide_sidebar():
    """Hiển thị mục thủy triều từ cache JSON Stormglass ở sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌊 Thủy triều Stormglass")

    stormglass_payload = load_stormglass_cache()
    next_high_tide = extract_next_high_tide(stormglass_payload)

    if next_high_tide is None:
        st.sidebar.info("Chưa có dữ liệu Thủy triều")
        return

    local_time = next_high_tide["time"].astimezone()
    st.sidebar.caption(f"Triều cao kế tiếp lúc: {local_time.strftime('%H:%M - %d/%m/%Y')}")

    metric_col_1, metric_col_2 = st.sidebar.columns(2)
    metric_col_1.metric("Chiều cao", f"{next_high_tide['height']:.2f} m")
    metric_col_2.metric("Loại", "Triều cao")


def render_cross_validation_table(df_open_meteo: pd.DataFrame) -> pd.DataFrame:
    """Tạo bảng đối chiếu Open-Meteo với NOAA CDO cho từng địa phương."""
    comparison_rows = []

    for _, row in df_open_meteo.iterrows():
        noaa_data = fetch_noaa_cdo_data(
            lat=float(row["Vĩ độ"]),
            lon=float(row["Kinh độ"]),
        )

        meteo_temp = row.get("Nhiệt độ (°C)")
        meteo_rain = row.get("Lượng mưa (mm)")
        noaa_temp = noaa_data.get("Nhiệt độ (NOAA)")
        noaa_rain = noaa_data.get("Mưa (NOAA)")

        comparison_rows.append(
            {
                "Địa phương": row.get("Địa phương"),
                "Nhiệt độ (Meteo)": format_table_value(meteo_temp),
                "Nhiệt độ (NOAA)": format_table_value(noaa_temp),
                "Mưa (Meteo)": format_table_value(meteo_rain),
                "Mưa (NOAA)": format_table_value(noaa_rain),
                "Độ tin cậy": derive_confidence_label(
                    meteo_temp=meteo_temp,
                    noaa_temp=noaa_temp,
                    meteo_rain=meteo_rain,
                    noaa_rain=noaa_rain,
                ),
            }
        )

    return pd.DataFrame(
        comparison_rows,
        columns=[
            "Địa phương",
            "Nhiệt độ (Meteo)",
            "Nhiệt độ (NOAA)",
            "Mưa (Meteo)",
            "Mưa (NOAA)",
            "Độ tin cậy",
        ],
    )


def normalize_location_name(name):
    """Chuẩn hóa tên địa phương để ghép dữ liệu dự báo với dữ liệu bản đồ."""
    if not name:
        return None
    return MAP_NAME_NORMALIZATION.get(str(name).strip(), str(name).strip())


@st.cache_data
def load_geojson_data():
    """Đọc dữ liệu GeoJSON và chuẩn hóa tên địa phương."""
    if not GEOJSON_PATH.exists():
        return None

    with GEOJSON_PATH.open("r", encoding="utf-8") as file:
        geojson_data = json.load(file)

    features = geojson_data.get("features", [])
    normalized_features = []
    for feature in features:
        properties = feature.get("properties", {})
        raw_name = (
            properties.get("name")
            or properties.get("NAME_2")
            or properties.get("ten")
            or properties.get("district")
        )
        normalized_name = normalize_location_name(raw_name)
        if not normalized_name:
            continue

        feature["properties"]["name"] = normalized_name
        normalized_features.append(feature)

    geojson_data["features"] = normalized_features
    return geojson_data


@st.cache_resource(show_spinner=False)
def load_routing_graph():
    """Nạp graph OSM đường bộ cho Huế và cache ở cấp resource."""
    return download_or_load_graph()


def initialize_routing_state():
    """Khởi tạo state cho tính năng click chọn điểm và định tuyến an toàn."""
    default_start = LOCATIONS[0]
    default_end = LOCATIONS[1] if len(LOCATIONS) > 1 else LOCATIONS[0]
    defaults = {
        "route_start_lat": float(default_start["lat"]),
        "route_start_lon": float(default_start["lon"]),
        "route_end_lat": float(default_end["lat"]),
        "route_end_lon": float(default_end["lon"]),
        "routing_enabled": False,
        "routing_last_clicked": None,
        "routing_last_error": None,
        "routing_last_summary": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def prediction_label_to_routing_risk(label: str) -> str:
    """Ánh xạ nhãn nguy cơ tiếng Việt sang nhãn routing_engine."""
    value = str(label or "")
    if "Ngập nặng" in value:
        return "Heavy Flood"
    if "Ngập nhẹ" in value:
        return "Light Flood"
    return "Safe"


def build_flooded_regions_from_predictions(df_predictions: pd.DataFrame) -> list[dict]:
    """Chuyển dự báo theo địa phương thành vùng flood để tô màu OSM roads."""
    risk_lookup = build_risk_lookup(df_predictions)
    geojson_data = load_geojson_data()
    flooded_regions = []

    if geojson_data and geojson_data.get("features"):
        for feature in geojson_data.get("features", []):
            properties = feature.get("properties", {})
            district_name = properties.get("name")
            district_info = risk_lookup.get(district_name)
            if not district_info:
                continue

            risk_status = prediction_label_to_routing_risk(district_info["risk"])
            if risk_status == "Safe":
                continue

            geometry = feature.get("geometry")
            if geometry is None:
                continue

            flooded_regions.append(
                {
                    "name": district_name,
                    "risk": risk_status,
                    "geometry": geometry,
                }
            )

        return flooded_regions

    for _, row in df_predictions.iterrows():
        risk_status = prediction_label_to_routing_risk(row.get("Nguy cơ"))
        if risk_status == "Safe":
            continue

        center_lat = float(row["Vĩ độ"])
        center_lon = float(row["Kinh độ"])
        flooded_regions.append(
            {
                "name": row["Địa phương"],
                "risk": risk_status,
                "bounds": [
                    center_lat - 0.045,
                    center_lon - 0.055,
                    center_lat + 0.045,
                    center_lon + 0.055,
                ],
            }
        )

    return flooded_regions


def add_selected_route_markers(base_map: folium.Map) -> None:
    """Hiển thị marker A/B hiện tại khi chưa chạy route hoặc route lỗi."""
    folium.Marker(
        location=[
            float(st.session_state["route_start_lat"]),
            float(st.session_state["route_start_lon"]),
        ],
        tooltip="Start Point (A)",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(base_map)
    folium.Marker(
        location=[
            float(st.session_state["route_end_lat"]),
            float(st.session_state["route_end_lon"]),
        ],
        tooltip="Destination (B)",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(base_map)


def update_routing_click_state(map_state):
    """Lưu tọa độ click mới nhất trên bản đồ để sidebar có thể gán cho A/B."""
    last_clicked = (map_state or {}).get("last_clicked")
    if not isinstance(last_clicked, dict):
        return

    lat = last_clicked.get("lat")
    lng = last_clicked.get("lng")
    if lat is None or lng is None:
        return

    st.session_state["routing_last_clicked"] = {
        "lat": float(lat),
        "lng": float(lng),
    }


def apply_last_click_to_route_point(target: str):
    """Gán click gần nhất trên bản đồ cho điểm xuất phát hoặc đích."""
    last_clicked = st.session_state.get("routing_last_clicked")
    if not isinstance(last_clicked, dict):
        return False

    if target == "start":
        st.session_state["route_start_lat"] = float(last_clicked["lat"])
        st.session_state["route_start_lon"] = float(last_clicked["lng"])
        return True

    if target == "end":
        st.session_state["route_end_lat"] = float(last_clicked["lat"])
        st.session_state["route_end_lon"] = float(last_clicked["lng"])
        return True

    return False


def build_risk_lookup(df_predictions):
    """Tạo dictionary lookup nhanh theo tên địa phương."""
    lookup = {}
    for _, row in df_predictions.iterrows():
        lookup[row["Địa phương"]] = {
            "risk": row["Nguy cơ"],
            "prob": row["Xác suất ngập (%)"],
            "temp": row["Nhiệt độ (°C)"],
            "humidity": row["Độ ẩm (%)"],
            "rain": row["Lượng mưa (mm)"],
            "soil": row["Độ ẩm đất"],
            "tide": row["Chiều cao triều (m)"],
        }
    return lookup


def extract_coordinate_pairs(coordinates):
    """Lấy toàn bộ cặp lon/lat từ Polygon hoặc MultiPolygon."""
    points = []
    if not isinstance(coordinates, list):
        return points

    if coordinates and isinstance(coordinates[0], (int, float)) and len(coordinates) >= 2:
        return [(coordinates[0], coordinates[1])]

    for item in coordinates:
        points.extend(extract_coordinate_pairs(item))
    return points


def estimate_feature_center(feature):
    """Ước lượng tâm polygon để đặt nhãn tên địa phương."""
    geometry = feature.get("geometry", {})
    points = extract_coordinate_pairs(geometry.get("coordinates", []))
    if not points:
        return None

    avg_lon = sum(point[0] for point in points) / len(points)
    avg_lat = sum(point[1] for point in points) / len(points)
    return avg_lat, avg_lon


def geojson_is_in_target_region(geojson_data):
    """Kiểm tra GeoJSON có thực sự nằm trong vùng Huế hay không."""
    all_points = []
    for feature in geojson_data.get("features", []):
        geometry = feature.get("geometry", {})
        all_points.extend(extract_coordinate_pairs(geometry.get("coordinates", [])))

    if not all_points:
        return False

    longitudes = [point[0] for point in all_points]
    latitudes = [point[1] for point in all_points]

    min_lon, max_lon = min(longitudes), max(longitudes)
    min_lat, max_lat = min(latitudes), max(latitudes)

    return (
        106.5 <= min_lon <= 108.5
        and 106.5 <= max_lon <= 108.8
        and 15.5 <= min_lat <= 17.5
        and 15.5 <= max_lat <= 17.8
    )


def add_map_tiles(base_map):
    """Thêm nền bản đồ OpenStreetMap làm bản đồ nền mặc định."""
    folium.TileLayer(
        tiles="OpenStreetMap",
        attr="© OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True,
    ).add_to(base_map)


def add_boundary_labels(base_map, geojson_data):
    """Thêm nhãn tên địa phương lên bản đồ."""
    for feature in geojson_data.get("features", []):
        center = estimate_feature_center(feature)
        if center is None:
            continue

        name = feature.get("properties", {}).get("name", "")
        folium.Marker(
            location=center,
            icon=folium.DivIcon(
                html=(
                    "<div style='font-size: 12px; font-weight: 700; color: white; "
                    "text-shadow: 0 0 4px black, 0 0 8px black; white-space: nowrap;'>"
                    f"{name}</div>"
                )
            ),
        ).add_to(base_map)


def render_boundary_map(df_predictions):
    """Render OSM road network với Visual Line và route detour an toàn nếu có."""
    flooded_regions = build_flooded_regions_from_predictions(df_predictions)
    routing_graph = load_routing_graph()
    map_center = [DEFAULT_CENTER[0], DEFAULT_CENTER[1]]

    st.session_state["routing_last_error"] = None
    st.session_state["routing_last_summary"] = None

    if st.session_state.get("routing_enabled", False):
        try:
            route_nodes, route_map, _ = get_safe_route(
                routing_graph,
                start_lat=float(st.session_state["route_start_lat"]),
                start_lon=float(st.session_state["route_start_lon"]),
                end_lat=float(st.session_state["route_end_lat"]),
                end_lon=float(st.session_state["route_end_lon"]),
                flooded_regions=flooded_regions,
            )
            st.session_state["routing_last_summary"] = (
                f"Computed safe route across {len(route_nodes)} OSM nodes."
            )
            return route_map
        except Exception as exc:
            st.session_state["routing_last_error"] = str(exc)

    visual_map = build_visual_line_map(
        routing_graph,
        flooded_regions=flooded_regions,
        map_center=map_center,
        zoom_start=11,
    )
    add_selected_route_markers(visual_map)
    return visual_map


def build_map_popup_html(row) -> str:
    """Tạo popup dạng thẻ vuông, gọn và dễ đọc cho marker bản đồ."""
    risk_text = str(row["Nguy cơ"])
    if "Ngập nặng" in risk_text:
        badge_bg = "#7f1d1d"
        badge_fg = "#fff1f2"
    elif "Ngập nhẹ" in risk_text:
        badge_bg = "#92400e"
        badge_fg = "#fffbeb"
    else:
        badge_bg = "#166534"
        badge_fg = "#f0fdf4"

    return f"""
    <div style="
        width: 220px;
        background: #f8fafc;
        color: #0f172a;
        border-radius: 12px;
        padding: 12px 14px;
        border: 1px solid #cbd5e1;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18);
        font-family: Arial, sans-serif;
        line-height: 1.45;
    ">
        <div style="font-size: 20px; font-weight: 800; margin-bottom: 8px; color: #1e293b;">
            {row["Địa phương"]}
        </div>
        <div style="
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            background: {badge_bg};
            color: {badge_fg};
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 10px;
        ">
            {risk_text}
        </div>
        <div style="display: grid; grid-template-columns: 1fr; gap: 4px; font-size: 14px;">
            <div><b>Xác suất ngập:</b> {row["Xác suất ngập (%)"]}%</div>
            <div><b>Nhiệt độ:</b> {row["Nhiệt độ (°C)"]} °C</div>
            <div><b>Độ ẩm:</b> {row["Độ ẩm (%)"]}%</div>
            <div><b>Lượng mưa:</b> {row["Lượng mưa (mm)"]} mm</div>
            <div><b>Độ ẩm đất:</b> {row["Độ ẩm đất"]}</div>
            <div><b>Chiều cao triều:</b> {row["Chiều cao triều (m)"]} m</div>
        </div>
    </div>
    """.strip()


def risk_cell_style(value: str) -> str:
    """Tạo style đậm và dễ nhìn cho cột mức nguy cơ."""
    value = str(value)
    if "Ngập nặng" in value:
        return "background-color: #7f1d1d; color: #fff1f2; font-weight: 700;"
    if "Ngập nhẹ" in value:
        return "background-color: #92400e; color: #fffbeb; font-weight: 700;"
    if "An toàn" in value:
        return "background-color: #166534; color: #f0fdf4; font-weight: 700;"
    return "background-color: #1f2937; color: #f8fafc; font-weight: 600;"


def confidence_cell_style(value: str) -> str:
    """Tô màu cho cột độ tin cậy NOAA."""
    value = str(value)
    if "Cao" in value:
        return "background-color: #14532d; color: #ecfdf5; font-weight: 700;"
    if "Thấp" in value:
        return "background-color: #7f1d1d; color: #fff1f2; font-weight: 700;"
    return "background-color: #334155; color: #f8fafc; font-weight: 700;"


def build_contrast_styler(
    df: pd.DataFrame,
    numeric_formats: dict | None = None,
    risk_column: str | None = None,
    confidence_column: str | None = None,
):
    """Tạo Styler có độ tương phản cao để bảng web dễ đọc hơn."""
    styled_df = df.style

    if numeric_formats:
        styled_df = styled_df.format(numeric_formats)

    def zebra_rows(row):
        background = "#0f172a" if row.name % 2 == 0 else "#172033"
        return [f"background-color: {background}; color: #f8fafc;" for _ in row]

    styled_df = styled_df.apply(zebra_rows, axis=1)
    styled_df = styled_df.set_properties(
        **{
            "color": "#f8fafc",
            "border": "1px solid #334155",
            "font-size": "14px",
            "padding": "8px 10px",
        }
    )
    styled_df = styled_df.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#1e293b"),
                    ("color", "#f8fafc"),
                    ("border", "1px solid #475569"),
                    ("font-weight", "700"),
                    ("font-size", "14px"),
                    ("padding", "10px 12px"),
                    ("text-align", "center"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("border", "1px solid #334155"),
                    ("font-size", "14px"),
                    ("padding", "8px 10px"),
                ],
            },
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("width", "100%"),
                    ("background-color", "#0b1220"),
                ],
            },
        ]
    )

    if risk_column and risk_column in df.columns:
        styled_df = styled_df.map(risk_cell_style, subset=[risk_column])

    if confidence_column and confidence_column in df.columns:
        styled_df = styled_df.map(confidence_cell_style, subset=[confidence_column])

    try:
        styled_df = styled_df.hide(axis="index")
    except Exception:
        pass

    return styled_df


def render_styled_table(styler, height: int = 360):
    """Render bảng trực tiếp trong layout Streamlit để tránh khung iframe bị lệch."""
    st.dataframe(
        styler,
        use_container_width=True,
        hide_index=True,
        height=height,
    )


def render_prediction_table(df_predictions):
    """Hiển thị bảng dự báo với định dạng màu cho cột nguy cơ."""
    display_df = df_predictions.drop(columns=["Vĩ độ", "Kinh độ"])
    numeric_columns = display_df.select_dtypes(include="number").columns.tolist()
    formatter = {column: "{:.2f}" for column in numeric_columns}
    render_styled_table(
        build_contrast_styler(
            display_df,
            numeric_formats=formatter,
            risk_column="Nguy cơ",
        ),
        height=320,
    )


def render_future_forecast_table(df_future, days):
    """Hiển thị bảng dự báo cho N ngày tiếp theo với định dạng 2 chữ số thập phân."""
    if df_future.empty:
        st.warning("Hiện chưa lấy được dữ liệu forecast cho các ngày tiếp theo.")
        return

    display_df = df_future.head(days).copy() if "Ngày" in df_future.columns else df_future.copy()
    numeric_columns = display_df.select_dtypes(include="number").columns.tolist()

    render_styled_table(
        build_contrast_styler(
            display_df,
            numeric_formats={column: "{:.2f}" for column in numeric_columns},
            risk_column="Nguy cơ",
        ),
        height=430,
    )

def render_future_forecast_sections(df_future):
    """Hiển thị dự báo 7 ngày và 14 ngày tiếp theo."""
    st.subheader("Dự báo mưa và nguy cơ ngập các ngày tới")
    st.caption(
        "Dữ liệu thời tiết forecast được tổng hợp theo ngày "
    )

    if df_future.empty:
        st.warning("Không có dữ liệu forecast 7 ngày / 14 ngày để hiển thị.")
        return

    horizon_7d = df_future.head(7 * len(LOCATIONS)).copy()
    horizon_14d = df_future.head(14 * len(LOCATIONS)).copy()

    tab_7d, tab_14d = st.tabs(["7 ngày tới", "14 ngày tới"])

    with tab_7d:
        render_future_forecast_table(horizon_7d, days=len(horizon_7d))

    with tab_14d:
        render_future_forecast_table(horizon_14d, days=len(horizon_14d))


def render_sidebar_info(model, scaler):
    """Hiển thị thông tin kỹ thuật"""
    expected_features = get_expected_feature_columns(model, scaler)
    st.sidebar.subheader("Thông tin hệ thống")
    st.sidebar.write(f"Số địa phương đang theo dõi: {len(LOCATIONS)}")
    #st.sidebar.write(f"Số biến đầu vào mô hình: {len(expected_features)}")
    #st.sidebar.write("Thứ tự biến đầu vào:")
    #st.sidebar.code(", ".join(expected_features))

    #if RAW_OSM_PATH.exists():
      #  st.sidebar.info(
     #       "Đã phát hiện file OSM thô `data/geo/map`, "
    #        "nhưng app hiện dùng `thuathienhue_districts.geojson` để render ranh giới."
   #     )

  #  st.sidebar.caption(f"Frontend hiện đọc artifact từ: `{LATEST_MODELS_DIR}`")

def render_training_controls():
    """Cụm điều khiển MLOps: chọn mô hình và chạy huấn luyện theo yêu cầu."""
    st.sidebar.markdown("---")
    with st.sidebar.expander("⚙️ Tùy chọn Huấn luyện AI (MLOps)", expanded=False):
        available_models = get_all_model_names()
        selected_models = st.multiselect(
            "Chọn mô hình cần huấn luyện",
            options=available_models,
            default=available_models,
            key="selected_training_models",
        )

        if st.button(
            "⚠️ Bắt đầu Huấn luyện",
            key="start_training_button",
            use_container_width=True,
        ):
            try:
                if not selected_models:
                    st.warning("Vui lòng chọn ít nhất 1 mô hình trước khi huấn luyện.")
                else:
                    with st.spinner("🧠 Đang huấn luyện mô hình đã chọn và cập nhật `models/latest/`..."):
                        train_module = get_train_module()
                        if not hasattr(train_module, "run_training_pipeline"):
                            raise AttributeError("`analyze_and_train.py` không có hàm `run_training_pipeline()`.")
                        result = train_module.run_training_pipeline(selected_models)
                    st.toast(
                        f"Hoàn tất huấn luyện. Best model: {result['best_model_name']}",
                        icon="✅",
                    )
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.rerun()
            except Exception as exc:
                st.error(str(exc))


def render_sidebar_controls(df_predictions: pd.DataFrame, df_future: pd.DataFrame):
    """Bảng điều khiển gọn gàng cho refresh API và tải dữ liệu CSV."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Bảng Điều Khiển")

    if st.sidebar.button(
        "☁️ Cập nhật API",
        key="refresh_weather_forecast_button",
        use_container_width=True,
    ):
        try:
            st.toast("Đang cập nhật dữ liệu mới...", icon="⏳")
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        except Exception as e:
            st.error(str(e))

    realtime_csv_data = None
    future_csv_data = None

    try:
        realtime_csv_data = df_predictions.to_csv(index=False).encode("utf-8-sig")
    except Exception as e:
        st.error(str(e))

    try:
        future_csv_data = df_future.to_csv(index=False).encode("utf-8-sig")
    except Exception as e:
        st.error(str(e))

    if realtime_csv_data is not None:
        st.sidebar.download_button(
            label="📥 Tải Bảng Dự báo Hiện tại (CSV)",
            data=realtime_csv_data,
            file_name="du_bao_hien_tai.csv",
            mime="text/csv",
            key="download_realtime_csv_button",
            use_container_width=True,
        )

    if future_csv_data is not None:
        st.sidebar.download_button(
            label="📥 Tải Dự báo 14 ngày (CSV)",
            data=future_csv_data,
            file_name="du_bao_14_ngay.csv",
            mime="text/csv",
            key="download_future_csv_button",
            use_container_width=True,
        )


def render_routing_sidebar():
    """Sidebar cho nhập tọa độ, nhận click bản đồ và chạy safe routing."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛣️ Dynamic Safe Routing")
    st.sidebar.caption(
        "Nhập tọa độ hoặc click lên bản đồ Huế rồi gán cho Start Point (A) và Destination (B)."
    )

    last_clicked = st.session_state.get("routing_last_clicked")
    if isinstance(last_clicked, dict):
        st.sidebar.caption(
            f"Map click gần nhất: {last_clicked['lat']:.6f}, {last_clicked['lng']:.6f}"
        )
    else:
        st.sidebar.caption("Map click gần nhất: chưa có")

    click_col_a, click_col_b = st.sidebar.columns(2)
    if click_col_a.button("Dùng click cho A", key="use_last_click_for_start", use_container_width=True):
        if apply_last_click_to_route_point("start"):
            st.rerun()
        else:
            st.sidebar.warning("Hãy click lên bản đồ trước khi gán cho điểm A.")
    if click_col_b.button("Dùng click cho B", key="use_last_click_for_end", use_container_width=True):
        if apply_last_click_to_route_point("end"):
            st.rerun()
        else:
            st.sidebar.warning("Hãy click lên bản đồ trước khi gán cho điểm B.")

    st.sidebar.number_input(
        "Start A - Latitude",
        min_value=HUE_LAT_RANGE[0],
        max_value=HUE_LAT_RANGE[1],
        step=0.0001,
        format="%.6f",
        key="route_start_lat",
    )
    st.sidebar.number_input(
        "Start A - Longitude",
        min_value=HUE_LON_RANGE[0],
        max_value=HUE_LON_RANGE[1],
        step=0.0001,
        format="%.6f",
        key="route_start_lon",
    )
    st.sidebar.number_input(
        "Destination B - Latitude",
        min_value=HUE_LAT_RANGE[0],
        max_value=HUE_LAT_RANGE[1],
        step=0.0001,
        format="%.6f",
        key="route_end_lat",
    )
    st.sidebar.number_input(
        "Destination B - Longitude",
        min_value=HUE_LON_RANGE[0],
        max_value=HUE_LON_RANGE[1],
        step=0.0001,
        format="%.6f",
        key="route_end_lon",
    )

    run_col, clear_col = st.sidebar.columns(2)
    if run_col.button("Run Dijkstra Route", key="run_safe_route_button", use_container_width=True):
        st.session_state["routing_enabled"] = True
        st.session_state["routing_last_error"] = None
        st.rerun()
    if clear_col.button("Clear Route", key="clear_safe_route_button", use_container_width=True):
        st.session_state["routing_enabled"] = False
        st.session_state["routing_last_error"] = None
        st.session_state["routing_last_summary"] = None
        st.rerun()

    if st.session_state.get("routing_enabled", False):
        st.sidebar.info("Safe route đang được overlay bằng Polyline màu xanh dương.")
    else:
        st.sidebar.info("Bản đồ hiện hiển thị Visual Line với đường xanh lá và đỏ.")

    if st.session_state.get("routing_last_error"):
        st.sidebar.error(f"Routing error: {st.session_state['routing_last_error']}")
    elif st.session_state.get("routing_last_summary"):
        st.sidebar.success(st.session_state["routing_last_summary"])


def render_model_metrics(evaluation_metrics, runtime_info):
    """Hiển thị bảng số liệu, biểu đồ Plotly và ảnh artifact đánh giá mô hình."""
    st.subheader("Model Evaluation Metrics")

    if not evaluation_metrics:
        st.info("Chưa có evaluation metrics để hiển thị.")
        return

    metrics_rows = []
    for model_name, metric_values in evaluation_metrics.items():
        if not isinstance(metric_values, dict):
            continue
        metrics_rows.append(
            {
                "Model": metric_values.get("model_name", model_name),
                "Accuracy": metric_values.get("accuracy"),
                "Precision (Macro)": metric_values.get("precision_macro"),
                "Recall (Macro)": metric_values.get("recall_macro"),
                "F1 (Macro)": metric_values.get("f1_macro"),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    if metrics_df.empty:
        st.info("Không có dữ liệu metrics hợp lệ để hiển thị.")
        return

    metrics_df = metrics_df.sort_values(
        by="F1 (Macro)", ascending=False
    ).reset_index(drop=True)
    render_styled_table(
        build_contrast_styler(
            metrics_df,
            numeric_formats={
                "Accuracy": "{:.4f}",
                "Precision (Macro)": "{:.4f}",
                "Recall (Macro)": "{:.4f}",
                "F1 (Macro)": "{:.4f}",
            },
        ),
        height=420,
    )

    st.markdown("### So sánh F1-Score của toàn bộ mô hình")
    f1_chart_df = metrics_df.sort_values(by="F1 (Macro)", ascending=False).copy()
    fig_f1 = px.bar(
        f1_chart_df,
        x="F1 (Macro)",
        y="Model",
        orientation="h",
        color="F1 (Macro)",
        color_continuous_scale="Viridis",
        text="F1 (Macro)",
    )
    fig_f1.update_layout(
        yaxis={"categoryorder": "total ascending"},
        coloraxis_colorbar_title="F1",
        xaxis_title="F1 (Macro)",
        yaxis_title="Mô hình",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig_f1.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    st.plotly_chart(fig_f1, use_container_width=True)

    st.markdown("### Top 5 mô hình: Accuracy / Precision / Recall")
    top5_df = metrics_df.head(5).copy()
    top5_long_df = top5_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision (Macro)", "Recall (Macro)"],
        var_name="Metric",
        value_name="Score",
    )
    fig_top5 = px.bar(
        top5_long_df,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text="Score",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_top5.update_layout(
        xaxis_title="Mô hình",
        yaxis_title="Điểm số",
        legend_title="Chỉ số",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig_top5.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    st.plotly_chart(fig_top5, use_container_width=True)

    st.markdown("### Artifact trực quan")
    image_col_1, image_col_2 = st.columns(2)
    confusion_matrix_path = Path(runtime_info.get("latest_dir", str(LATEST_MODELS_DIR))) / "confusion_matrix.png"
    feature_importance_path = Path(runtime_info.get("latest_dir", str(LATEST_MODELS_DIR))) / "feature_importance.png"

    with image_col_1:
        st.markdown("**Confusion Matrix**")
        if confusion_matrix_path.exists():
            st.image(str(confusion_matrix_path), use_container_width=True)
        else:
            st.info("Chưa có ảnh `confusion_matrix.png` trong `models/latest/`.")

    with image_col_2:
        st.markdown("**Feature Importance**")
        if feature_importance_path.exists():
            st.image(str(feature_importance_path), use_container_width=True)
        else:
            st.info("Chưa có ảnh `feature_importance.png` trong `models/latest/`.")

    st.caption(
        f"Model artifact đang được dùng: `{Path(runtime_info['model_path']).name}` | "
        f"Metrics source: `{Path(runtime_info['metrics_path']).name}` | "
        f"Artifact dir: `{Path(runtime_info['latest_dir']).name}`"
    )


def main():
    """Điểm vào chính của ứng dụng."""
    apply_global_ui_theme()
    st.title("🌊 Hệ thống Dự báo Ngập lụt tại Thành phố Huế")
    st.markdown("---")

    try:
        model, scaler, evaluation_metrics, runtime_info = load_runtime_artifacts()
    except Exception as exc:
        st.error(f"❌ Không thể khởi tạo hệ thống tự động: {exc}")
        return

    df_predictions = get_realtime_prediction(model, scaler)
    df_future = get_future_predictions(model, scaler, forecast_days=14)
    current_update_time = get_processing_timestamp_display()
    initialize_routing_state()
    render_sidebar_info(model, scaler)
    render_tomorrow_nowcasting()
    render_stormglass_tide_sidebar()
    render_training_controls()

    map_col, table_col = st.columns([2.1, 1.2])

    with map_col:
        st.subheader("🗺️ Bản đồ Rủi ro Ngập lụt")
        st.caption(
            f"🕒 Dữ liệu cập nhật lúc: {current_update_time} | "
            "Click lên bản đồ để lấy tọa độ cho điểm A/B."
        )
        flood_map = render_boundary_map(df_predictions)
        map_state = st_folium(
            flood_map,
            width=None,
            height=560,
            use_container_width=True,
            returned_objects=["last_clicked"],
            key="hue_visual_line_map",
        )
        update_routing_click_state(map_state)

    render_routing_sidebar()
    render_sidebar_controls(df_predictions, df_future)

    with table_col:
        st.subheader("📊 Bảng dữ liệu Dự báo trực tuyến")
        st.caption(f"🕒 Dữ liệu cập nhật lúc: {current_update_time}")
        render_prediction_table(df_predictions)
        with st.expander(
            "🔍 Đối chiếu sự thật nền (Ground Truth Validation with NOAA CDO)",
            expanded=False,
        ):
            st.caption(
                "NOAA CDO test"
            )
            if st.button(
                "Khởi chạy luồng kiểm chứng NOAA",
                key="run_noaa_ground_truth_validation",
                use_container_width=True,
            ):
                st.session_state["run_noaa_validation"] = True

            if st.session_state.get("run_noaa_validation", False):
                with st.spinner("Đang kết nối vệ tinh NOAA..."):
                    validation_df = render_cross_validation_table(df_predictions)
                    st.session_state["noaa_validation_df"] = validation_df
                    st.session_state["run_noaa_validation"] = False

            if "noaa_validation_df" in st.session_state:
                render_styled_table(
                    build_contrast_styler(
                        st.session_state["noaa_validation_df"],
                        confidence_column="Độ tin cậy",
                    ),
                    height=280,
                )

    st.markdown("---")
    render_future_forecast_sections(df_future)

    st.markdown("---")
    render_model_metrics(evaluation_metrics, runtime_info)

    st.markdown("---")
    st.subheader("📌 Ghi chú")
    st.write(
        "Ứng dụng tự động kiểm tra dữ liệu lịch sử và mô hình khi khởi động. "
        "Nếu thiếu dữ liệu, app sẽ tự gọi `fetch_data.py`; nếu thiếu model/metrics, "
        "app sẽ tự gọi `analyze_and_train.py`. "
        "Ứng dụng lấy dữ liệu thời tiết hiện tại từ Open-Meteo, "
        "kết hợp giá trị triều cường fallback khi nguồn biển không ổn định. "
        "Phần dự báo 7 ngày / 14 ngày được tổng hợp từ forecast thời tiết theo giờ, "
        "sau đó đưa qua mô hình ML để ước lượng nguy cơ ngập theo từng ngày. "
        "Nếu muốn hiển thị ranh giới đẹp và chính xác, hãy đảm bảo file GeoJSON "
        "trong thư mục `data/geo/` là dữ liệu hành chính đã chuẩn hóa tên địa phương."
    )

    if st.button("🔄 Làm mới dữ liệu"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


if __name__ == "__main__":
    main()
