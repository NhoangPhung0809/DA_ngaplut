import json
import importlib.util
import math
import os
import subprocess
import sys
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
CTGAN_BEFORE_PATH = BASE_DIR / "data" / "data_before_ctgan.csv"
CTGAN_AFTER_PATH = BASE_DIR / "data" / "data_after_ctgan.csv"
CTGAN_DISTRIBUTION_PATH = BASE_DIR / "data" / "ctgan_class_distribution.json"
CACHE_DIR = BASE_DIR / "cache"
TRAINING_WORKER_PATH = BASE_DIR / "training_worker.py"
TRAINING_STATUS_PATH = CACHE_DIR / "training_status.json"
TRAINING_LOG_PATH = CACHE_DIR / "training_output.log"
MENU_OPTIONS = [
    "🌊 Tổng quan dự báo",
    "📊 So sánh dữ liệu CTGAN",
    "🔮 Dự báo Nâng cao (Chronos LLM)",
]

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
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_default_training_state() -> dict:
    """Trạng thái mặc định của background training job."""
    return {
        "status": "idle",
        "pid": None,
        "selected_models": [],
        "balancing_method": "auto",
        "started_at": None,
        "finished_at": None,
        "best_model_name": None,
        "run_dir": None,
        "error": None,
    }


def write_training_state(state: dict) -> None:
    """Lưu trạng thái train nền ra JSON để Streamlit và worker cùng đọc."""
    ensure_latest_models_dir()
    with TRAINING_STATUS_PATH.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, ensure_ascii=False)


def read_training_state() -> dict:
    """Đọc trạng thái train nền hiện tại."""
    ensure_latest_models_dir()
    if not TRAINING_STATUS_PATH.exists():
        return build_default_training_state()
    with TRAINING_STATUS_PATH.open("r", encoding="utf-8") as file:
        state = json.load(file)
    default_state = build_default_training_state()
    default_state.update(state)
    return default_state


def is_process_alive(pid: int | None) -> bool:
    """Kiểm tra PID còn sống hay không mà không cần thư viện ngoài."""
    if pid in (None, 0):
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def reconcile_training_state() -> dict:
    """Đồng bộ trạng thái train với tình trạng process thực tế."""
    state = read_training_state()
    if state["status"] in {"starting", "running"} and not is_process_alive(state.get("pid")):
        state["status"] = "failed"
        state["finished_at"] = state.get("finished_at") or datetime.now().isoformat(timespec="seconds")
        state["error"] = state.get("error") or "Background training process đã dừng ngoài dự kiến."
        write_training_state(state)
    return state


def read_training_log_tail(max_lines: int = 80) -> str:
    """Đọc phần cuối log train để hiển thị nhanh trên UI."""
    ensure_latest_models_dir()
    if not TRAINING_LOG_PATH.exists():
        return ""
    with TRAINING_LOG_PATH.open("r", encoding="utf-8", errors="replace") as file:
        lines = file.readlines()
    return "".join(lines[-max_lines:]).strip()


def start_background_training(selected_models: list[str], balancing_method: str = "auto") -> dict:
    """Khởi chạy worker train nền bằng process riêng, tách khỏi Streamlit session."""
    ensure_latest_models_dir()
    current_state = reconcile_training_state()
    if current_state["status"] in {"starting", "running"} and is_process_alive(current_state.get("pid")):
        raise RuntimeError("Đang có một tiến trình huấn luyện nền chạy sẵn. Hãy chờ tiến trình hiện tại hoàn tất.")

    with TRAINING_LOG_PATH.open("w", encoding="utf-8") as log_file:
        log_file.write(f"=== TRAINING JOB START REQUESTED AT {datetime.now().isoformat(timespec='seconds')} ===\n")
        log_file.write(f"Selected models: {selected_models}\n")
        log_file.write(f"Balancing method: {balancing_method}\n\n")

    stdout_handle = TRAINING_LOG_PATH.open("a", encoding="utf-8")
    command = [
        sys.executable,
        str(TRAINING_WORKER_PATH),
        "--models-json",
        json.dumps(selected_models, ensure_ascii=False),
        "--balancing-method",
        balancing_method,
    ]
    popen_kwargs = {
        "cwd": str(BASE_DIR),
        "stdout": stdout_handle,
        "stderr": subprocess.STDOUT,
        "env": {**os.environ, "PYTHONUNBUFFERED": "1"},
    }
    if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    process = subprocess.Popen(command, **popen_kwargs)
    state = {
        "status": "starting",
        "pid": process.pid,
        "selected_models": selected_models,
        "balancing_method": balancing_method,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "best_model_name": None,
        "run_dir": None,
        "error": None,
    }
    write_training_state(state)
    stdout_handle.close()
    return state


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
        "KNN",
        "SVC",
        "AdaBoost",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "ARIMA",
        "SARIMA",
        "LSTM",
        "GRU",
        "1D-CNN",
        "CNN-LSTM",
        "LSTM + XGBoost Hybrid",
    ]
    try:
        train_module = get_train_module()
        if hasattr(train_module, "list_available_models"):
            return list(train_module.list_available_models())
    except Exception:
        pass
    return fallback_names


def render_navigation_menu() -> str:
    """Điều hướng nhanh giữa dashboard chính và màn hình so sánh CTGAN."""
    st.sidebar.markdown("## Menu")
    return st.sidebar.radio(
        "Chọn màn hình",
        options=MENU_OPTIONS,
        key="app_menu_selection",
        label_visibility="collapsed",
    )


def training_artifacts_missing() -> bool:
    """Kiểm tra các artifact tối thiểu trong `models/latest/`."""
    ensure_latest_models_dir()
    return not PRIMARY_MODEL_PATH.exists() or not SCALER_PATH.exists() or not EVALUATION_METRICS_PATH.exists()


@st.cache_resource(show_spinner=False)
def initialize_system():
    """
    Khởi tạo hệ thống đúng một lần khi app bắt đầu:
    - Tự tải dữ liệu lịch sử nếu thiếu
    - Không tự train trong Streamlit nếu thiếu artifact
    """
    ensure_latest_models_dir()

    if historical_data_missing():
        with st.spinner("📥 Downloading 10-year historical data..."):
            fetch_module = dynamically_import_module("fetch_data_runtime", FETCH_SCRIPT_PATH)
            if not hasattr(fetch_module, "main"):
                raise AttributeError("`fetch_data.py` không có hàm `main()` để thực thi.")
            fetch_module.main()

    if training_artifacts_missing():
        raise FileNotFoundError(
            "Chưa có artifact huấn luyện trong `models/latest/`. "
            "Hãy khởi chạy background training từ sidebar để tạo model trước."
        )

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


@st.cache_data(show_spinner=False)
def load_ctgan_comparison_artifacts():
    """Đọc dữ liệu export trước/sau CTGAN để hiển thị nhanh trên Streamlit."""
    if not CTGAN_DISTRIBUTION_PATH.exists():
        return None

    with CTGAN_DISTRIBUTION_PATH.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    before_df = pd.read_csv(CTGAN_BEFORE_PATH) if CTGAN_BEFORE_PATH.exists() else pd.DataFrame()
    after_df = pd.read_csv(CTGAN_AFTER_PATH) if CTGAN_AFTER_PATH.exists() else pd.DataFrame()

    return {
        "summary": summary,
        "before_df": before_df,
        "after_df": after_df,
    }


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
    """Render bản đồ nhẹ để ưu tiên hiệu năng cho phần ML/deep learning."""
    risk_lookup = build_risk_lookup(df_predictions)
    base_map = folium.Map(location=[16.47, 107.63], zoom_start=10, tiles=None)
    add_map_tiles(base_map)
    heat_colormap = LinearColormap(
        colors=["#11c26d", "#ffe082", "#ff8f00", "#d50000"],
        vmin=0,
        vmax=100,
    )
    heat_colormap.caption = "Xác suất ngập (%)"

    geojson_data = load_geojson_data()

    if geojson_data and geojson_data.get("features") and geojson_is_in_target_region(geojson_data):
        def style_function(feature):
            name = feature["properties"].get("name")
            info = risk_lookup.get(name, {})
            probability = float(info.get("prob", 0))
            fill_color = heat_colormap(probability)
            return {
                "fillColor": fill_color,
                "color": "#80ffff",
                "weight": 3,
                "fillOpacity": 0.45,
            }

        def highlight_function(feature):
            _ = feature
            return {
                "color": "#fff176",
                "weight": 4,
                "fillOpacity": 0.5,
            }

        folium.GeoJson(
            geojson_data,
            name="Ranh giới hành chính",
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=folium.GeoJsonTooltip(
                fields=["name"],
                aliases=["Địa phương"],
                localize=True,
                sticky=False,
            ),
        ).add_to(base_map)
        add_boundary_labels(base_map, geojson_data)
    else:
        for _, row in df_predictions.iterrows():
            probability = float(row["Xác suất ngập (%)"])
            color = heat_colormap(probability)
            folium.CircleMarker(
                location=[row["Vĩ độ"], row["Kinh độ"]],
                radius=10,
                color="#ffffff",
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=0.88,
                tooltip=f"{row['Địa phương']} - {row['Xác suất ngập (%)']}%",
                popup=folium.Popup(
                    build_map_popup_html(row),
                    max_width=260,
                ),
            ).add_to(base_map)

        bounds = [
            [float(row["Vĩ độ"]), float(row["Kinh độ"])]
            for _, row in df_predictions.iterrows()
        ]
        if bounds:
            base_map.fit_bounds(bounds, padding=(30, 30))

    heat_colormap.add_to(base_map)
    folium.LayerControl().add_to(base_map)
    return base_map


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
    st.sidebar.info(
        "App đang dùng bản đồ nhẹ để ưu tiên hiệu năng cho huấn luyện ML/deep learning. "
        "Routing OSM tạm thời được tách khỏi màn hình chính để tránh lag và crash trình duyệt."
    )
    st.sidebar.toggle(
        "Hiển thị bản đồ nhẹ",
        value=False,
        key="show_light_map",
        help="Để tắt mặc định nhằm giảm tải cho trình duyệt khi tập trung huấn luyện mô hình.",
    )
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
        training_state = reconcile_training_state()
        available_models = get_all_model_names()
        selected_models = st.multiselect(
            "Chọn mô hình cần huấn luyện",
            options=available_models,
            default=available_models,
            key="selected_training_models",
        )
        balancing_method = st.selectbox(
            "Phương pháp cân bằng dữ liệu",
            options=["auto", "gan", "smote"],
            index=0,
            key="selected_balancing_method",
        )

        status_label_map = {
            "idle": "Chưa chạy",
            "starting": "Đang khởi tạo",
            "running": "Đang huấn luyện",
            "completed": "Hoàn tất",
            "failed": "Thất bại",
        }
        st.caption(f"Trạng thái hiện tại: {status_label_map.get(training_state['status'], training_state['status'])}")
        if training_state.get("selected_models"):
            st.caption(f"Mô hình đã chọn: {', '.join(training_state['selected_models'])}")
        if training_state.get("started_at"):
            st.caption(f"Bắt đầu: {training_state['started_at']}")
        if training_state.get("finished_at"):
            st.caption(f"Kết thúc: {training_state['finished_at']}")
        if training_state.get("best_model_name"):
            st.success(f"Best model gần nhất: {training_state['best_model_name']}")
        if training_state.get("error"):
            st.error(training_state["error"])

        if st.button(
            "⚠️ Bắt đầu Huấn luyện Nền",
            key="start_training_button",
            use_container_width=True,
            disabled=training_state["status"] in {"starting", "running"},
        ):
            try:
                if not selected_models:
                    st.warning("Vui lòng chọn ít nhất 1 mô hình trước khi huấn luyện.")
                else:
                    start_background_training(selected_models, balancing_method=balancing_method)
                    st.success("Đã khởi chạy background training. Bạn có thể tiếp tục dùng app mà không làm dừng tiến trình train.")
            except Exception as exc:
                st.error(str(exc))

        action_col_1, action_col_2 = st.columns(2)
        if action_col_1.button("Làm mới trạng thái", key="refresh_training_status_button", use_container_width=True):
            st.info("Đã cập nhật trạng thái train nền.")
        if action_col_2.button("Nạp artifact mới", key="reload_trained_artifacts_button", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Đã xóa cache artifact. App sẽ dùng artifact mới ở lần render hiện tại.")

        training_log_tail = read_training_log_tail()
        if training_log_tail:
            st.text_area(
                "Log huấn luyện gần nhất",
                value=training_log_tail,
                height=220,
                key="training_log_tail_view",
            )


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


def build_ctgan_distribution_dataframe(section_summary: dict | None) -> pd.DataFrame:
    """Chuyển class distribution JSON sang DataFrame gọn cho UI."""
    distribution = (section_summary or {}).get("class_distribution", {})
    if not distribution:
        return pd.DataFrame(columns=["Lớp", "Số lượng"])
    rows = [
        {"Lớp": str(label), "Số lượng": int(count)}
        for label, count in sorted(distribution.items(), key=lambda item: int(item[0]))
    ]
    return pd.DataFrame(rows)


def render_ctgan_dataset_panel(
    title: str,
    subtitle: str,
    summary: dict | None,
    dataset_df: pd.DataFrame,
) -> None:
    """Render 1 cột dữ liệu trước hoặc sau CTGAN."""
    st.markdown(f"### {title}")
    st.caption(subtitle)
    total_rows = (summary or {}).get("total_rows")
    sample_rows = (summary or {}).get("sample_rows")
    metric_col_1, metric_col_2 = st.columns(2)
    metric_col_1.metric("Tổng số dòng", total_rows if total_rows is not None else "N/A")
    metric_col_2.metric("Số dòng hiển thị", sample_rows if sample_rows is not None else "N/A")

    distribution_df = build_ctgan_distribution_dataframe(summary)
    if distribution_df.empty:
        st.info("Chưa có thống kê phân phối lớp.")
    else:
        st.dataframe(distribution_df, use_container_width=True, hide_index=True, height=160)

    if dataset_df.empty:
        st.info("Chưa có file dữ liệu để hiển thị.")
    else:
        st.dataframe(dataset_df, use_container_width=True, hide_index=True, height=420)


def render_ctgan_comparison_page() -> None:
    """Màn hình so sánh trực quan dữ liệu trước và sau CTGAN."""
    st.subheader("📊 So sánh dữ liệu CTGAN")
    st.caption("Trang này đọc các file export từ pipeline huấn luyện để so sánh nhanh dữ liệu trước và sau augmentation.")

    artifacts = load_ctgan_comparison_artifacts()
    if artifacts is None:
        st.info(
            "Chưa tìm thấy file export CTGAN. Hãy chạy huấn luyện với CTGAN trong `analyze_and_train.py` "
            "để tạo `data_before_ctgan.csv`, `data_after_ctgan.csv` và file thống kê phân phối lớp."
        )
        return

    summary = artifacts["summary"]
    method_used = summary.get("method_used", "Unknown")
    status = summary.get("status", "unknown")
    if method_used != "CTGAN":
        st.warning(
            f"Kết quả export gần nhất không hoàn tất bằng CTGAN thuần. Method dùng thực tế: `{method_used}` | trạng thái: `{status}`."
        )
    else:
        st.success(f"Export CTGAN sẵn sàng. Trạng thái gần nhất: `{status}`.")

    col1, col2 = st.columns(2)
    with col1:
        render_ctgan_dataset_panel(
            title="Dữ liệu Gốc (Bị mất cân bằng)",
            subtitle="Snapshot trước khi áp dụng CTGAN.",
            summary=summary.get("before"),
            dataset_df=artifacts["before_df"],
        )
    with col2:
        render_ctgan_dataset_panel(
            title="Dữ liệu sau CTGAN (Đã cân bằng)",
            subtitle="Snapshot sau khi augmentation hoàn tất.",
            summary=summary.get("after"),
            dataset_df=artifacts["after_df"],
        )


@st.cache_data(show_spinner=False)
def load_daily_rainfall_history(location_csv_name: str) -> pd.DataFrame:
    file_path = HISTORICAL_DIR / location_csv_name
    if not file_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    if "Thời_gian" not in df.columns or "Lượng_mưa_mm" not in df.columns:
        return pd.DataFrame()
    df["Thời_gian"] = pd.to_datetime(df["Thời_gian"], errors="coerce")
    df = df.dropna(subset=["Thời_gian"]).sort_values("Thời_gian")
    df["date"] = df["Thời_gian"].dt.floor("D")
    df["Lượng_mưa_mm"] = pd.to_numeric(df["Lượng_mưa_mm"], errors="coerce").fillna(0.0)
    daily = df.groupby("date", as_index=False)["Lượng_mưa_mm"].sum().sort_values("date").reset_index(drop=True)
    return daily


def get_chronos_module():
    import importlib
    import sys

    base_dir_str = str(BASE_DIR)
    if base_dir_str not in sys.path:
        sys.path.insert(0, base_dir_str)

    return importlib.import_module("chronos_predictor")


def render_chronos_llm_page() -> None:
    st.subheader("🔮 Dự báo Nâng cao (Chronos LLM)")
    st.markdown(
        "Trang này sử dụng Amazon Time Series Foundation Model `amazon/chronos-t5-mini` theo chế độ Zero-shot "
        "(không fine-tune, không huấn luyện). Model chạy trên CPU nên thời gian suy luận có thể chậm hơn ML truyền thống."
    )

    location_options = {
        "TP Huế": "TP_Hue_10years.csv",
        "Hương Thủy": "Huong_Thuy_10years.csv",
        "Phú Vang": "Phu_Vang_10years.csv",
        "Hương Trà": "Huong_Tra_10years.csv",
        "Quảng Điền": "Quang_Dien_10years.csv",
    }
    selected_location = st.selectbox(
        "Chọn khu vực (dùng lịch sử mưa để dự báo)",
        options=list(location_options.keys()),
        index=0,
        key="chronos_location_select",
    )
    prediction_length = st.slider("Số ngày dự báo", min_value=3, max_value=14, value=7, step=1)
    light_threshold = st.number_input("Ngưỡng Ngập nhẹ (mm/ngày)", min_value=0.0, value=25.0, step=1.0)
    heavy_threshold = st.number_input("Ngưỡng Ngập nặng (mm/ngày)", min_value=0.0, value=50.0, step=1.0)

    daily_history = load_daily_rainfall_history(location_options[selected_location])
    if daily_history.empty:
        st.warning("Chưa có dữ liệu lịch sử để chạy Chronos. Hãy chạy `fetch_data.py` trước.")
        return

    st.caption(f"Dữ liệu lịch sử theo ngày: {len(daily_history)} dòng")
    history_context_df = daily_history.tail(30).copy()

    if st.button("Tạo dự báo 7 ngày tới" if prediction_length == 7 else f"Tạo dự báo {prediction_length} ngày tới", key="chronos_generate_button"):
        try:
            chronos_module = get_chronos_module()
        except Exception as exc:
            st.error("Không thể nạp Chronos module/dependencies.")
            st.exception(exc)
            return

        with st.spinner("Đang chạy Chronos Zero-shot inference trên CPU..."):
            try:
                raw_hourly = pd.read_csv(HISTORICAL_DIR / location_options[selected_location])
                result = chronos_module.run_chronos_forecast(
                    raw_hourly,
                    prediction_length=int(prediction_length),
                    value_column="Lượng_mưa_mm",
                    time_column="Thời_gian",
                    light_threshold=float(light_threshold),
                    heavy_threshold=float(heavy_threshold),
                )
                plot_df = chronos_module.chronos_result_to_plotly_frame(result)
            except Exception as exc:
                st.error("Chronos inference thất bại.")
                st.exception(exc)
                return

        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=plot_df.loc[plot_df["segment"] == "history", "date"],
                y=plot_df.loc[plot_df["segment"] == "history", "value"],
                mode="lines",
                name="Lịch sử (30 ngày)",
                line=dict(color="#60a5fa", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df.loc[plot_df["segment"] == "forecast", "date"],
                y=plot_df.loc[plot_df["segment"] == "forecast", "value"],
                mode="lines+markers",
                name="Dự báo (Chronos)",
                line=dict(color="#f59e0b", width=3, dash="dash"),
            )
        )
        fig.add_hline(y=float(light_threshold), line_width=2, line_dash="dot", line_color="#fb7185")
        fig.add_hline(y=float(heavy_threshold), line_width=2, line_dash="dot", line_color="#ef4444")
        fig.update_layout(
            title=f"Chronos forecast - {selected_location}",
            xaxis_title="Ngày",
            yaxis_title="Lượng mưa (mm/ngày)",
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        forecast_table = pd.DataFrame(
            {
                "Ngày": result.forecast_dates,
                "Mưa dự báo (mm)": result.forecast_values,
                "Nhãn dự báo": result.forecast_labels,
            }
        )
        st.dataframe(forecast_table, use_container_width=True, hide_index=True, height=260)


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

    st.markdown("---")
    st.subheader("📊 Phân tích Đường cong ROC-AUC (One-vs-Rest)")
    roc_path = Path(runtime_info.get("latest_dir", str(LATEST_MODELS_DIR))) / "roc_curve_data.json"
    if not roc_path.exists():
        st.info("Chưa có `roc_curve_data.json`. Hãy train lại mô hình để xuất ROC-AUC.")
        return

    try:
        with roc_path.open("r", encoding="utf-8") as file:
            roc_payload = json.load(file)
    except Exception as exc:
        st.error("Không thể đọc `roc_curve_data.json`.")
        st.exception(exc)
        return

    if roc_payload.get("status") != "ok":
        reason = roc_payload.get("reason", "unknown")
        st.warning(f"ROC-AUC không khả dụng cho best model. Lý do: {reason}")
        return

    import plotly.graph_objects as go

    class_name_map = roc_payload.get("class_names", {})
    class_label_vi = {
        "0": "Không ngập",
        "1": "Ngập nhẹ",
        "2": "Ngập nặng",
    }
    curves = roc_payload.get("curves", {})
    model_name = roc_payload.get("model_name", "Best Model")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Baseline (AUC = 0.5)",
            line=dict(color="#94a3b8", width=2, dash="dot"),
        )
    )

    colors = {"0": "#60a5fa", "1": "#f59e0b", "2": "#ef4444"}
    for class_key in ["0", "1", "2"]:
        curve = curves.get(class_key) or {}
        fpr = curve.get("fpr") or []
        tpr = curve.get("tpr") or []
        auc_value = curve.get("auc")
        if not fpr or not tpr:
            continue
        auc_text = "N/A" if auc_value is None else f"{float(auc_value):.4f}"
        label_name = class_label_vi.get(class_key, class_name_map.get(class_key, f"Class {class_key}"))
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"Class {class_key} ({label_name}) - AUC: {auc_text}",
                line=dict(color=colors.get(class_key, "#22c55e"), width=3),
            )
        )

    fig.update_layout(
        title=f"ROC-AUC OvR cho Best Model: {model_name}",
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR)",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Điểm vào chính của ứng dụng."""
    apply_global_ui_theme()
    st.title("🌊 Hệ thống Dự báo Ngập lụt tại Thành phố Huế")
    st.markdown("---")
    selected_menu = render_navigation_menu()

    if selected_menu == "📊 So sánh dữ liệu CTGAN":
        render_training_controls()
        render_ctgan_comparison_page()
        return
    if selected_menu == "🔮 Dự báo Nâng cao (Chronos LLM)":
        render_training_controls()
        render_chronos_llm_page()
        return

    try:
        model, scaler, evaluation_metrics, runtime_info = load_runtime_artifacts()
    except Exception as exc:
        render_training_controls()
        st.warning(
            f"❌ Chưa thể nạp artifact mô hình: {exc} "
            "Bạn có thể khởi chạy background training từ sidebar rồi bấm `Nạp artifact mới` sau khi train xong."
        )
        return

    df_predictions = get_realtime_prediction(model, scaler)
    df_future = get_future_predictions(model, scaler, forecast_days=14)
    current_update_time = get_processing_timestamp_display()
    render_sidebar_info(model, scaler)
    render_tomorrow_nowcasting()
    render_stormglass_tide_sidebar()
    render_training_controls()

    map_col, table_col = st.columns([2.1, 1.2])

    with map_col:
        st.subheader("🗺️ Bản đồ Rủi ro Ngập lụt")
        st.caption(f"🕒 Dữ liệu cập nhật lúc: {current_update_time} | Chế độ bản đồ nhẹ")
        if st.session_state.get("show_light_map", False):
            flood_map = render_boundary_map(df_predictions)
            st_folium(
                flood_map,
                width=None,
                height=520,
                use_container_width=True,
                key="hue_light_map",
            )
        else:
            st.info(
                "Bản đồ đang được ẩn để giảm tải cho trình duyệt và ưu tiên hiệu năng huấn luyện ML/deep learning. "
                "Bạn có thể bật lại trong sidebar bằng tùy chọn `Hiển thị bản đồ nhẹ`."
            )

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
        st.success("Đã làm mới cache dữ liệu cho lần render hiện tại.")


if __name__ == "__main__":
    main()
