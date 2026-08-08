import json
import importlib.util
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium

# Tải biến môi trường từ file .env nếu có (ví dụ TOMTOM_API_KEY).
load_dotenv()

# Cấu hình giao diện trang Streamlit.
st.set_page_config(
    page_title="Dự báo Ngập lụt Thừa Thiên Huế",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================================================================
# ĐƯỜNG DẪN & HẰNG SỐ DÙNG CHUNG
# ==================================================================================================
BASE_DIR = Path(__file__).resolve().parent
HISTORICAL_DIR = BASE_DIR / "data" / "historical"
MODELS_DIR = BASE_DIR / "models"
LATEST_MODELS_DIR = MODELS_DIR / "latest"
PLOTS_DIR = BASE_DIR / "plots"
FETCH_SCRIPT_PATH = BASE_DIR / "fetch_data.py"
TRAIN_SCRIPT_PATH = BASE_DIR / "analyze_and_train.py"
EVALUATION_METRICS_PATH = LATEST_MODELS_DIR / "evaluation_metrics.json"
# `deployment_config.json` do `analyze_and_train.py` (hàm `save_deployment_config`) sinh ra sau mỗi
# lần huấn luyện - đây là "bản đồ chỉ dẫn" DUY NHẤT mà app.py cần đọc để biết best model hiện tại
# thuộc loại gì (sklearn_tabular / keras_sequence / hybrid_lstm_xgboost) và file artifact tương ứng
# nằm ở đâu. KHÔNG còn giả định cứng tên file `best_model.pkl`/`scaler.pkl` như trước, vì tên file
# thực tế phụ thuộc vào loại model thắng leaderboard (xem `load_deployment_model()` bên dưới).
DEPLOYMENT_CONFIG_PATH = LATEST_MODELS_DIR / "deployment_config.json"
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

# ------------------------------------------------------------------------------------------------
# CẤU HÌNH SMART ROUTING (TomTom Routing API) - dùng cho tab "🗺️ Bản đồ Tránh ngập".
# LƯU Ý BẢO MẬT: key được hardcode trực tiếp theo yêu cầu để chạy nhanh cho đồ án; `os.getenv(...)`
# vẫn được ưu tiên đọc trước nếu bạn khai báo TOMTOM_API_KEY trong file `.env` (đã có sẵn cơ chế
# `load_dotenv()` ở đầu file). Trước khi đẩy code lên GitHub public, nên XÓA key hardcode và chỉ
# giữ lại biến môi trường, để tránh lộ API key trong lịch sử commit.
# ------------------------------------------------------------------------------------------------
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "opH6G1fIc4yptvSrCqQ6iZI5yaifz1Je")
TOMTOM_ROUTING_BASE_URL = "https://api.tomtom.com/routing/1/calculateRoute"

# 5 ĐIỂM GIÁM SÁT NGẬP LỤT THỰC TẾ tại Thừa Thiên Huế (tọa độ trung tâm gần đúng của mỗi địa
# phương) - thay thế hoàn toàn cho dữ liệu giả lập (dummy) trước đây. Đây là DUY NHẤT nguồn tọa độ
# dùng cho cả bản đồ giám sát lẫn 2 ô chọn điểm đi/điểm đến của tính năng định tuyến bên dưới.
REAL_MONITORED_LOCATIONS: dict[str, tuple[float, float]] = {
    "TP Huế": (16.4637, 107.5909),
    "Hương Thủy": (16.4022, 107.6833),
    "Hương Trà": (16.4525, 107.4989),
    "Phú Vang": (16.4506, 107.7289),
    "Quảng Điền": (16.5925, 107.5256),
}

# Ánh xạ tên địa phương (khớp cột 'Địa phương' của df_predictions) -> file CSV lịch sử tương ứng
# trong data/historical/. Dùng làm dự phòng khi CHƯA có model đã triển khai (xem
# get_latest_flood_predictions()) và để lấy quan trắc gần nhất phục vụ suy luận model thật.
LOCATION_HISTORICAL_FILE: dict[str, str] = {
    "TP Huế": "TP_Hue_10years.csv",
    "Hương Thủy": "Huong_Thuy_10years.csv",
    "Hương Trà": "Huong_Tra_10years.csv",
    "Phú Vang": "Phu_Vang_10years.csv",
    "Quảng Điền": "Quang_Dien_10years.csv",
}

# Danh sách đặc trưng đầu vào của model - PHẢI khớp đúng FEATURE_COLS trong analyze_and_train.py.
FEATURE_COLS_FOR_INFERENCE = [
    "Nhiệt_độ_C",
    "Độ_ẩm_%",
    "Lượng_mưa_mm",
    "Độ_ẩm_đất",
    "Chiều_cao_triều_m",
]

# Nửa cạnh (độ) của hình vuông xấp xỉ vùng ngập được vẽ quanh 1 điểm giám sát đang có nguy cơ
# 'Ngập' - khoảng 0.015 độ vĩ/kinh ~ 1.5km, đủ nhỏ để không "nuốt" luôn toàn bộ khu vực lân cận.
FLOOD_ZONE_HALF_SIZE_DEG = 0.015


def apply_global_ui_theme():
    """Tăng độ tương phản tổng thể cho giao diện Streamlit, style riêng cho st.tabs."""
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


# ==================================================================================================
# HẠ TẦNG DÙNG CHUNG: khởi tạo hệ thống, quản lý trạng thái huấn luyện nền (MLOps)
# ==================================================================================================
def dynamically_import_module(module_name: str, module_path: Path):
    """Import module động từ đường dẫn file Python (dùng để gọi lại fetch_data.py / analyze_and_train.py)."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Không thể tạo spec cho module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def historical_data_missing() -> bool:
    """Kiểm tra bộ dữ liệu lịch sử 10 năm có đầy đủ hay chưa (dùng cho Tab EDA và bước khởi tạo)."""
    if not HISTORICAL_DIR.exists():
        return True

    existing_files = {file_path.name for file_path in HISTORICAL_DIR.glob("*.csv")}
    return not EXPECTED_HISTORICAL_FILES.issubset(existing_files)


def ensure_latest_models_dir() -> None:
    """Đảm bảo thư mục `models/latest/` và `cache/` luôn tồn tại cho frontend."""
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
    """Đọc phần cuối log train/tuning để hiển thị nhanh trên UI."""
    ensure_latest_models_dir()
    if not TRAINING_LOG_PATH.exists():
        return ""
    with TRAINING_LOG_PATH.open("r", encoding="utf-8", errors="replace") as file:
        lines = file.readlines()
    return "".join(lines[-max_lines:]).strip()


def start_background_training(selected_models: list[str], balancing_method: str = "auto") -> dict:
    """Khởi chạy worker train nền (`training_worker.py`) bằng process riêng, tách khỏi Streamlit session."""
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
    """Nạp module huấn luyện (`analyze_and_train.py`) để tái sử dụng danh sách model."""
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


def training_artifacts_missing() -> bool:
    """
    Kiểm tra các artifact tối thiểu trong `models/latest/` (dùng cho Tab Đánh giá).
    Chỉ cần `deployment_config.json` + `evaluation_metrics.json` tồn tại là đủ - KHÔNG kiểm tra cứng
    sự tồn tại của `best_model.pkl`/`scaler.pkl` nữa, vì 2 file đó chỉ tồn tại khi best model là
    sklearn_tabular; nếu best model là Deep Learning/Hybrid thì artifact thật sẽ có tên khác
    (`best_model.keras`, `best_model_xgb_head.json`...) - `deployment_config.json` mới là nguồn xác
    nhận đáng tin cậy rằng quá trình huấn luyện đã hoàn tất và có model sẵn sàng.
    """
    ensure_latest_models_dir()
    return not DEPLOYMENT_CONFIG_PATH.exists() or not EVALUATION_METRICS_PATH.exists()


@st.cache_resource(show_spinner=False)
def initialize_system():
    """
    Khởi tạo hệ thống đúng một lần khi app bắt đầu:
    - Tự tải dữ liệu lịch sử nếu thiếu (gọi lại `fetch_data.py`)
    - KHÔNG tự huấn luyện trong Streamlit nếu thiếu artifact - việc train phải được kích hoạt tường
      minh từ Tab "⚙️ Tiền xử lý & Huấn luyện" để tránh block giao diện.
    """
    ensure_latest_models_dir()

    if historical_data_missing():
        with st.spinner("📥 Đang tải dữ liệu lịch sử 10 năm..."):
            fetch_module = dynamically_import_module("fetch_data_runtime", FETCH_SCRIPT_PATH)
            if not hasattr(fetch_module, "main"):
                raise AttributeError("`fetch_data.py` không có hàm `main()` để thực thi.")
            fetch_module.main()

    if training_artifacts_missing():
        raise FileNotFoundError(
            "Chưa có artifact huấn luyện trong `models/latest/`. "
            "Hãy khởi chạy background training ở Tab 2 để tạo model trước."
        )

    return {
        "deployment_config_path": str(DEPLOYMENT_CONFIG_PATH),
        "metrics_path": str(EVALUATION_METRICS_PATH),
        "latest_dir": str(LATEST_MODELS_DIR),
    }


@st.cache_resource(show_spinner=False)
def load_evaluation_artifacts():
    """
    Nạp `evaluation_metrics.json` + `deployment_config.json` + thông tin runtime cho Tab Đánh giá.
    Lưu ý: hàm này chỉ đọc THÔNG TIN MÔ TẢ (JSON nhẹ), CHƯA nạp model thật vào bộ nhớ (không
    joblib.load()/keras.load_model() ở đây) - việc nạp model thật chỉ nên thực hiện khi thật sự cần
    suy luận, xem `load_deployment_model()` bên dưới, để tránh tốn RAM/thời gian tải chỉ để xem bảng
    chỉ số.
    """
    runtime_info = initialize_system()
    with open(runtime_info["metrics_path"], "r", encoding="utf-8") as file:
        evaluation_metrics = json.load(file)
    with open(runtime_info["deployment_config_path"], "r", encoding="utf-8") as file:
        deployment_config = json.load(file)
    return evaluation_metrics, deployment_config, runtime_info


def load_deployment_model(deployment_config: dict, latest_dir: str | Path) -> dict:
    """
    NẠP LẠI BEST MODEL ĐÚNG CÁCH THEO `deployment_config.json` (universal loading mechanism).

    ----------------------------------------------------------------------------------------------
    TẠI SAO CẦN HÀM NÀY (thay vì luôn `joblib.load("best_model.pkl")` như code cũ)?
    ----------------------------------------------------------------------------------------------
    Sau khi sửa bug chọn best model trong `analyze_and_train.py`, best model của một lần huấn luyện
    có thể là 1 trong 3 dạng hoàn toàn khác nhau về cách lưu/nạp:
      - "sklearn_tabular": model + scaler đều là object Python thuần -> nạp bằng `joblib.load()`.
      - "keras_sequence" (GRU/LSTM/1D-CNN/CNN-LSTM): model được lưu bằng định dạng Keras gốc
        (`.keras`), BẮT BUỘC nạp lại bằng `tensorflow.keras.models.load_model()` - `joblib.load()`
        sẽ LỖI hoặc nạp sai vì Keras model không phải object pickle thuần.
      - "hybrid_lstm_xgboost": có 2 thành phần lưu riêng - LSTM feature-extractor (`.keras`, nạp
        bằng `load_model()`) và đầu phân loại XGBoost (`.json`, nạp bằng `XGBClassifier().load_model()`
        - định dạng native của XGBoost, KHÔNG phải joblib/pickle).

    Hàm này đọc `model_type` trong `deployment_config.json` và tự động dùng ĐÚNG loader tương ứng,
    để phần code gọi (ví dụ nút "Kiểm tra nạp model" ở Tab Đánh giá, hoặc sau này là tab suy luận
    thời gian thực) không cần biết trước hôm nay best model là loại gì.

    Trả về dict với khóa `model_type` luôn có mặt, cộng thêm các khóa object tương ứng
    (`model`+`scaler` cho 2 loại đầu, hoặc `feature_extractor`+`classifier`+`scaler` cho hybrid).
    """
    latest_dir = Path(latest_dir)
    model_type = deployment_config.get("model_type")
    artifacts = deployment_config.get("artifacts", {})

    missing_keys = [key for key in artifacts if not (latest_dir / artifacts[key]).exists()]
    if missing_keys:
        raise FileNotFoundError(
            f"Thiếu file artifact khai báo trong deployment_config.json: {missing_keys} "
            f"(thư mục: {latest_dir})"
        )

    if model_type == "sklearn_tabular":
        model = joblib.load(latest_dir / artifacts["model_path"])
        scaler = joblib.load(latest_dir / artifacts["scaler_path"])
        return {"model_type": model_type, "model": model, "scaler": scaler}

    if model_type == "keras_sequence":
        try:
            from tensorflow.keras.models import load_model as keras_load_model
        except ImportError as exc:
            raise ImportError(
                "Best model hiện tại là Deep Learning (định dạng .keras) nhưng môi trường đang chạy "
                "app.py chưa cài TensorFlow. Cài bằng: pip install tensorflow"
            ) from exc
        model = keras_load_model(str(latest_dir / artifacts["model_path"]))
        scaler = joblib.load(latest_dir / artifacts["scaler_path"])
        return {
            "model_type": model_type,
            "model": model,
            "scaler": scaler,
            "window_size": deployment_config.get("window_size"),
        }

    if model_type == "hybrid_lstm_xgboost":
        try:
            from tensorflow.keras.models import load_model as keras_load_model
        except ImportError as exc:
            raise ImportError(
                "Best model hiện tại là Hybrid LSTM+XGBoost nhưng môi trường đang chạy app.py chưa "
                "cài TensorFlow. Cài bằng: pip install tensorflow"
            ) from exc
        from xgboost import XGBClassifier

        feature_extractor = keras_load_model(str(latest_dir / artifacts["feature_extractor_path"]))
        classifier = XGBClassifier()
        classifier.load_model(str(latest_dir / artifacts["classifier_path"]))
        scaler = joblib.load(latest_dir / artifacts["scaler_path"])
        return {
            "model_type": model_type,
            "feature_extractor": feature_extractor,
            "classifier": classifier,
            "scaler": scaler,
            "window_size": deployment_config.get("window_size"),
        }

    raise ValueError(f"Không hỗ trợ nạp model_type='{model_type}'.")


# ==================================================================================================
# TIỆN ÍCH HIỂN THỊ BẢNG & NHẬN XÉT (dùng chung cho nhiều tab)
# ==================================================================================================
def render_chart_discussion(text: str) -> None:
    """Hiển thị đoạn nhận xét/diễn giải ngay bên dưới một bảng hoặc biểu đồ - áp dụng cho MỌI
    bảng/biểu đồ trong app để giảng viên thấy được phần "đọc số liệu", không chỉ số liệu thô."""
    st.info(text)


def render_full_width_image(image_path: str) -> None:
    """
    Hiển thị ảnh full-width, TƯƠNG THÍCH NHIỀU PHIÊN BẢN STREAMLIT khác nhau giữa máy dev và server.

    Streamlit đã đổi cách khai báo "ảnh chiếm full chiều rộng" qua nhiều phiên bản:
    - Bản cũ (vd 1.38.x): dùng `use_column_width=True`.
    - Bản mới hơn: đổi tên thành `use_container_width=True`.
    - Bản mới nhất: cả 2 tham số trên đều bị loại bỏ (deprecated), thay bằng `width="stretch"`.

    Do máy dev/test và server Ubuntu triển khai có thể cài 2 phiên bản Streamlit khác nhau (như đã
    gặp: server báo warning "use_column_width đã bị loại bỏ, dùng width thay thế"), hàm này THỬ dùng
    API mới (`width="stretch"`) trước; nếu môi trường đang chạy là bản Streamlit cũ chưa hỗ trợ tham
    số này (ném `TypeError` vì tên tham số không tồn tại), tự động lùi về `use_column_width=True`.
    Nhờ vậy CÙNG MỘT đoạn code chạy đúng trên cả 2 môi trường, không cần biết trước server cài bản nào.
    """
    try:
        st.image(image_path, width="stretch")
    except TypeError:
        st.image(image_path, use_column_width=True)


def build_contrast_styler(df: pd.DataFrame, numeric_formats: dict | None = None):
    """Tạo Styler có độ tương phản cao (nền tối, chữ sáng, sọc ngựa vằn) để bảng web dễ đọc hơn."""
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


# ==================================================================================================
# TAB 1 - 📊 KHÁM PHÁ DỮ LIỆU (EDA)
# ==================================================================================================
@st.cache_data(show_spinner=False)
def load_eda_sample_dataframe() -> pd.DataFrame:
    """
    Đọc và gộp dữ liệu lịch sử từ `data/historical/*.csv` để hiển thị cho Tab EDA.
    Dùng `on_bad_lines="skip"` vì một số file CSV lịch sử có cấu trúc cột không đồng nhất tuyệt đối
    giữa các lần ghi (do được append qua nhiều lần chạy `fetch_data.py` ở các phiên bản khác nhau) -
    bỏ qua vài dòng lỗi hiếm gặp còn hơn là làm crash toàn bộ Tab EDA.
    """
    if not HISTORICAL_DIR.exists():
        return pd.DataFrame()

    csv_files = sorted(HISTORICAL_DIR.glob("*.csv"))
    if not csv_files:
        return pd.DataFrame()

    frames = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, on_bad_lines="skip", engine="python")
        except Exception:
            continue
        df["Địa phương"] = csv_file.stem.replace("_10years", "").replace("_", " ").strip()
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined_df = pd.concat(frames, ignore_index=True)
    if "Thời_gian" in combined_df.columns:
        combined_df["Thời_gian"] = pd.to_datetime(combined_df["Thời_gian"], errors="coerce")
        combined_df = combined_df.dropna(subset=["Thời_gian"]).sort_values("Thời_gian").reset_index(drop=True)

    return combined_df


def render_eda_tab() -> None:
    """
    Nội dung Tab 1 - Khám phá Dữ liệu (EDA), bước ĐẦU TIÊN của vòng đời Data Science.
    Bố cục: 2 cột song song (Raw Data | Thống kê mô tả) phía trên, tiếp theo là 1 khối biểu đồ phân
    phối/tương quan, và cuối cùng là khối xử lý giá trị thiếu/ngoại lai - mỗi khối đặt trong
    `st.expander` để trang không bị dồn cục, người xem chỉ mở phần mình cần.
    """
    st.subheader("📊 Khám phá Dữ liệu (EDA)")
    st.caption(
        "Bước 1/4 của pipeline: hiểu dữ liệu trước khi làm sạch và huấn luyện. Các biểu đồ tĩnh bên dưới "
        "được sinh sẵn bởi `eda_analysis.py` (chạy `python eda_analysis.py` để làm mới sau khi có dữ liệu mới)."
    )

    eda_df = load_eda_sample_dataframe()

    # ---- Hàng 1: bố cục 2 CỘT song song bằng st.columns ----
    col_raw, col_stats = st.columns(2)

    with col_raw:
        with st.expander("🔎 Dữ liệu thô (Raw Data)", expanded=True):
            # TODO: nếu bạn có logic đọc dữ liệu thô khác (ví dụ đọc trực tiếp từ 1 file cụ thể),
            # hãy thay `eda_df` bên dưới bằng DataFrame của bạn.
            if eda_df.empty:
                st.info("Chưa có dữ liệu lịch sử trong `data/historical/`. Hãy chạy `fetch_data.py` trước.")
            else:
                st.dataframe(eda_df.head(20), use_container_width=True, hide_index=True)
                render_chart_discussion(
                    f"Bảng hiển thị 20 dòng đầu (`df.head()`) trong tổng số {len(eda_df):,} dòng đã gộp từ "
                    f"{eda_df['Địa phương'].nunique()} địa phương - gồm các biến khí tượng, thủy văn theo giờ "
                    "và nhãn nguy cơ ngập. Đây là nguyên liệu đầu vào cho toàn bộ pipeline ở các tab phía sau."
                )

    with col_stats:
        with st.expander("📐 Thống kê mô tả (Descriptive Statistics)", expanded=True):
            # TODO: dán code `df.describe()` / thống kê chi tiết hơn của bạn (ví dụ describe theo
            # từng địa phương, theo từng lớp nguy cơ ngập...) vào đây.
            if eda_df.empty:
                st.info("Chưa có dữ liệu để thống kê.")
            else:
                numeric_df = eda_df.select_dtypes(include="number")
                st.dataframe(numeric_df.describe().T, use_container_width=True)
                render_chart_discussion(
                    "Bảng `describe()` cho thấy khoảng giá trị, trung bình và độ lệch chuẩn của từng biến số - "
                    "cơ sở để phát hiện đơn vị đo bất thường (ví dụ độ ẩm âm, nhiệt độ ngoài khoảng hợp lý) "
                    "trước khi đưa dữ liệu vào bước tiền xử lý ở Tab 2."
                )

    st.markdown("---")

    # ---- Hàng 2: biểu đồ phân phối / tương quan (ảnh tĩnh do eda_analysis.py sinh sẵn) ----
    with st.expander("📈 Phân phối dữ liệu & Ma trận tương quan (Distribution / Heatmap)", expanded=True):
        # TODO: đây là placeholder hiển thị ẢNH TĨNH từ `eda_analysis.py` để tránh vẽ lại biểu đồ nặng
        # mỗi lần Streamlit rerun. Nếu muốn biểu đồ TƯƠNG TÁC, có thể thay bằng `px.imshow()` (heatmap)
        # hoặc `px.histogram()` (phân phối) ngay trong hàm này.
        eda_chart_files = {
            "Ma trận tương quan (Heatmap)": PLOTS_DIR / "correlation_heatmap.png",
            "Phân bố lớp mục tiêu": PLOTS_DIR / "class_distribution.png",
            "Xu hướng mưa & tỷ lệ ngập theo tháng": PLOTS_DIR / "monthly_trend.png",
            "Tỷ lệ ngập theo địa phương": PLOTS_DIR / "flood_share_by_location.png",
            "Phân bố lượng mưa theo lớp": PLOTS_DIR / "rain_distribution_by_class.png",
            "Phân bố triều cường theo lớp": PLOTS_DIR / "tide_distribution_by_class.png",
        }
        chart_columns = st.columns(2)
        for index, (chart_title, chart_path) in enumerate(eda_chart_files.items()):
            with chart_columns[index % 2]:
                st.markdown(f"**{chart_title}**")
                if chart_path.exists():
                    render_full_width_image(str(chart_path))
                else:
                    st.info(f"Chưa có `{chart_path.name}`. Hãy chạy `python eda_analysis.py` để sinh ảnh.")
        render_chart_discussion(
            "Các biểu đồ trên tổng hợp quan hệ tương quan giữa các biến, phân phối lớp mục tiêu, xu hướng "
            "mưa/ngập theo mùa vụ, và tỷ lệ ngập theo địa phương - cung cấp căn cứ định lượng cho khuyến "
            "nghị quản trị ở Tab 3 (ví dụ: tháng nào, khu vực nào cần ưu tiên nguồn lực phòng chống ngập)."
        )

    # ---- Hàng 3: xử lý giá trị thiếu / ngoại lai ----
    with st.expander("🧹 Xử lý giá trị thiếu & ngoại lai (Missing Value / Outlier)", expanded=False):
        # TODO: dán logic phát hiện/xử lý outlier thật của bạn (IQR, Z-score theo từng lớp...) vào đây.
        if eda_df.empty:
            st.info("Chưa có dữ liệu để kiểm tra.")
        else:
            missing_summary = eda_df.isna().sum().rename("Số lượng thiếu").to_frame()
            missing_summary["Tỷ lệ thiếu (%)"] = (missing_summary["Số lượng thiếu"] / len(eda_df) * 100).round(2)
            st.dataframe(missing_summary, use_container_width=True)
            render_chart_discussion(
                "Bảng trên thống kê số lượng và tỷ lệ giá trị thiếu theo từng cột - căn cứ để quyết định "
                "chiến lược xử lý (loại bỏ, nội suy, hay điền giá trị trung vị) ở Tab 2. Với biến ngoại lai "
                "(outlier), khuyến nghị dùng phương pháp IQR hoặc Z-score TÍNH RIÊNG cho từng lớp nguy cơ ngập, "
                "vì giá trị mưa/triều cực đoan trong lớp `Ngập nặng` là TÍN HIỆU THẬT có giá trị dự báo, "
                "không nên loại bỏ nhầm như outlier thông thường."
            )


# ==================================================================================================
# TAB 2 - ⚙️ TIỀN XỬ LÝ & HUẤN LUYỆN
# ==================================================================================================
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


def build_ctgan_distribution_discussion(distribution_df: pd.DataFrame, title: str) -> str:
    """Sinh nhận xét động về mức độ mất cân bằng lớp cho một bảng phân phối CTGAN."""
    if distribution_df.empty:
        return "Chưa có đủ dữ liệu phân phối lớp để đánh giá mức độ cân bằng."

    max_row = distribution_df.loc[distribution_df["Số lượng"].idxmax()]
    min_row = distribution_df.loc[distribution_df["Số lượng"].idxmin()]
    imbalance_ratio = max_row["Số lượng"] / max(min_row["Số lượng"], 1)

    return (
        f"Ở bộ dữ liệu **{title}**, lớp `{max_row['Lớp']}` có {int(max_row['Số lượng']):,} quan sát trong khi lớp "
        f"`{min_row['Lớp']}` chỉ có {int(min_row['Số lượng']):,} quan sát — tỷ lệ mất cân bằng khoảng "
        f"{imbalance_ratio:.1f} lần. Tỷ lệ này càng gần 1 thì mô hình càng ít bị thiên lệch về lớp đa số khi huấn luyện."
    )


@st.cache_data(show_spinner=False)
def load_ctgan_comparison_artifacts():
    """Đọc dữ liệu export trước/sau CTGAN (do `analyze_and_train.py` xuất ra) để hiển thị nhanh trên Streamlit."""
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


def render_ctgan_dataset_panel(title: str, subtitle: str, summary: dict | None, dataset_df: pd.DataFrame) -> None:
    """Render 1 cột dữ liệu trước hoặc sau CTGAN (dùng cho cả 2 cột trong `render_ctgan_section`)."""
    st.markdown(f"#### {title}")
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
        render_chart_discussion(build_ctgan_distribution_discussion(distribution_df, title))

    if dataset_df.empty:
        st.info("Chưa có file dữ liệu để hiển thị.")
    else:
        st.dataframe(dataset_df, use_container_width=True, hide_index=True, height=320)


def render_ctgan_section() -> None:
    """
    Khối 'Cân bằng dữ liệu' bên trong Tab 2: so sánh trực quan dữ liệu TRƯỚC và SAU khi áp dụng CTGAN
    (hoặc SMOTE fallback) - đây là bước bắt buộc trước khi huấn luyện vì dữ liệu ngập lụt luôn mất cân
    bằng nặng giữa 3 lớp (An toàn / Ngập nhẹ / Ngập nặng).
    """
    artifacts = load_ctgan_comparison_artifacts()
    if artifacts is None:
        st.info(
            "Chưa tìm thấy file export CTGAN. Hãy chạy huấn luyện trong `analyze_and_train.py` để tạo "
            "`data_before_ctgan.csv`, `data_after_ctgan.csv` và file thống kê phân phối lớp."
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


def render_training_controls_panel() -> None:
    """
    Cụm điều khiển MLOps: chọn mô hình, chạy huấn luyện/tinh chỉnh NỀN (background) và xem log.
    Đặt ngay trong Tab 2 (thay vì giấu trong sidebar như trước) vì đây chính là hành động trung tâm
    của bước "Huấn luyện" trong pipeline, nên cần hiển thị rõ ràng ở khu vực nội dung chính.
    """
    training_state = reconcile_training_state()
    available_models = get_all_model_names()

    control_col, status_col = st.columns([1.3, 1])
    with control_col:
        selected_models = st.multiselect(
            "Chọn mô hình cần huấn luyện / tinh chỉnh",
            options=available_models,
            default=available_models,
            key="selected_training_models",
        )
        balancing_method = st.selectbox(
            "Phương pháp cân bằng dữ liệu",
            options=["auto", "gan", "smote"],
            index=0,
            key="selected_balancing_method",
            help="'auto' ưu tiên CTGAN, tự fallback sang SMOTE nếu thiếu thư viện hoặc lỗi khi chạy.",
        )
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
                    st.success("Đã khởi chạy background training - có thể tiếp tục dùng app trong lúc chờ.")
            except Exception as exc:
                st.error(str(exc))

    with status_col:
        status_label_map = {
            "idle": "Chưa chạy",
            "starting": "Đang khởi tạo",
            "running": "Đang huấn luyện",
            "completed": "Hoàn tất",
            "failed": "Thất bại",
        }
        st.metric("Trạng thái", status_label_map.get(training_state["status"], training_state["status"]))
        if training_state.get("best_model_name"):
            st.success(f"Best model gần nhất: {training_state['best_model_name']}")
        if training_state.get("error"):
            st.error(training_state["error"])
        if st.button("🔄 Nạp lại artifact mới nhất", key="reload_trained_artifacts_button", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Đã xóa cache - dữ liệu/metrics sẽ được nạp lại ở lần chạy tiếp theo.")

    training_log_tail = read_training_log_tail()
    if training_log_tail:
        st.text_area(
            "Log huấn luyện / tinh chỉnh gần nhất",
            value=training_log_tail,
            height=220,
            key="training_log_tail_view",
        )
    else:
        st.caption("Chưa có log huấn luyện nào - log sẽ xuất hiện tại đây sau khi bấm 'Bắt đầu Huấn luyện Nền'.")


def render_preprocessing_training_tab() -> None:
    """
    Nội dung Tab 2 - Tiền xử lý & Huấn luyện, bước THỨ HAI của vòng đời Data Science.
    Bố cục: 2 cột song song (Dữ liệu đã làm sạch | Chia Train/Test) phía trên, tiếp theo là khối
    Cân bằng dữ liệu (CTGAN) và khối Log Tinh chỉnh siêu tham số - mỗi khối 1 `st.expander`.
    """
    st.subheader("⚙️ Tiền xử lý & Huấn luyện")
    st.caption(
        "Bước 2/4 của pipeline: làm sạch dữ liệu, chia tập train/test đúng đặc thù chuỗi thời gian, "
        "cân bằng lớp thiểu số, và tinh chỉnh siêu tham số (Optuna / GridSearchCV)."
    )

    col_clean, col_split = st.columns(2)

    with col_clean:
        with st.expander("🧼 Dữ liệu đã làm sạch (Cleaned Data)", expanded=True):
            # TODO: dán code hiển thị dữ liệu SAU khi qua preprocess_data()/preprocess_features()
            # (xem `analyze_and_train.py`) vào đây, ví dụ: st.dataframe(cleaned_df.head(20)).
            st.info(
                "Placeholder: hiển thị `df.head()` của dữ liệu SAU khi xử lý missing values, ép kiểu dữ liệu, "
                "và tạo lại nhãn 3 lớp theo rule-based (xem hàm `preprocess_data()` trong `analyze_and_train.py`)."
            )
            st.code(
                "# Ví dụ dán code thật của bạn vào đây:\n"
                "# from analyze_and_train import load_and_concatenate_csvs, preprocess_features\n"
                "# raw_df = load_and_concatenate_csvs()\n"
                "# cleaned_df = preprocess_features(raw_df)\n"
                "# st.dataframe(cleaned_df.head(20))",
                language="python",
            )

    with col_split:
        with st.expander("✂️ Chia tập Train / Test (Data Splitting)", expanded=True):
            st.markdown(
                "- **Tỷ lệ chia**: 80% Train / 20% Test.\n"
                "- **Phương pháp**: chia theo MỐC THỜI GIAN (`shuffle=False`) - tập Test luôn nằm SAU "
                "tập Train, không chia ngẫu nhiên.\n"
            )
            # QUAN TRỌNG - GIẢI THÍCH KỸ THUẬT DÙNG CHO PHẦN BẢO VỆ LUẬN VĂN:
            # Với dữ liệu chuỗi thời gian, TUYỆT ĐỐI không dùng train_test_split(shuffle=True) hay K-Fold
            # thông thường, vì sẽ để lọt thông tin TƯƠNG LAI vào tập huấn luyện (data leakage), khiến độ
            # chính xác đánh giá bị "ảo" (cao hơn thực tế khi triển khai thật). Thay vào đó nên dùng
            # `sklearn.model_selection.TimeSeriesSplit` - một dạng cross-validation walk-forward: mỗi fold
            # sau luôn dùng NHIỀU dữ liệu quá khứ hơn để dự báo một đoạn TƯƠNG LAI kế tiếp, đảm bảo mọi lần
            # đánh giá đều mô phỏng đúng bối cảnh "chỉ biết quá khứ, dự báo tương lai" như khi vận hành thực tế.
            st.code(
                "from sklearn.model_selection import TimeSeriesSplit\n\n"
                "# TimeSeriesSplit đảm bảo không rò rỉ dữ liệu tương lai (data leakage) vào tập huấn luyện\n"
                "splitter = TimeSeriesSplit(n_splits=5)\n"
                "for train_idx, test_idx in splitter.split(X):\n"
                "    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]\n"
                "    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]\n"
                "    # TODO: fit/evaluate mô hình cho từng fold tại đây",
                language="python",
            )
            render_chart_discussion(
                "TimeSeriesSplit khác K-Fold thông thường ở chỗ nó KHÔNG xáo trộn dữ liệu - đảm bảo mọi "
                "lần đánh giá đều mô phỏng đúng bối cảnh dự báo thực tế (chỉ dùng dữ liệu quá khứ để dự "
                "báo tương lai), tránh đánh giá bị 'ảo' do rò rỉ thông tin tương lai."
            )

    st.markdown("---")

    with st.expander("⚖️ Cân bằng dữ liệu (CTGAN Before / After)", expanded=False):
        st.caption(
            "Đọc file export từ `analyze_and_train.py` để so sánh dữ liệu trước/sau khi cân bằng lớp "
            "thiểu số bằng CTGAN (tự fallback sang SMOTE nếu cần)."
        )
        render_ctgan_section()

    with st.expander("🎯 Log Tinh chỉnh Siêu tham số (Optuna / GridSearchCV)", expanded=True):
        # TODO: dán code hiển thị log/kết quả thật từ Optuna Study hoặc GridSearchCV.cv_results_ vào đây,
        # ví dụ: st.dataframe(study.trials_dataframe()) hoặc st.dataframe(pd.DataFrame(grid_search.cv_results_)).
        # Xem file `hyperparameter_tuning.py` (hàm tune_random_forest_gridsearch / tune_xgboost_optuna /
        # tune_lstm_optuna) để lấy code tuning đầy đủ - GridSearchCV phù hợp cho Random Forest vì không
        # gian tham số nhỏ, rời rạc; Optuna phù hợp cho XGBoost/LSTM vì không gian tham số lớn, liên tục
        # và có yếu tố kiến trúc (số lớp/số unit của LSTM) mà GridSearchCV không biểu diễn hiệu quả được.
        st.info(
            "Placeholder: dán bảng log/kết quả tuning thật (Optuna `trials_dataframe()` hoặc "
            "`GridSearchCV.cv_results_`) tại đây. Bên dưới là cụm điều khiển MLOps thật, dùng để "
            "khởi chạy huấn luyện nền và xem log ngay trong ứng dụng."
        )
        render_training_controls_panel()


# ==================================================================================================
# TAB 3 - 📈 ĐÁNH GIÁ MÔ HÌNH
# ==================================================================================================
def render_model_metrics(evaluation_metrics, deployment_config, runtime_info) -> None:
    """Hiển thị bảng số liệu, biểu đồ Plotly và ảnh artifact đánh giá mô hình (Model Comparison Metrics)."""
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

    metrics_df = metrics_df.sort_values(by="F1 (Macro)", ascending=False).reset_index(drop=True)
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
    best_metrics_row = metrics_df.iloc[0]
    worst_metrics_row = metrics_df.iloc[-1]
    render_chart_discussion(
        f"Xét theo F1-Score (macro), **{best_metrics_row['Model']}** đang là mô hình tốt nhất "
        f"({best_metrics_row['F1 (Macro)']:.4f}), trong khi **{worst_metrics_row['Model']}** có kết quả thấp nhất "
        f"({worst_metrics_row['F1 (Macro)']:.4f}) trong số các mô hình đã huấn luyện. Chênh lệch F1 giữa hai mô hình "
        f"khoảng {(best_metrics_row['F1 (Macro)'] - worst_metrics_row['F1 (Macro)']):.4f} điểm, cho thấy việc lựa chọn "
        "đúng thuật toán có tác động đáng kể đến chất lượng cảnh báo ngập trước khi đưa vào vận hành thực tế."
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
    render_chart_discussion(
        f"Biểu đồ xếp hạng toàn bộ mô hình theo F1-Score cho thấy nhóm mô hình dẫn đầu tách biệt khá rõ so với "
        f"nhóm cuối bảng. Mô hình `{best_metrics_row['Model']}` hiện đang được chọn làm best model, đảm bảo cân "
        "bằng tốt nhất giữa khả năng phát hiện đúng các trường hợp ngập (Recall) và hạn chế cảnh báo sai (Precision)."
    )

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
    top5_best_precision = top5_df.loc[top5_df["Precision (Macro)"].idxmax(), "Model"]
    top5_best_recall = top5_df.loc[top5_df["Recall (Macro)"].idxmax(), "Model"]
    render_chart_discussion(
        f"Trong Top 5 mô hình, `{top5_best_precision}` đạt Precision cao nhất (ít cảnh báo giả nhất), còn "
        f"`{top5_best_recall}` đạt Recall cao nhất (bỏ sót ít trường hợp ngập nhất). Với bài toán cảnh báo thiên tai, "
        "Recall thường quan trọng hơn Precision vì bỏ sót một đợt ngập thực tế gây hậu quả nghiêm trọng hơn một lần "
        "cảnh báo dư thừa — đây là yếu tố cần cân nhắc khi lựa chọn mô hình triển khai chính thức."
    )

    st.markdown("### Artifact trực quan")
    image_col_1, image_col_2 = st.columns(2)
    confusion_matrix_path = Path(runtime_info.get("latest_dir", str(LATEST_MODELS_DIR))) / "confusion_matrix.png"
    feature_importance_path = Path(runtime_info.get("latest_dir", str(LATEST_MODELS_DIR))) / "feature_importance.png"

    with image_col_1:
        st.markdown("**Confusion Matrix**")
        if confusion_matrix_path.exists():
            render_full_width_image(str(confusion_matrix_path))
        else:
            st.info("Chưa có ảnh `confusion_matrix.png` trong `models/latest/`.")

    with image_col_2:
        st.markdown("**Feature Importance**")
        if feature_importance_path.exists():
            render_full_width_image(str(feature_importance_path))
        else:
            st.info("Chưa có ảnh `feature_importance.png` trong `models/latest/`.")

    if confusion_matrix_path.exists() or feature_importance_path.exists():
        render_chart_discussion(
            "Confusion Matrix cho biết mô hình đang nhầm lẫn giữa lớp nào với lớp nào nhiều nhất — cần đặc biệt "
            "lưu ý nếu các trường hợp `Ngập nặng` bị dự đoán nhầm thành `An toàn` hoặc `Ngập nhẹ`, vì đây là loại "
            "sai số nguy hiểm nhất trong bài toán cảnh báo. Feature Importance cho thấy biến khí tượng - thủy văn "
            "nào (mưa, độ ẩm đất, triều cường...) đóng góp nhiều nhất vào quyết định của mô hình."
        )

    model_type = deployment_config.get("model_type", "unknown")
    model_type_label_map = {
        "sklearn_tabular": "Machine Learning (sklearn/XGBoost - joblib)",
        "keras_sequence": "Deep Learning (Keras .keras)",
        "hybrid_lstm_xgboost": "Hybrid LSTM + XGBoost (Keras + XGBoost native)",
    }
    artifact_file_names = ", ".join(
        f"`{name}`" for name in (deployment_config.get("artifacts") or {}).values()
    )
    st.caption(
        f"Best model: `{deployment_config.get('model_name')}` | "
        f"Loại triển khai: {model_type_label_map.get(model_type, model_type)} | "
        f"Artifact: {artifact_file_names or '—'} | "
        f"Artifact dir: `{Path(runtime_info['latest_dir']).name}`"
    )

    # ---- Kiểm tra nạp lại model thật bằng đúng loader tương ứng model_type (load_deployment_model) ----
    # Nút này giúp xác nhận NGAY TRÊN GIAO DIỆN rằng cơ chế lưu vạn năng ở `analyze_and_train.py` và
    # cơ chế nạp vạn năng ở `app.py` khớp nhau - tức deployment_config.json không chỉ là văn bản mô tả
    # suông mà thực sự dùng để nạp lại được model gốc, sẵn sàng cho bước suy luận sau này.
    if st.button("🔧 Kiểm tra nạp model triển khai", key="test_load_deployment_model_button"):
        try:
            with st.spinner("Đang nạp model theo deployment_config.json..."):
                loaded = load_deployment_model(deployment_config, runtime_info["latest_dir"])
            if loaded["model_type"] == "hybrid_lstm_xgboost":
                st.success(
                    f"✅ Nạp thành công model_type=`{loaded['model_type']}`: "
                    f"feature_extractor=`{type(loaded['feature_extractor']).__name__}`, "
                    f"classifier=`{type(loaded['classifier']).__name__}`, "
                    f"scaler=`{type(loaded['scaler']).__name__}`."
                )
            else:
                st.success(
                    f"✅ Nạp thành công model_type=`{loaded['model_type']}`: "
                    f"model=`{type(loaded['model']).__name__}`, scaler=`{type(loaded['scaler']).__name__}`."
                )
        except Exception as exc:
            st.error(f"❌ Nạp model thất bại: {exc}")

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
    class_label_vi = {"0": "Không ngập", "1": "Ngập nhẹ", "2": "Ngập nặng"}
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

    auc_by_class = {
        class_key: curves.get(class_key, {}).get("auc")
        for class_key in ["0", "1", "2"]
        if curves.get(class_key, {}).get("auc") is not None
    }
    if auc_by_class:
        best_class_key = max(auc_by_class, key=auc_by_class.get)
        worst_class_key = min(auc_by_class, key=auc_by_class.get)
        best_class_label = class_label_vi.get(best_class_key, best_class_key)
        worst_class_label = class_label_vi.get(worst_class_key, worst_class_key)
        render_chart_discussion(
            f"Đường cong ROC cho thấy mô hình phân biệt tốt nhất lớp `{best_class_label}` với AUC = "
            f"{auc_by_class[best_class_key]:.4f} (càng gần 1 càng tốt), trong khi lớp `{worst_class_label}` có AUC "
            f"thấp nhất, khoảng {auc_by_class[worst_class_key]:.4f}. Nếu lớp `Ngập nặng` có AUC thấp, đây là tín hiệu "
            "cần bổ sung thêm dữ liệu ngập nặng (qua CTGAN/SMOTE) hoặc tinh chỉnh ngưỡng cảnh báo."
        )


def render_evaluation_tab() -> None:
    """
    Nội dung Tab 3 - Đánh giá Mô hình, bước THỨ BA của vòng đời Data Science.
    Bố cục: khối so sánh chỉ số (tái sử dụng `render_model_metrics`) rồi đến khối "Kết luận quản trị" -
    phần bắt buộc phải có để nối kết quả kỹ thuật với ý nghĩa thực tiễn cho người ra quyết định.
    """
    st.subheader("📈 Đánh giá Mô hình")
    st.caption("Bước 3/4 của pipeline: so sánh hiệu năng các mô hình đã huấn luyện và rút ra khuyến nghị quản trị.")

    try:
        evaluation_metrics, deployment_config, runtime_info = load_evaluation_artifacts()
    except Exception as exc:
        st.warning(
            f"❌ Chưa thể nạp evaluation metrics: {exc} "
            "Hãy khởi chạy huấn luyện ở Tab 2 (⚙️ Tiền xử lý & Huấn luyện) trước."
        )
        return

    with st.expander("📊 So sánh chỉ số mô hình (F1-Score / Precision / Recall)", expanded=True):
        render_model_metrics(evaluation_metrics, deployment_config, runtime_info)

    with st.expander("🏛️ Nhận định & Kết luận quản trị (Managerial Insights)", expanded=True):
        # TODO: thay nội dung placeholder này bằng nhận định THẬT rút ra từ kết quả mô hình + EDA (Tab 1)
        # của bạn - đây là phần quan trọng nhất khi bảo vệ luận văn vì nối kết quả kỹ thuật với hành động
        # quản trị thực tế, không chỉ dừng lại ở con số.
        st.markdown(
            """
            **Gợi ý cấu trúc phần Kết luận quản trị (điền số liệu thật của bạn vào đây):**

            1. **Hiệu năng mô hình đề xuất triển khai** — nêu tên mô hình tốt nhất, F1-macro, và lý do
               chọn (cân bằng Precision/Recall, ưu tiên Recall cho lớp `Ngập nặng` vì bỏ sót nguy hiểm
               hơn cảnh báo dư).
            2. **Khu vực ưu tiên** — dựa trên EDA ở Tab 1 (`flood_share_by_location.png`), địa phương nào
               có tần suất ngập cao nhất cần được ưu tiên đầu tư trạm quan trắc / lực lượng ứng trực.
            3. **Thời điểm ưu tiên** — dựa trên `monthly_trend.png`, giai đoạn nào trong năm cần tăng
               cường giám sát và chuẩn bị phương án sơ tán.
            4. **Rủi ro còn tồn đọng** — nêu giới hạn của mô hình (ví dụ AUC lớp `Ngập nặng` thấp do
               thiếu dữ liệu) và đề xuất hướng khắc phục (thu thập thêm dữ liệu, cải thiện CTGAN,
               dùng `hyperparameter_tuning.py` để tinh chỉnh sâu hơn).
            """
        )


# ==================================================================================================
# TAB 4 - 🗺️ BẢN ĐỒ TRÁNH NGẬP (Smart Routing bằng TomTom Routing API)
# ==================================================================================================
def polygon_to_bounding_rectangle(polygon_points: list[tuple[float, float]]) -> dict:
    """
    Quy đổi 1 polygon (danh sách điểm lat/lon khoanh vùng ngập) thành 1 hình chữ nhật bao ngoài
    (bounding box), đúng định dạng `avoidAreas.rectangles` mà TomTom Routing API hỗ trợ.

    QUAN TRỌNG - LƯU Ý KỸ THUẬT CẦN GIẢI TRÌNH VỚI GIẢNG VIÊN:
    TomTom Routing API (endpoint `calculateRoute`) CHỈ hỗ trợ `avoidAreas.rectangles` (hình chữ nhật
    theo tọa độ southWestCorner/northEastCorner), KHÔNG hỗ trợ `avoidAreas.polygons` (đa giác tự do)
    như một số API định tuyến khác (vd. OpenRouteService có `avoid_polygons`). Do đó, để payload gửi
    lên TomTom thực sự hợp lệ và có tác dụng né vùng ngập thật, ta cần "xấp xỉ" polygon ngập bằng
    hình chữ nhật bao ngoài nhỏ nhất (min/max lat, min/max lon) trước khi đưa vào request.
    Phần polygon GỐC (chi tiết, không phải hình chữ nhật) vẫn được giữ nguyên để VẼ trên bản đồ
    Folium cho trực quan - chỉ có phần gửi lên API là bị đơn giản hóa thành rectangle.
    """
    latitudes = [point[0] for point in polygon_points]
    longitudes = [point[1] for point in polygon_points]
    return {
        "southWestCorner": {"latitude": min(latitudes), "longitude": min(longitudes)},
        "northEastCorner": {"latitude": max(latitudes), "longitude": max(longitudes)},
    }


def fetch_tomtom_route(
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    flooded_polygons: list | None = None,
    api_key: str = TOMTOM_API_KEY,
) -> dict:
    """
    LUÔN gọi TomTom Routing API để lấy tuyến đường THẬT bám theo mạng lưới đường (snap-to-road) từ
    `start_point` đến `end_point` - không còn nhánh vẽ đường chim bay minh họa như trước:
      - NẾU `flooded_polygons` RỖNG: gọi TomTom KHÔNG kèm `avoidAreas` (định tuyến bình thường,
        TomTom tự chọn tuyến nhanh nhất qua mạng lưới đường thật).
      - NẾU `flooded_polygons` CÓ dữ liệu: gọi TomTom KÈM `avoidAreas` để tính đường vòng né vùng
        ngập AI vừa cảnh báo.
    Trong cả 2 trường hợp, `route_points` trả về LUÔN là tọa độ do TomTom tính, không phải điểm nối
    thẳng đi/đến.

    TẠI SAO CHỌN `travelMode=motorcycle`?
    Đối tượng phục vụ chính của hệ thống cảnh báo là NGƯỜI DÂN VÀ LỰC LƯỢNG CỨU HỘ tại Huế - phương
    tiện di chuyển phổ biến nhất trong đô thị Việt Nam khi có ngập cục bộ là XE MÁY, không phải ô tô.
    Xe máy có khả năng len lỏi qua các tuyến đường nhỏ/hẻm mà ô tô (`car`) không đi được, đồng thời
    vẫn cần né vùng ngập sâu (khác với đi bộ `pedestrian` - không tối ưu về thời gian di chuyển trong
    tình huống khẩn cấp). Vì vậy `motorcycle` là travel mode sát với bài toán thực tế nhất.

    Trả về dict:
        {"success": bool, "route_points": [(lat, lon), ...], "distance_km": float,
         "travel_time_min": float, "used_avoid_areas": bool, "error": str | None}
    """
    has_flood_zones = bool(flooded_polygons)
    locations = f"{start_point[0]},{start_point[1]}:{end_point[0]},{end_point[1]}"

    request_url = f"{TOMTOM_ROUTING_BASE_URL}/{locations}/json"
    query_params = {
        "key": api_key,
        "traffic": "true",
        "travelMode": "motorcycle",
        "routeType": "fastest",
    }
    try:
        if has_flood_zones:
            # TomTom BẮT BUỘC dùng POST kèm body 'avoidAreas' khi cần né vùng ngập - request body
            # RỖNG sẽ bị TomTom từ chối với lỗi 400 (đã kiểm chứng thực tế), nên nhánh KHÔNG có vùng
            # ngập bên dưới phải dùng GET thay vì POST với body {}.
            request_body = {
                "avoidAreas": {
                    "rectangles": [polygon_to_bounding_rectangle(polygon) for polygon in flooded_polygons]
                }
            }
            response = requests.post(request_url, params=query_params, json=request_body, timeout=15)
        else:
            # Không có vùng ngập cần né -> gọi GET tiêu chuẩn, TomTom tự tính tuyến nhanh nhất.
            response = requests.get(request_url, params=query_params, timeout=15)
        response.raise_for_status()
        payload = response.json()

        route = payload["routes"][0]
        summary = route["summary"]
        route_points = [
            (point["latitude"], point["longitude"])
            for leg in route["legs"]
            for point in leg["points"]
        ]

        return {
            "success": True,
            "route_points": route_points,
            "distance_km": round(summary["lengthInMeters"] / 1000, 2),
            "travel_time_min": round(summary["travelTimeInSeconds"] / 60, 1),
            "used_avoid_areas": has_flood_zones,
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "route_points": [],
            "distance_km": None,
            "travel_time_min": None,
            "used_avoid_areas": has_flood_zones,
            "error": str(exc),
        }


def build_flood_zone_polygon(
    center_point: tuple[float, float], half_size_deg: float = FLOOD_ZONE_HALF_SIZE_DEG
) -> list[tuple[float, float]]:
    """Sinh 1 hình vuông nhỏ (bounding box) bao quanh 1 tọa độ trung tâm, đại diện cho vùng ngập
    ước tính tại địa phương đó. Đây là polygon THẬT (không phải dummy) vì tâm của nó chính là tọa độ
    giám sát thực tế trong REAL_MONITORED_LOCATIONS, chỉ có KÍCH THƯỚC là xấp xỉ."""
    center_lat, center_lon = center_point
    return [
        (center_lat + half_size_deg, center_lon - half_size_deg),
        (center_lat + half_size_deg, center_lon + half_size_deg),
        (center_lat - half_size_deg, center_lon + half_size_deg),
        (center_lat - half_size_deg, center_lon - half_size_deg),
    ]


def predict_flood_class(deployed_model: dict, location_df: pd.DataFrame, feature_columns: list[str]) -> int:
    """
    Suy luận nhãn nguy cơ ngập (0 = An toàn, 1 = Ngập nhẹ, 2 = Ngập nặng) mới nhất bằng model THẬT
    đã triển khai, tự động chọn đúng luồng xử lý theo `model_type` (xem `load_deployment_model()`
    để biết chi tiết 3 loại model: sklearn_tabular / keras_sequence / hybrid_lstm_xgboost).
    """
    model_type = deployed_model["model_type"]
    scaler = deployed_model["scaler"]

    if model_type == "sklearn_tabular":
        latest_features = location_df[feature_columns].iloc[[-1]]
        scaled_features = scaler.transform(latest_features)
        return int(deployed_model["model"].predict(scaled_features)[0])

    window_size = deployed_model.get("window_size") or 24
    if len(location_df) < window_size:
        raise ValueError("Không đủ dữ liệu quan trắc để tạo chuỗi thời gian cho model tuần tự.")
    window_features = location_df[feature_columns].iloc[-window_size:]
    scaled_window = scaler.transform(window_features)
    model_input = scaled_window.reshape(1, window_size, len(feature_columns))

    if model_type == "keras_sequence":
        class_probabilities = deployed_model["model"].predict(model_input, verbose=0)
        return int(class_probabilities.argmax(axis=-1)[0])

    if model_type == "hybrid_lstm_xgboost":
        embedding = deployed_model["feature_extractor"].predict(model_input, verbose=0)
        return int(deployed_model["classifier"].predict(embedding)[0])

    raise ValueError(f"model_type không được hỗ trợ cho suy luận thời gian thực: {model_type}")


@st.cache_data(ttl=300, show_spinner=False)
def get_latest_flood_predictions() -> pd.DataFrame:
    """
    ĐỌC ĐỘNG trạng thái nguy cơ ngập MỚI NHẤT cho 5 địa phương giám sát thực tế - đây chính là
    `df_predictions` mà bản đồ định tuyến bên dưới dựa vào, với 2 cột ['Địa phương', 'Nguy cơ'].

    Đây là "cầu nối" giữa pipeline AI (Tab 2 huấn luyện, Tab 3 đánh giá) và bản đồ định tuyến
    (Tab 4): thay vì hardcode nguy cơ ngập giả lập như bản demo cũ, hàm này tự động suy ra nguy cơ
    ngập THẬT cho từng địa phương, để routing engine phía dưới TỰ ĐỘNG né đúng khu vực đang được AI
    cảnh báo mà không cần chỉnh sửa code bằng tay mỗi khi tình hình thời tiết thay đổi.

    Thứ tự ưu tiên khi xác định 'Nguy cơ':
      1. Nếu đã có model được triển khai (`deployment_config.json` tồn tại sau khi huấn luyện ở
         Tab 2) -> nạp model thật và suy luận trên quan trắc GẦN NHẤT của từng địa phương.
      2. Nếu CHƯA huấn luyện model nào -> dự phòng bằng chính nhãn `Nguy_cơ_ngập` THỰC TẾ ở dòng dữ
         liệu quan trắc gần nhất trong data/historical/*.csv (vẫn là dữ liệu thật, không phải số
         ngẫu nhiên) để bản đồ luôn phản ánh tình trạng có cơ sở dữ liệu, không bao giờ "đứng im".
    """
    deployed_model = None
    feature_columns = FEATURE_COLS_FOR_INFERENCE
    try:
        evaluation_metrics, deployment_config, runtime_info = load_evaluation_artifacts()
        deployed_model = load_deployment_model(deployment_config, runtime_info["latest_dir"])
        feature_columns = deployment_config.get("feature_columns", feature_columns)
    except Exception:
        deployed_model = None  # Chưa có model triển khai -> dùng nhánh dự phòng bên dưới.

    records = []
    for location_name, csv_filename in LOCATION_HISTORICAL_FILE.items():
        risk_status = "An toàn"
        try:
            location_df = pd.read_csv(HISTORICAL_DIR / csv_filename).sort_values("Thời_gian")
            if deployed_model is not None:
                predicted_class = predict_flood_class(deployed_model, location_df, feature_columns)
            else:
                predicted_class = int(location_df["Nguy_cơ_ngập"].iloc[-1])
            risk_status = "An toàn" if predicted_class == 0 else "Ngập"
        except Exception:
            risk_status = "An toàn"  # Thiếu dữ liệu/model lỗi -> mặc định an toàn, không chặn UI.

        records.append({"Địa phương": location_name, "Nguy cơ": risk_status})

    return pd.DataFrame(records, columns=["Địa phương", "Nguy cơ"])


# ==================================================================================================
# MODULE DỰ BÁO 4 NGÀY (Ngày T, T+1, T+2, T+3) - CORE FORECASTING MODULE
# ==================================================================================================
def _fetch_or_estimate_tide_heights(lat: float, lon: float, forecast_dates: pd.DatetimeIndex) -> list[float]:
    """
    Lấy chiều cao triều (m) cho từng ngày trong `forecast_dates`.

    Ưu tiên dữ liệu THẬT từ Open-Meteo Marine API (`wave_height_max`) tại chính tọa độ (lat, lon).
    Marine API CHỈ có dữ liệu tại các điểm lưới nằm trên/gần biển - với tọa độ NỘI ĐỊA (ví dụ TP Huế,
    Hương Trà, Quảng Điền), API trả về `null` cho toàn bộ ngày, khi đó hàm dùng công thức triều tổng
    hợp (bán nhật triều chu kỳ ~12.42 giờ + chu kỳ mặt trăng ~29.53 ngày) làm giá trị xấp xỉ - ĐÚNG
    phương pháp đã dùng để sinh cột `Chiều_cao_triều_m` khi xây dựng dữ liệu huấn luyện lịch sử (xem
    `calculate_synthetic_tide()` trong `fetch_data.py`), giúp đầu vào suy luận nhất quán về mặt phân
    phối với dữ liệu mà model đã học, thay vì dùng một hằng số mặc định tùy tiện.
    """
    try:
        marine_response = requests.get(
            "https://marine-api.open-meteo.com/v1/marine",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "wave_height_max",
                "forecast_days": len(forecast_dates),
                "timezone": "auto",
            },
            timeout=15,
        )
        marine_response.raise_for_status()
        wave_heights = marine_response.json()["daily"]["wave_height_max"][: len(forecast_dates)]
        if wave_heights and all(value is not None for value in wave_heights):
            return [float(value) for value in wave_heights]
    except (requests.exceptions.RequestException, KeyError, ValueError, TypeError):
        pass  # Tọa độ nội địa hoặc Marine API lỗi -> rơi xuống nhánh công thức dự phòng bên dưới.

    synthetic_heights = []
    for forecast_date in forecast_dates:
        elapsed_seconds = (forecast_date.to_pydatetime() - datetime(2000, 1, 1)).total_seconds()
        lunar_phase = 2 * math.pi * elapsed_seconds / (29.53 * 86400)
        semi_daily_phase = 2 * math.pi * elapsed_seconds / (12.42 * 3600)
        tide_value = 1.0 + 0.5 * math.sin(lunar_phase) + 0.8 * math.sin(semi_daily_phase)
        synthetic_heights.append(max(0.1, min(4.0, tide_value)))
    return synthetic_heights


def predict_4_days_forecast(lat: float, lon: float, model, scaler) -> pd.DataFrame | None:
    """
    Dự báo nguy cơ ngập cho 4 NGÀY LIÊN TIẾP - Ngày T (hôm nay), T+1, T+2, T+3 - tại 1 tọa độ
    (lat, lon) bất kỳ, dùng `model` + `scaler` ĐÃ HUẤN LUYỆN SẴN được truyền vào (không tự nạp model
    trong hàm này, để hàm dùng được với bất kỳ model nào - XGBoost, RandomForest, hay Deep Learning).

    Trả về DataFrame gồm 3 cột: ['Ngày', 'Dự báo Lượng mưa (mm)', 'Dự đoán Ngập'], hoặc `None` nếu
    gọi API/model lỗi (đã được xử lý gracefully bằng try-except, không raise exception ra ngoài).
    """
    FEATURE_COLS = ["Nhiệt_độ_C", "Độ_ẩm_%", "Lượng_mưa_mm", "Độ_ẩm_đất", "Chiều_cao_triều_m"]
    FORECAST_DAYS = 4

    # ==============================================================================================
    # BƯỚC A - FETCH DATA: gọi Open-Meteo FORECAST API (dự báo tương lai - khác với Archive API chỉ
    # có dữ liệu QUÁ KHỨ mà fetch_data.py dùng để xây tập huấn luyện) để lấy dữ liệu khí tượng THEO
    # NGÀY cho đúng 4 ngày kể từ hôm nay.
    #
    # GIẢI THÍCH XỬ LÝ CỬA SỔ THỜI GIAN (datetime window) - để đưa vào báo cáo luận văn:
    # Tham số `forecast_days=4` kết hợp `timezone="auto"` khiến Open-Meteo tự suy ra múi giờ ĐỊA
    # PHƯƠNG của tọa độ (lat, lon) - với Huế là Asia/Ho_Chi_Minh (UTC+7) - rồi trả về mảng `daily.time`
    # LUÔN bắt đầu từ NGÀY HIỆN TẠI theo múi giờ đó. Nhờ vậy, index 0 của mảng chính xác là Ngày T
    # (hôm nay), index 1/2/3 lần lượt là T+1/T+2/T+3, mà KHÔNG cần tự tính `datetime.now() + timedelta`
    # thủ công - cách làm thủ công dễ bị lệch 1 ngày nếu server chạy ở múi giờ UTC trong khi vị trí
    # cần dự báo lại ở múi giờ khác (UTC+7).
    # ==============================================================================================
    try:
        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": (
                    "precipitation_sum,rain_sum,temperature_2m_mean,"
                    "relative_humidity_2m_mean,soil_moisture_0_to_7cm_mean"
                ),
                "forecast_days": FORECAST_DAYS,
                "timezone": "auto",
            },
            timeout=15,
        )
        weather_response.raise_for_status()
        daily_weather = weather_response.json()["daily"]
        forecast_dates = pd.to_datetime(daily_weather["time"])
        if len(forecast_dates) < FORECAST_DAYS:
            raise ValueError(f"API chỉ trả về {len(forecast_dates)}/{FORECAST_DAYS} ngày dữ liệu.")
    except (requests.exceptions.RequestException, KeyError, ValueError, TypeError) as exc:
        print(f"[predict_4_days_forecast] Lỗi khi gọi Open-Meteo Forecast API: {exc}")
        return None

    # Chiều cao triều lấy riêng (Forecast API thường không có biến này) - xem docstring hàm phụ trợ.
    tide_heights = _fetch_or_estimate_tide_heights(lat, lon, forecast_dates)

    # ==============================================================================================
    # BƯỚC B - PREPROCESSING: gộp toàn bộ đặc trưng vào 1 DataFrame theo ĐÚNG THỨ TỰ CỘT mà `scaler`
    # đã ghi nhớ lúc huấn luyện (`scaler.feature_names_in_`), sau đó `transform()` để đưa dữ liệu thô
    # (đơn vị gốc: độ C, %, mm, m³/m³, m) về cùng thang đo chuẩn hóa (mean=0, std=1) mà model đã học.
    # Bỏ qua bước này sẽ khiến model suy luận sai nghiêm trọng dù code chạy không lỗi.
    # ==============================================================================================
    try:
        daily_features_df = pd.DataFrame(
            {
                "Nhiệt_độ_C": daily_weather["temperature_2m_mean"][:FORECAST_DAYS],
                "Độ_ẩm_%": daily_weather["relative_humidity_2m_mean"][:FORECAST_DAYS],
                "Lượng_mưa_mm": daily_weather["rain_sum"][:FORECAST_DAYS],
                "Độ_ẩm_đất": daily_weather["soil_moisture_0_to_7cm_mean"][:FORECAST_DAYS],
                "Chiều_cao_triều_m": tide_heights,
            }
        )
        feature_columns = list(getattr(scaler, "feature_names_in_", FEATURE_COLS))
        X_scaled = scaler.transform(daily_features_df[feature_columns])
    except Exception as exc:
        print(f"[predict_4_days_forecast] Lỗi khi tiền xử lý/chuẩn hóa dữ liệu: {exc}")
        return None

    # ==============================================================================================
    # BƯỚC C - PREDICTION: `model` có thể là XGBoost/RandomForest (API kiểu scikit-learn, `.predict()`
    # trả về THẲNG nhãn lớp 0/1/2 - mảng 1 chiều) hoặc Deep Learning/Keras (`.predict()` trả về ma
    # trận xác suất softmax theo từng lớp - mảng 2 chiều, cần `argmax` để lấy nhãn lớp có xác suất cao
    # nhất). Tự động nhận diện theo số chiều (`ndim`) của kết quả để xử lý đúng cho cả 2 trường hợp mà
    # không cần biết trước loại model.
    # ==============================================================================================
    try:
        raw_output = np.asarray(model.predict(X_scaled))
        predicted_classes = np.argmax(raw_output, axis=1) if raw_output.ndim == 2 else raw_output.astype(int)
    except Exception as exc:
        print(f"[predict_4_days_forecast] Lỗi khi suy luận bằng model: {exc}")
        return None

    # ==============================================================================================
    # BƯỚC D - OUTPUT: đóng gói kết quả thành DataFrame dễ đọc - gắn nhãn ngày thứ (T/T+1/T+2/T+3) và
    # ánh xạ nhãn số sang văn bản tiếng Việt dễ hiểu (0 -> 'An toàn', khác 0 -> 'Nguy cơ ngập').
    # ==============================================================================================
    day_labels = ["T (Hôm nay)", "T+1", "T+2", "T+3"]
    result_rows = [
        {
            "Ngày": f"{forecast_dates[offset].strftime('%d/%m/%Y')} ({day_labels[offset]})",
            "Dự báo Lượng mưa (mm)": round(float(daily_features_df["Lượng_mưa_mm"].iloc[offset]), 1),
            "Dự đoán Ngập": "An toàn" if int(predicted_classes[offset]) == 0 else "Nguy cơ ngập",
        }
        for offset in range(FORECAST_DAYS)
    ]
    return pd.DataFrame(result_rows, columns=["Ngày", "Dự báo Lượng mưa (mm)", "Dự đoán Ngập"])


def build_smart_routing_map(
    df_predictions: pd.DataFrame,
    real_flooded_polygons: list,
    start_point: tuple[float, float] | None = None,
    end_point: tuple[float, float] | None = None,
    route_points: list | None = None,
) -> folium.Map:
    """Dựng bản đồ Folium DUY NHẤT gộp cả 2 chức năng:
    1) Giám sát 5 địa phương thực tế - marker XANH LÁ ('An toàn') / ĐỎ ('Ngập') + vùng ngập tô đỏ.
    2) Định tuyến - marker điểm đi/đến + tuyến đường né ngập vẽ XANH DƯƠNG.
    """
    center_lat = sum(lat for lat, _ in REAL_MONITORED_LOCATIONS.values()) / len(REAL_MONITORED_LOCATIONS)
    center_lon = sum(lon for _, lon in REAL_MONITORED_LOCATIONS.values()) / len(REAL_MONITORED_LOCATIONS)
    routing_map = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")

    # ---- (1) Giám sát: vẽ 5 địa phương thực tế, màu marker theo đúng 'Nguy cơ' dự báo của AI ----
    for location_name, coordinates in REAL_MONITORED_LOCATIONS.items():
        risk_rows = df_predictions.loc[df_predictions["Địa phương"] == location_name, "Nguy cơ"]
        risk_status = risk_rows.iloc[0] if not risk_rows.empty else "An toàn"

        if risk_status == "Ngập":
            folium.Marker(
                location=coordinates,
                tooltip=f"🔴 {location_name}: Nguy cơ NGẬP (dự báo AI)",
                icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa"),
            ).add_to(routing_map)
        else:
            folium.Marker(
                location=coordinates,
                tooltip=f"🟢 {location_name}: An toàn",
                icon=folium.Icon(color="green", icon="check", prefix="fa"),
            ).add_to(routing_map)

    # ---- Vẽ vùng ngập THỰC TẾ (đã được suy ra từ df_predictions, không còn là dummy cố định) ----
    for polygon in real_flooded_polygons:
        folium.Polygon(
            locations=polygon,
            color="#B91C1C",
            weight=2,
            fill=True,
            fill_color="#EF4444",
            fill_opacity=0.35,
            tooltip="🔴 Vùng ngập ước tính - routing engine tự động né khu vực này",
        ).add_to(routing_map)

    # ---- (2) Định tuyến: điểm đi/đến (màu riêng, tránh trùng với màu xanh lá/đỏ của giám sát) ----
    if start_point is not None:
        folium.Marker(
            location=start_point,
            tooltip="🏁 Điểm xuất phát",
            icon=folium.Icon(color="blue", icon="play", prefix="fa"),
        ).add_to(routing_map)
    if end_point is not None:
        folium.Marker(
            location=end_point,
            tooltip="🏁 Điểm đến",
            icon=folium.Icon(color="cadetblue", icon="flag-checkered", prefix="fa"),
        ).add_to(routing_map)

    if route_points:
        folium.PolyLine(
            locations=route_points,
            color="#2563EB",
            weight=6,
            opacity=0.85,
            tooltip="🔵 Tuyến đường di chuyển (đã né vùng ngập)",
        ).add_to(routing_map)

    return routing_map


def render_smart_routing_tab() -> None:
    """
    Nội dung Tab 4 - Bản đồ Tránh ngập, bước THỨ TƯ (sản phẩm ứng dụng thực tế) của pipeline.
    Bố cục: bảng giám sát 5 địa phương thực tế -> cụm chọn điểm đi/đến + nút tìm đường (CĂN GIỮA
    bằng `st.columns([1,2,1])`) -> bản đồ Folium gộp cả giám sát lẫn định tuyến (CĂN GIỮA rộng hơn
    bằng `st.columns([1,6,1])`).
    """
    st.subheader("🗺️ Bản đồ Tránh ngập")
    st.caption(
        "Bước 4/4 của pipeline: giám sát 5 địa phương THỰC TẾ tại Thừa Thiên Huế bằng kết quả dự báo "
        "của model AI, sau đó dùng TomTom Routing API (travelMode=motorcycle) để tự động tính tuyến "
        "đường phù hợp cho xe máy, né các khu vực đang được AI cảnh báo ngập."
    )

    # ==============================================================================================
    # BƯỚC 1: ĐỌC ĐỘNG kết quả dự báo mới nhất (df_predictions) cho 5 địa phương giám sát thực tế.
    # Đây là điểm khác biệt CỐT LÕI so với bản demo cũ: bản đồ không còn hiển thị vùng ngập cố định
    # (dummy) nữa, mà LUÔN phản ánh đúng đầu ra hiện tại của model AI (get_latest_flood_predictions()).
    # ==============================================================================================
    df_predictions = get_latest_flood_predictions()

    # ---- BƯỚC 2: từ df_predictions, suy ra danh sách vùng ngập THỰC TẾ cần né (real_flooded_polygons)
    # - CHỈ những địa phương có 'Nguy cơ' = 'Ngập' mới được thêm vào đây. Danh sách này sẽ được
    # truyền thẳng vào tham số avoidAreas của TomTom Routing API bên dưới, nghĩa là routing engine
    # TỰ ĐỘNG chặn đúng khu vực AI đang cảnh báo, không cần con người khoanh vùng thủ công.
    real_flooded_polygons: list[list[tuple[float, float]]] = []
    for location_name, coordinates in REAL_MONITORED_LOCATIONS.items():
        risk_rows = df_predictions.loc[df_predictions["Địa phương"] == location_name, "Nguy cơ"]
        if not risk_rows.empty and risk_rows.iloc[0] == "Ngập":
            real_flooded_polygons.append(build_flood_zone_polygon(coordinates))

    st.markdown("##### 📡 Trạng thái giám sát 5 địa phương (từ dự báo AI mới nhất)")
    st.dataframe(df_predictions, use_container_width=True, hide_index=True)

    # ---- Cụm chọn điểm đi/điểm đến + nút tìm đường, đặt CĂN GIỮA bằng bố cục 3 cột ----
    control_left, control_center, control_right = st.columns([1, 2, 1])
    with control_center:
        location_names = list(REAL_MONITORED_LOCATIONS.keys())
        start_location_name = st.selectbox(
            "📍 Điểm xuất phát", location_names, index=0, key="routing_start_select"
        )
        end_location_name = st.selectbox(
            "🏁 Điểm đến",
            location_names,
            index=3 if len(location_names) > 3 else 0,
            key="routing_end_select",
        )

        find_route_clicked = st.button(
            "🧭 Tìm tuyến đường",
            key="find_smart_route_button",
            use_container_width=True,
        )

    start_point = REAL_MONITORED_LOCATIONS[start_location_name]
    end_point = REAL_MONITORED_LOCATIONS[end_location_name]

    # Chỉ gọi API khi người dùng bấm nút (tránh gọi lại TomTom mỗi lần Streamlit rerun do tương tác
    # UI khác - đây cũng là một hình thức tiết kiệm lượt gọi API bổ sung), kết quả được lưu vào
    # session_state để vẫn hiển thị lại khi chuyển qua chuyển lại giữa các tab.
    if find_route_clicked:
        if start_location_name == end_location_name:
            st.session_state["smart_route_result"] = None
            st.session_state["smart_route_same_point_error"] = True
        else:
            st.session_state["smart_route_same_point_error"] = False
            with st.spinner("Đang tính toán tuyến đường né vùng ngập..."):
                # ---- BƯỚC 3: gọi TomTom, truyền real_flooded_polygons vào avoidAreas ----
                st.session_state["smart_route_result"] = fetch_tomtom_route(
                    start_point, end_point, flooded_polygons=real_flooded_polygons
                )
            st.session_state["smart_route_points"] = (start_location_name, end_location_name)

    route_result = st.session_state.get("smart_route_result")

    with control_center:
        if st.session_state.get("smart_route_same_point_error"):
            st.warning("⚠️ Điểm xuất phát và điểm đến đang trùng nhau - vui lòng chọn 2 địa điểm khác nhau.")
        elif route_result is None:
            st.info("Bấm **Tìm tuyến đường** để bắt đầu tính toán.")
        elif not route_result["success"]:
            st.error(f"Không thể lấy tuyến đường từ TomTom: {route_result['error']}")
        else:
            # Luôn gọi TomTom nên luôn có số liệu THẬT (quãng đường/thời gian bám theo mạng lưới
            # đường thật) - khác nhau duy nhất giữa 2 trường hợp là có kèm avoidAreas hay không.
            metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
            metric_col_1.metric("Quãng đường", f"{route_result['distance_km']} km")
            metric_col_2.metric("Thời gian di chuyển", f"{route_result['travel_time_min']} phút")
            metric_col_3.metric("Vùng ngập đã né", f"{len(real_flooded_polygons)}")
            if route_result["used_avoid_areas"]:
                st.success("✅ Đã tính tuyến đường né vùng ngập thành công bằng TomTom Routing API.")
            else:
                st.success(
                    "✅ Không địa phương nào đang có nguy cơ ngập - TomTom tính tuyến đường bình "
                    "thường (không kèm avoidAreas)."
                )

    # ---- Bản đồ đặt CĂN GIỮA, rộng và cân đối trong tab - gộp cả giám sát 5 địa phương lẫn tuyến
    # đường vừa tính (nếu có) trên cùng 1 bản đồ Folium duy nhất ----
    map_left, map_center, map_right = st.columns([1, 6, 1])
    with map_center:
        route_points_to_draw = route_result["route_points"] if route_result and route_result["success"] else None
        smart_map = build_smart_routing_map(
            df_predictions=df_predictions,
            real_flooded_polygons=real_flooded_polygons,
            start_point=start_point,
            end_point=end_point,
            route_points=route_points_to_draw,
        )
        # `returned_objects=[]` tắt việc st_folium gửi trạng thái pan/zoom/click ngược về Streamlit -
        # đây là nguyên nhân khiến app rerun toàn bộ mỗi lần người dùng chỉ kéo/phóng bản đồ. Vì tab
        # này không cần đọc lại tương tác trên bản đồ (chỉ hiển thị), tắt hẳn để tránh rerun thừa.
        st_folium(smart_map, width=1000, height=600, returned_objects=[])


# ==================================================================================================
# SIDEBAR & ĐIỂM VÀO CHÍNH
# ==================================================================================================
def render_sidebar() -> None:
    """Sidebar tối giản: giới thiệu nhanh 4 bước pipeline + nút làm mới cache toàn cục."""
    st.sidebar.markdown("## 🌊 Flood Prediction Pipeline")
    st.sidebar.caption("Đồ án Thạc sĩ - Dự báo & Cảnh báo Ngập lụt Thừa Thiên Huế")
    st.sidebar.markdown(
        "1. 📊 Khám phá Dữ liệu (EDA)\n"
        "2. ⚙️ Tiền xử lý & Huấn luyện\n"
        "3. 📈 Đánh giá Mô hình\n"
        "4. 🗺️ Bản đồ Tránh ngập\n"
    )
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Làm mới toàn bộ cache", key="clear_all_cache_button", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.sidebar.success("Đã xóa cache - dữ liệu sẽ được nạp lại ở lần chạy tiếp theo.")


def main():
    """Điểm vào chính của ứng dụng - 4 tab bám sát đúng vòng đời Data Science (Data Science Lifecycle)."""
    apply_global_ui_theme()
    render_sidebar()

    st.title("🌊 Hệ thống Dự báo Ngập lụt Thừa Thiên Huế")
    st.caption(
        "Pipeline: Khám phá dữ liệu → Tiền xử lý & Huấn luyện → Đánh giá mô hình → Bản đồ chỉ đường tránh ngập."
    )
    st.markdown("---")

    # ------------------------------------------------------------------------------------------
    # BỐ CỤC TOÀN TRANG BẰNG st.tabs: 4 tab bám sát đúng vòng đời Data Science, giúp người xem
    # (giảng viên/hội đồng) theo dõi mạch pipeline từ dữ liệu thô đến sản phẩm ứng dụng thực tế,
    # thay vì một trang dashboard vận hành dài dòng, khó tách bạch từng giai đoạn.
    # ------------------------------------------------------------------------------------------
    tab_eda, tab_train, tab_eval, tab_map = st.tabs(
        [
            "📊 Khám phá Dữ liệu (EDA)",
            "⚙️ Tiền xử lý & Huấn luyện",
            "📈 Đánh giá Mô hình",
            "🗺️ Bản đồ Tránh ngập",
        ]
    )

    with tab_eda:
        render_eda_tab()

    with tab_train:
        render_preprocessing_training_tab()

    with tab_eval:
        render_evaluation_tab()

    with tab_map:
        render_smart_routing_tab()


if __name__ == "__main__":
    main()
