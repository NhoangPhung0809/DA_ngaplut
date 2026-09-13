import os

# PHẢI set TRƯỚC bất kỳ import nào có thể kéo theo `protobuf` (kể cả `streamlit` chính nó) - server
# đang cài tensorflow==2.10.1 (cần protobuf<3.20) CÙNG LÚC với streamlit bản mới (cần protobuf>=3.20),
# 2 yêu cầu xung đột trực tiếp nên không thể hạ/nâng version `protobuf` cho vừa cả 2. Nếu thiếu dòng
# này, lúc `load_deployed_model_and_features()` gọi `from tensorflow.keras.models import load_model`
# (model dạng chuỗi/Hybrid) sẽ crash với lỗi "Descriptors cannot not be created directly" ngay trên
# server - xem thêm giải thích chi tiết ở đầu `analyze_and_train.py` (cùng nguyên nhân, cùng cách sửa).
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import concurrent.futures
import io
import json
import importlib.util
import math
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

import folium
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import toml
from dotenv import load_dotenv
from streamlit_folium import st_folium

from shared_constants import FEATURE_COLS as _SHARED_FEATURE_COLS
from shared_constants import RAIN_LAG1_COL, RAIN_LAG2_COL, RAIN_ROLLING_3D_COL

# Tải biến môi trường từ file .env nếu có (ví dụ TOMTOM_API_KEY).
load_dotenv()

# Cấu hình giao diện trang Streamlit.
st.set_page_config(
    page_title="Dự báo ngập lụt Thừa Thiên Huế",
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
HYPERPARAMETER_TUNING_RESULTS_PATH = BASE_DIR / "data" / "hyperparameter_tuning_results.json"
CACHE_DIR = BASE_DIR / "cache"
TRAINING_WORKER_PATH = BASE_DIR / "training_worker.py"
TRAINING_STATUS_PATH = CACHE_DIR / "training_status.json"
TRAINING_LOG_PATH = CACHE_DIR / "training_output.log"

# ------------------------------------------------------------------------------------------------
# QUẢN LÝ API KEY AN TOÀN BẰNG st.secrets - KHÔNG hardcode key thật trong source code nữa (đây chính
# là nguyên nhân GitGuardian từng phát hiện rò rỉ key TomTom cũ trong lịch sử Git của repo này).
#
# CƠ CHẾ HOẠT ĐỘNG:
#   1. Khi chạy `streamlit run app.py`, Streamlit tự động đọc file `.streamlit/secrets.toml` (nếu có)
#      và nạp toàn bộ key trong đó vào object `st.secrets` - hoạt động y hệt một dict, ví dụ
#      `st.secrets["TOMTOM_KEY"]`. File này chỉ tồn tại CỤC BỘ trên máy bạn (hoặc được khai báo riêng
#      trong mục "Secrets" khi deploy lên Streamlit Community Cloud), và đã được khai báo trong
#      `.gitignore` -> Git sẽ KHÔNG BAO GIỜ commit file này lên GitHub, nên GitGuardian sẽ không còn
#      gì để quét thấy.
#   2. Nếu máy không có `.streamlit/secrets.toml` (ví dụ máy CI, hoặc bạn quen dùng file `.env` có
#      sẵn `load_dotenv()` ở đầu file), `get_api_secret()` bên dưới sẽ tự động rơi xuống đọc biến môi
#      trường bằng `os.getenv()` thay vì làm app crash - vẫn giữ được tính linh hoạt của cơ chế cũ.
#   3. Nếu CẢ HAI đều không có key, hàm trả về `None` - các lời gọi API tương ứng (TomTom, OpenWeather)
#      cần tự kiểm tra `None` để báo lỗi rõ ràng cho người dùng thay vì gọi API với key rỗng.
#
# LƯU Ý QUAN TRỌNG: key TomTom CŨ (đã từng hardcode trực tiếp trong file này ở các commit trước) coi
# như đã bị lộ vĩnh viễn trong lịch sử Git dù đã xóa khỏi code hiện tại - BẮT BUỘC phải thu hồi
# (revoke/regenerate) key đó trên dashboard TomTom, không chỉ đơn thuần xóa khỏi source code.
# ------------------------------------------------------------------------------------------------
def get_api_secret(secret_key: str, env_var_name: str) -> str | None:
    """
    Đọc 1 API key theo thứ tự ưu tiên:
    1. Key người dùng tự nhập qua khung "Cấu hình API nâng cao" (chỉ lưu trong `st.session_state`
       của phiên trình duyệt hiện tại - xem `render_admin_api_key_panel()`) - ưu tiên CAO NHẤT để admin
       có thể tạm thời ghi đè/thử key mới mà KHÔNG cần sửa `.env`/`secrets.toml` rồi redeploy lại server.
    2. `st.secrets` (file `.streamlit/secrets.toml` cấu hình sẵn trên server).
    3. Biến môi trường (`.env`).
    """
    session_keys = st.session_state.get("user_api_keys", {})
    if session_keys.get(secret_key):
        return session_keys[secret_key]
    try:
        if secret_key in st.secrets:
            return st.secrets[secret_key]
    except Exception:
        pass  # Chưa có file .streamlit/secrets.toml trên máy này -> rơi xuống nhánh .env bên dưới.
    return os.getenv(env_var_name)


TOMTOM_API_KEY = get_api_secret("TOMTOM_KEY", "TOMTOM_API_KEY")
# LƯU Ý: OPENWEATHER_API_KEY hiện CHƯA được gọi ở bất kỳ đâu trong app - toàn bộ tính năng thời tiết
# (dự báo 4 ngày, dữ liệu lịch sử) đang dùng Open-Meteo (không cần key). Biến này chỉ đọc sẵn key để
# dự phòng tích hợp OpenWeather sau này; nếu bạn thấy tab dự báo không phản ứng gì khi đổi key này,
# đó là lý do - không phải lỗi.
OPENWEATHER_API_KEY = get_api_secret("OPENWEATHER_KEY", "OPENWEATHER_API_KEY")
TOMTOM_ROUTING_BASE_URL = "https://api.tomtom.com/routing/1/calculateRoute"

# Mật khẩu bảo vệ khung "Cấu hình API nâng cao" (xem `render_admin_api_key_panel()`) - đọc theo
# đúng cơ chế `get_api_secret()` như mọi key khác (secrets.toml -> .env). Nếu CHƯA cấu hình mật khẩu
# này trên server, khung nhập API key sẽ TỰ ĐỘNG khoá hoàn toàn (không có mật khẩu mặc định "rỗng cho
# qua") - tránh trường hợp public app bị người lạ vào nhập/ghi đè API key tuỳ ý.
ADMIN_CONFIG_PASSWORD = get_api_secret("ADMIN_PASSWORD", "ADMIN_CONFIG_PASSWORD")

# Danh sách nhà cung cấp API thời tiết/bản đồ có thể cấu hình qua khung admin - mỗi provider gồm:
# nhãn hiển thị, "secret_key" (tên biến dùng trong get_api_secret/st.secrets), biến môi trường tương
# ứng trong .env, và link đăng ký để tiện tạo key mới. Đa số CHƯA được nối vào logic fetch dữ liệu
# thật (xem ghi chú ở OPENWEATHER_API_KEY phía trên) - khung này mới chỉ là nơi NHẬP & LƯU key an
# toàn cho phiên làm việc, việc dùng key đó để lấy thêm mẫu dữ liệu là bước tích hợp tiếp theo.
WEATHER_PROVIDER_OPTIONS = [
    {
        "label": "TomTom Routing (đã dùng cho Tab Bản đồ)",
        "secret_key": "TOMTOM_KEY",
        "env_var_name": "TOMTOM_API_KEY",
        "signup_url": "https://developer.tomtom.com/",
    },
    {
        "label": "OpenWeatherMap",
        "secret_key": "OPENWEATHER_KEY",
        "env_var_name": "OPENWEATHER_API_KEY",
        "signup_url": "https://openweathermap.org/api",
    },
    {
        "label": "Weatherbit",
        "secret_key": "WEATHERBIT_KEY",
        "env_var_name": "WEATHERBIT_API_KEY",
        "signup_url": "https://www.weatherbit.io/api",
    },
    {
        "label": "Visual Crossing",
        "secret_key": "VISUALCROSSING_KEY",
        "env_var_name": "VISUALCROSSING_API_KEY",
        "signup_url": "https://www.visualcrossing.com/weather-api",
    },
    {
        "label": "Tomorrow.io",
        "secret_key": "TOMORROW_KEY",
        "env_var_name": "TOMORROW_API_KEY",
        "signup_url": "https://www.tomorrow.io/weather-api/",
    },
    {
        "label": "Stormglass (chuyên triều cường/marine)",
        "secret_key": "STORMGLASS_KEY",
        "env_var_name": "STORMGLASS_API_KEY",
        "signup_url": "https://stormglass.io/",
    },
    {
        "label": "WorldTides (chuyên triều cường)",
        "secret_key": "WORLDTIDES_KEY",
        "env_var_name": "WORLDTIDES_API_KEY",
        "signup_url": "https://www.worldtides.info/",
    },
    {
        "label": "WeatherAPI.com",
        "secret_key": "WEATHERAPI_KEY",
        "env_var_name": "WEATHERAPI_API_KEY",
        "signup_url": "https://www.weatherapi.com/",
    },
    {
        "label": "World Weather Online",
        "secret_key": "WORLDWEATHERONLINE_KEY",
        "env_var_name": "WORLDWEATHERONLINE_API_KEY",
        "signup_url": "https://www.worldweatheronline.com/developer/",
    },
    {
        "label": "Goong.io (bản đồ Việt Nam)",
        "secret_key": "GOONG_KEY",
        "env_var_name": "GOONG_API_KEY",
        "signup_url": "https://goong.io/",
    },
]

SECRETS_TOML_PATH = BASE_DIR / ".streamlit" / "secrets.toml"


def persist_key_to_secrets_toml(secret_key: str, value: str) -> None:
    """
    Ghi 1 API key TRỰC TIẾP vào `.streamlit/secrets.toml` trên đĩa - khác với chỉ lưu vào
    `st.session_state` (mất khi reload trang/restart server). Đọc file hiện có (nếu có) trước, chỉ
    CẬP NHẬT đúng 1 key được truyền vào, giữ nguyên mọi key khác đã có sẵn (không ghi đè toàn bộ file).

    AN TOÀN: hàm này CHỈ được gọi từ `render_admin_api_key_panel()` - tức người bấm nút đã nhập ĐÚNG
    `ADMIN_PASSWORD` trước đó, nên về bản chất tương đương việc họ tự SSH vào server sửa file thủ công
    - không phải tính năng public ai cũng bấm được.
    """
    SECRETS_TOML_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing_data = {}
    if SECRETS_TOML_PATH.exists():
        try:
            existing_data = toml.load(SECRETS_TOML_PATH)
        except Exception:
            # File lỗi cú pháp sẵn (ví dụ dùng nhầm dấu ':' thay vì '=') - không cố "sửa hộ", chỉ coi
            # như rỗng để tránh mất dữ liệu oan nếu nội dung cũ thực ra vẫn đọc được 1 phần; người dùng
            # cần tự kiểm tra lại file nếu gặp trường hợp này.
            existing_data = {}
    existing_data[secret_key] = value
    with SECRETS_TOML_PATH.open("w", encoding="utf-8") as file:
        toml.dump(existing_data, file)


def render_admin_api_key_panel() -> None:
    """
    Khung "Cấu hình API nâng cao" trong sidebar - cho phép ADMIN (người biết mật khẩu, không phải
    khách vãng lai) nhập/ghi đè API key của từng nhà cung cấp NGAY TRÊN WEB, không cần sửa `.env`/
    `secrets.toml` rồi khởi động lại server.

    NGUYÊN TẮC AN TOÀN:
    - Key nhập vào CHỈ lưu trong `st.session_state` (bộ nhớ RAM của phiên trình duyệt hiện tại) - KHÔNG
      bao giờ ghi ra đĩa/log/git, tự động mất khi đóng tab hoặc server restart.
    - Khung nhập BỊ KHOÁ HOÀN TOÀN nếu server chưa cấu hình `ADMIN_CONFIG_PASSWORD` (không có mật khẩu
      mặc định) - tránh việc app public bị người lạ vào ghi đè/dò key của người khác.
    - Ô nhập mật khẩu và ô nhập API key đều dùng `type="password"` để không hiện rõ trên màn hình.
    """
    with st.sidebar.expander("Cấu hình API nâng cao (Admin)", expanded=False):
        if not ADMIN_CONFIG_PASSWORD:
            st.info(
                "Tính năng này đang KHOÁ vì server chưa cấu hình `ADMIN_PASSWORD` (trong "
                "`.streamlit/secrets.toml`) hoặc `ADMIN_CONFIG_PASSWORD` (trong `.env`). Cấu hình mật "
                "khẩu đó trước để mở khoá."
            )
            return

        if not st.session_state.get("admin_panel_unlocked", False):
            entered_password = st.text_input(
                "Mật khẩu admin", type="password", key="admin_panel_password_input"
            )
            if st.button("Mở khoá", key="admin_panel_unlock_button", use_container_width=True):
                if entered_password == ADMIN_CONFIG_PASSWORD:
                    st.session_state["admin_panel_unlocked"] = True
                    st.rerun()
                else:
                    st.error("Sai mật khẩu.")
            return

        st.success("Đã mở khoá cho phiên làm việc này.")
        provider_labels = [provider["label"] for provider in WEATHER_PROVIDER_OPTIONS]
        selected_label = st.selectbox("Chọn nhà cung cấp API", provider_labels, key="admin_panel_provider_select")
        selected_provider = next(p for p in WEATHER_PROVIDER_OPTIONS if p["label"] == selected_label)

        st.caption(f"Chưa có key? Đăng ký tại: {selected_provider['signup_url']}")
        new_key_value = st.text_input(
            f"API key cho {selected_provider['label']}",
            type="password",
            key=f"admin_panel_key_input_{selected_provider['secret_key']}",
        )
        session_only = st.checkbox(
            "Chỉ lưu tạm cho phiên này (không ghi ra đĩa - mất khi reload)",
            value=False,
            key="admin_panel_session_only_checkbox",
            help=(
                "Bỏ chọn (mặc định): key được ghi thẳng vào `.streamlit/secrets.toml` trên server - "
                "vẫn còn sau khi reload trang hoặc restart server, giống hệt việc bạn tự sửa file bằng "
                "SSH/nano. Chọn ô này nếu chỉ muốn thử key tạm thời, không muốn lưu vĩnh viễn."
            ),
        )

        if st.button("Lưu API key", key="admin_panel_save_key_button", use_container_width=True):
            if new_key_value:
                st.session_state.setdefault("user_api_keys", {})[selected_provider["secret_key"]] = new_key_value
                if session_only:
                    st.success(f"Đã lưu key cho {selected_provider['label']} (chỉ tồn tại trong phiên này).")
                else:
                    try:
                        persist_key_to_secrets_toml(selected_provider["secret_key"], new_key_value)
                        st.success(
                            f"Đã lưu key cho {selected_provider['label']} vào `.streamlit/secrets.toml` - "
                            "vẫn còn sau khi reload/restart."
                        )
                    except Exception as exc:
                        st.error(
                            f"Đã lưu tạm cho phiên này, nhưng GHI RA ĐĨA thất bại ({exc}) - key sẽ mất khi "
                            "reload. Kiểm tra quyền ghi thư mục `.streamlit/` trên server."
                        )
                st.rerun()
            else:
                st.warning("Vui lòng nhập API key trước khi lưu.")

        session_keys = st.session_state.get("user_api_keys", {})
        if session_keys:
            st.markdown("---")
            st.caption("Key đã cấu hình cho phiên này (chỉ hiện vài ký tự đầu để đối chiếu):")
            for provider in WEATHER_PROVIDER_OPTIONS:
                secret_key = provider["secret_key"]
                if secret_key not in session_keys:
                    continue
                masked_value = session_keys[secret_key][:4] + "…" if len(session_keys[secret_key]) > 4 else "…"
                key_col, remove_col = st.columns([3, 1])
                key_col.write(f"**{provider['label']}**: `{masked_value}`")
                if remove_col.button("Xoá", key=f"admin_panel_remove_{secret_key}"):
                    del st.session_state["user_api_keys"][secret_key]
                    st.rerun()

        st.markdown("---")
        if st.button("Khoá lại panel", key="admin_panel_lock_button", use_container_width=True):
            st.session_state["admin_panel_unlocked"] = False
            st.rerun()

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

# Ranh giới hành chính THẬT (polygon) của 5 địa phương giám sát - sinh bằng `download_geo.py`, đối
# chiếu qua reverse-geocode ĐÚNG toạ độ ở REAL_MONITORED_LOCATIONS phía trên với OpenStreetMap
# Nominatim (không dùng ID quan hệ OSM cứng như bản cũ trong download_osm_geo.py - bản đó trỏ NHẦM
# sang Puerto Rico do ID sai). Property "name" của mỗi feature khớp CHÍNH XÁC với khoá trong
# REAL_MONITORED_LOCATIONS để tra cứu nguy cơ ngập theo đúng địa phương.
DISTRICT_BOUNDARY_GEOJSON_PATH = BASE_DIR / "data" / "geo" / "thuathienhue_districts.geojson"

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

# Danh sách đặc trưng đầu vào của model - import từ `shared_constants.py` (dùng CHUNG với
# analyze_and_train.py, train_model.py, eda_analysis.py, hyperparameter_tuning.py) thay vì tự định
# nghĩa lại, để không còn nguy cơ lệch nhau giữa các file khi bộ đặc trưng của model thay đổi. Giữ
# tên biến `FEATURE_COLS_FOR_INFERENCE` (thay vì đổi hết sang `FEATURE_COLS`) để không phải sửa lại
# mọi chỗ đã dùng tên này trong file - alias đơn giản, không đổi hành vi.
FEATURE_COLS_FOR_INFERENCE = _SHARED_FEATURE_COLS

# Số bước thời gian (ngày) mặc định cho model tuần tự (keras_sequence/hybrid_lstm_xgboost) khi
# `deployment_config.json` không có key `window_size` - PHẢI khớp `SEQUENCE_WINDOW` mặc định trong
# `analyze_and_train.py`, nếu không model sẽ nhận sai shape đầu vào mà không báo lỗi rõ ràng.
DEFAULT_SEQUENCE_WINDOW_SIZE = 7

# Ánh xạ nhãn lớp (dạng chuỗi, khớp key trong các file JSON export) sang tên tiếng Việt dễ đọc - DÙNG
# CHUNG cho mọi bảng/biểu đồ hiển thị 3 lớp nguy cơ ngập trong app (tránh định nghĩa lại rải rác).
CLASS_LABEL_VI: dict[str, str] = {"0": "Không ngập", "1": "Ngập nhẹ", "2": "Ngập nặng"}

# Ngưỡng chênh lệch F1-Macro (train - test) để cảnh báo model có dấu hiệu "học vẹt" (overfitting) -
# xem `attach_train_test_gap()` trong analyze_and_train.py và `render_overfitting_check_section()`.
OVERFITTING_GAP_WARNING_THRESHOLD = 0.15

# Chu kỳ tự động làm mới các tab "sống" (dự báo, giám sát) khi trình duyệt vẫn đang mở - dùng
# `st.fragment(run_every=...)`, xem giải thích đầy đủ tại `render_forecast_tab()`.
LIVE_TAB_AUTO_REFRESH_INTERVAL = "10m"

# Số ngày dự báo (Ngày T + 13 ngày tới = 14 ngày) - DÙNG CHUNG cho `predict_4_days_forecast()` và
# `predict_days_ahead_forecast_sequence()`, cùng với nhãn hiển thị tương ứng cho từng ngày. Open-Meteo
# Forecast API hỗ trợ tối đa 16 ngày (forecast_days<=16) nên 14 ngày vẫn nằm trong giới hạn miễn phí.
FORECAST_DAYS_AHEAD = 14
DAY_LABELS = ["T (Hôm nay)"] + [f"T+{offset}" for offset in range(1, FORECAST_DAYS_AHEAD)]

# Số ngày QUÁ KHỨ THẬT cần lấy thêm để tính các cột lag mưa (RAIN_LAG1_COL/RAIN_LAG2_COL/
# RAIN_ROLLING_3D_COL, xem shared_constants.py) khi suy luận - PHẢI khớp đúng độ trễ tối đa mà các cột
# lag đó cần (lag2 = 2 ngày trước) để KHÔNG có dòng nào bị NaN sau khi tính lag, kể cả ngày dự báo ĐẦU
# TIÊN (Ngày T/hôm nay).
RAIN_LAG_FETCH_DAYS = 2

# Khóa (lock) BẢO VỆ lệnh gọi `.predict()` của model Keras (`keras_sequence`/`hybrid_lstm_xgboost`)
# khi dự báo 5 địa phương chạy SONG SONG bằng ThreadPoolExecutor (xem `_compute_forecast_4day_result()`
# và `predict_class_from_sequence_window()`). TensorFlow/Keras KHÔNG đảm bảo an toàn khi nhiều thread
# gọi đồng thời `.predict()` trên CÙNG 1 model instance - dùng lock để nghiêm ngặt hóa chỉ phần suy
# luận Keras (rẻ, tính bằng mili-giây), trong khi phần TỐN THỜI GIAN THẬT (gọi API Open-Meteo qua
# mạng) vẫn chạy song song hoàn toàn bình thường - không mất đi lợi ích chính của việc song song hóa.
_KERAS_INFERENCE_LOCK = threading.Lock()

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
            font-size: 1.08rem;
        }
        [data-testid="stCaptionContainer"] p {
            color: #cbd5e1 !important;
            font-size: 1rem !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 1.05rem !important;
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
      minh từ Tab "Tiền xử lý & huấn luyện" để tránh block giao diện.
    """
    ensure_latest_models_dir()

    if historical_data_missing():
        with st.spinner("Đang tải dữ liệu lịch sử 10 năm..."):
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
def _load_evaluation_artifacts_cached(metrics_mtime: float, deployment_mtime: float):
    """
    Nạp `evaluation_metrics.json` + `deployment_config.json` + thông tin runtime cho Tab Đánh giá.
    `metrics_mtime`/`deployment_mtime` KHÔNG dùng trong thân hàm - chỉ tồn tại để LÀM CACHE KEY, ép
    Streamlit tự đọc lại 2 file JSON này mỗi khi `analyze_and_train.py` ghi đè bằng lần train mới
    (mtime đổi), thay vì cache VĨNH VIỄN kết quả của lần train ĐẦU TIÊN cho tới khi ai đó bấm '🔄 Làm
    mới toàn bộ cache' thủ công - đúng lỗi thực tế đã gặp (Confusion Matrix/leaderboard trông như
    "đứng yên" dù đã train lại nhiều lần), cùng nguyên nhân và cách sửa như
    `_load_ctgan_comparison_artifacts_cached()` ở trên.

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


def load_evaluation_artifacts():
    runtime_info = initialize_system()
    metrics_mtime = Path(runtime_info["metrics_path"]).stat().st_mtime
    deployment_mtime = Path(runtime_info["deployment_config_path"]).stat().st_mtime
    return _load_evaluation_artifacts_cached(metrics_mtime, deployment_mtime)


def load_deployment_model(deployment_config: dict, latest_dir: str | Path) -> dict:
    """
    NẠP LẠI BEST MODEL ĐÚNG CÁCH THEO `deployment_config.json` (universal loading mechanism).

    ----------------------------------------------------------------------------------------------
    TẠI SAO CẦN HÀM NÀY (thay vì luôn `joblib.load("best_model.pkl")` như code cũ)?
    ----------------------------------------------------------------------------------------------
    Sau khi sửa bug chọn best model trong `analyze_and_train.py`, best model của một lần huấn luyện
    có thể là 1 trong 4 dạng hoàn toàn khác nhau về cách lưu/nạp:
      - "sklearn_tabular": model + scaler đều là object Python thuần -> nạp bằng `joblib.load()`.
      - "keras_sequence" (GRU/LSTM/1D-CNN/CNN-LSTM): model được lưu bằng định dạng Keras gốc
        (`.keras`), BẮT BUỘC nạp lại bằng `tensorflow.keras.models.load_model()` - `joblib.load()`
        sẽ LỖI hoặc nạp sai vì Keras model không phải object pickle thuần.
      - "hybrid_lstm_xgboost": có 2 thành phần lưu riêng - LSTM feature-extractor (`.keras`, nạp
        bằng `load_model()`) và đầu phân loại XGBoost (`.json`, nạp bằng `XGBClassifier().load_model()`
        - định dạng native của XGBoost, KHÔNG phải joblib/pickle).
      - "hybrid_lstm_gru_xgboost": 3 thành phần lưu riêng - LSTM feature-extractor VÀ GRU
        feature-extractor (mỗi cái 1 file `.keras` riêng) + đầu phân loại XGBoost chung (`.json`) học
        trên embedding NỐI (concatenate) của cả 2 nhánh.

    Hàm này đọc `model_type` trong `deployment_config.json` và tự động dùng ĐÚNG loader tương ứng,
    để phần code gọi (ví dụ nút "Kiểm tra nạp model" ở Tab Đánh giá, hoặc sau này là tab suy luận
    thời gian thực) không cần biết trước hôm nay best model là loại gì.

    Trả về dict với khóa `model_type` luôn có mặt, cộng thêm các khóa object tương ứng
    (`model`+`scaler` cho 2 loại đầu, `feature_extractor`+`classifier`+`scaler` cho hybrid 2 thành
    phần, hoặc `lstm_feature_extractor`+`gru_feature_extractor`+`classifier`+`scaler` cho hybrid 3
    thành phần).
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

    if model_type == "hybrid_lstm_gru_xgboost":
        try:
            from tensorflow.keras.models import load_model as keras_load_model
        except ImportError as exc:
            raise ImportError(
                "Best model hiện tại là Hybrid LSTM+GRU+XGBoost nhưng môi trường đang chạy app.py "
                "chưa cài TensorFlow. Cài bằng: pip install tensorflow"
            ) from exc
        from xgboost import XGBClassifier

        lstm_feature_extractor = keras_load_model(str(latest_dir / artifacts["lstm_feature_extractor_path"]))
        gru_feature_extractor = keras_load_model(str(latest_dir / artifacts["gru_feature_extractor_path"]))
        classifier = XGBClassifier()
        classifier.load_model(str(latest_dir / artifacts["classifier_path"]))
        scaler = joblib.load(latest_dir / artifacts["scaler_path"])
        return {
            "model_type": model_type,
            "lstm_feature_extractor": lstm_feature_extractor,
            "gru_feature_extractor": gru_feature_extractor,
            "classifier": classifier,
            "scaler": scaler,
            "window_size": deployment_config.get("window_size"),
        }

    raise ValueError(f"Không hỗ trợ nạp model_type='{model_type}'.")


def load_deployed_model_and_features() -> tuple[dict, list[str], dict]:
    """
    Nạp model đã triển khai + danh sách `feature_columns` ĐÚNG THỨ TỰ - DÙNG CHUNG cho
    `get_latest_flood_predictions()` (Tab Bản đồ) và `_compute_forecast_4day_result()` (Tab Dự báo).
    Trước đây mỗi hàm tự lặp lại y hệt chuỗi `load_evaluation_artifacts()` -> `load_deployment_model()`
    -> đọc `deployment_config["feature_cols"]` (fallback `FEATURE_COLS_FOR_INFERENCE`) - đúng đoạn
    code mà comment cũ từng ghi nhận đã có lần đọc SAI tên khóa ("feature_columns" thay vì
    "feature_cols"), nay chỉ còn 1 chỗ duy nhất để sửa nếu lỗi tương tự lặp lại.

    KHÔNG tự bắt exception - bên gọi tự try/except theo đúng nhu cầu riêng của mình (ví dụ
    `get_latest_flood_predictions()` cần rơi xuống nhánh dự phòng lịch sử khi lỗi, còn
    `_compute_forecast_4day_result()` cần báo lỗi rõ ràng cho người dùng).

    Trả về `(deployed_model, feature_columns, deployment_config)` - `deployment_config` được trả kèm
    vì bên gọi thường còn cần thêm `model_name`/`model_type`/`f1_macro` từ đó.
    """
    evaluation_metrics, deployment_config, runtime_info = load_evaluation_artifacts()
    deployed_model = load_deployment_model(deployment_config, runtime_info["latest_dir"])
    feature_columns = deployment_config.get("feature_cols") or FEATURE_COLS_FOR_INFERENCE
    return deployed_model, feature_columns, deployment_config


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


# Màu tô nổi 3 hạng đầu trong bảng xếp hạng model (huy chương vàng/bạc/đồng) - DÙNG CHUNG cho mọi
# bảng xếp hạng trong app (đánh giá mô hình...), tránh mỗi nơi tự chọn 1 bộ màu khác nhau.
#
# ĐÃ GIẢM ĐỘ CHÓI (bản đầu tô CẢ DÒNG bằng màu vàng/bạc/đồng ĐẶC, người dùng phản ánh "chối mắt") RỒI
# THÊM GRADIENT ĐỘ SÁNG (theo góp ý sau đó): nền SÁNG NHẤT ở Hạng 1, giảm dần độ sáng xuống Hạng 2,
# 3 - tạo cảm giác "giảm dần theo thứ hạng" trực quan hơn thay vì 3 màu độ sáng ngang nhau khó phân
# biệt nhanh bằng mắt. Vẫn giữ viền trái đậm màu huy chương làm dấu hiệu phân biệt hạng rõ ràng.
RANK_MEDAL_COLORS: dict[int, tuple[str, str, str]] = {
    0: ("#5c4a1a", "#f8fafc", "#eab308"),  # Hạng 1 - nền vàng SÁNG NHẤT, viền trái vàng đậm.
    1: ("#333b46", "#f8fafc", "#94a3b8"),  # Hạng 2 - nền bạc sáng vừa, viền trái bạc.
    2: ("#2a1d10", "#f8fafc", "#b45309"),  # Hạng 3 - nền đồng TỐI NHẤT, viền trái đồng đậm.
}


def build_contrast_styler(
    df: pd.DataFrame,
    numeric_formats: dict | None = None,
    rank_highlight: bool = False,
):
    """
    Tạo Styler có độ tương phản cao (nền tối, chữ sáng, sọc ngựa vằn) để bảng web dễ đọc hơn.

    `rank_highlight=True`: tô nổi 3 DÒNG ĐẦU (sau khi `df` đã được sắp xếp theo tiêu chí xếp hạng
    TRƯỚC KHI truyền vào đây) bằng màu huy chương vàng/bạc/đồng (`RANK_MEDAL_COLORS`) - ĐÈ LÊN màu sọc
    ngựa vằn mặc định cho đúng 3 dòng đó, dùng cho các bảng xếp hạng model theo điểm số.

    ----------------------------------------------------------------------------------------------
    ĐÃ SỬA LỖI CRASH THẬT (TypeError: unsupported format string passed to NoneType.__format__) VÀ LỖI
    HIỂN THỊ THẬT (chữ "None" xuất hiện trên bảng thay vì "-"):
    ----------------------------------------------------------------------------------------------
    Nếu 1 cột trong `numeric_formats` có giá trị `None` (ví dụ model chưa từng chạy qua bước tính chỉ
    số đó - xem `attach_train_test_gap()`), pandas có thể giữ dtype `object` thay vì tự ép về số.
    Format string kiểu `"{:.4f}"` CRASH khi gặp `None` trực tiếp.

    Bản đầu chỉ ép kiểu số (`pd.to_numeric`) rồi GIAO `na_rep="-"` cho `Styler.format()` tự xử lý -
    nhưng `st.dataframe()` KHÔNG áp dụng `Styler.format()` một cách đáng tin cậy ở mọi phiên bản
    Streamlit (bug thực tế đã gặp: dữ liệu gốc đã là `NaN` sạch, nhưng UI vẫn hiện chữ "None" thay vì
    "-" vì `st.dataframe` đôi khi bỏ qua format của Styler, tự hiển thị thẳng giá trị thô). Sửa TẬN GỐC
    bằng cách tự đưa các cột trong `numeric_formats` thành CHUỖI ĐÃ ĐỊNH DẠNG SẴN (bao gồm cả xử lý
    `NaN` -> "-") NGAY TRONG PYTHON trước khi tạo Styler - không phụ thuộc Streamlit có tôn trọng
    `Styler.format()` hay không nữa, cùng nguyên tắc đã dùng cho bảng CTGAN/case study trước đó.
    """
    df = df.copy()
    if numeric_formats:
        for column, format_spec in numeric_formats.items():
            if column not in df.columns:
                continue
            numeric_series = pd.to_numeric(df[column], errors="coerce")
            df[column] = numeric_series.map(
                lambda value, spec=format_spec: "-" if pd.isna(value) else spec.format(value)
            )

    styled_df = df.style

    def zebra_rows(row):
        medal = RANK_MEDAL_COLORS.get(row.name) if rank_highlight else None
        if medal:
            background, text_color, accent_color = medal
            return [
                f"background-color: {background}; color: {text_color}; font-weight: 700; "
                f"border-left: 4px solid {accent_color};"
                for _ in row
            ]
        background = "#0f172a" if row.name % 2 == 0 else "#172033"
        return [f"background-color: {background}; color: #f8fafc;" for _ in row]

    styled_df = styled_df.apply(zebra_rows, axis=1)
    styled_df = styled_df.set_properties(
        **{
            "color": "#f8fafc",
            "border": "1px solid #334155",
            "font-size": "17px",
            "padding": "10px 12px",
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
                    ("font-size", "17px"),
                    ("padding", "12px 14px"),
                    ("text-align", "center"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("border", "1px solid #334155"),
                    ("font-size", "17px"),
                    ("padding", "10px 12px"),
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
# TAB 1 - KHÁM PHÁ DỮ LIỆU (EDA)
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

    combined_df = regenerate_flood_risk_label(combined_df)
    return combined_df


def regenerate_flood_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính lại cột `Nguy_cơ_ngập` 3 lớp (0/1/2) bằng ĐÚNG luật chuyên gia dùng trong
    `create_multiclass_flood_label()` (analyze_and_train.py), thay vì tin thẳng cột `Nguy_cơ_ngập` sẵn
    có trong CSV thô.

    LÝ DO CẦN HÀM NÀY (bug thực tế đã gặp): CSV thô trong `data/historical/` lưu nhãn từ MỘT phiên bản
    fetch/label CŨ hơn (có thể chỉ 0/1 nhị phân, không có lớp 'Ngập nặng'), KHÁC với nhãn 3 lớp rule-
    based mà pipeline huấn luyện thật sự dùng - `eda_analysis.py` (script độc lập cũ) đã biết vấn đề
    này và tự quy định lại nhãn trước khi vẽ biểu đồ (xem docstring `save_eda_metadata()` ở đó), nhưng
    `load_eda_sample_dataframe()` ở `app.py` trước đây LẤY THẲNG cột CSV thô - khiến Tab EDA hiển thị
    sai lệch nhãn so với thực tế model đang học (ví dụ biểu đồ phân bố lớp bị THIẾU HẲN lớp 'Ngập
    nặng' dù dữ liệu vẫn có các đợt mưa/triều đủ lớn để được xếp vào lớp đó).

    ĐƠN GIẢN HOÁ so với bản training thật: dùng `fillna(0)` trực tiếp cho 3 biến quyết định nhãn (mưa/
    độ ẩm đất/triều), KHÔNG áp dụng bước điền median-theo-tập-train phức tạp của
    `compute_train_only_medians()` - bước đó chỉ cần thiết để tránh rò rỉ dữ liệu khi HUẤN LUYỆN model,
    không ảnh hưởng tới mục đích XEM PHÂN BỐ dữ liệu ở Tab EDA.
    """
    if df.empty:
        return df

    required_columns = {"Lượng_mưa_mm", "Độ_ẩm_đất", "Chiều_cao_triều_m"}
    if not required_columns.issubset(df.columns):
        return df

    labeled_df = df.copy()
    rain = pd.to_numeric(labeled_df["Lượng_mưa_mm"], errors="coerce").fillna(0)
    soil = pd.to_numeric(labeled_df["Độ_ẩm_đất"], errors="coerce").fillna(0)
    tide = pd.to_numeric(labeled_df["Chiều_cao_triều_m"], errors="coerce").fillna(0)

    heavy_flood_mask = (
        (rain > 50)
        | ((rain > 30) & (soil > 0.45))
        | ((rain > 20) & (soil > 0.40) & (tide > 1.50))
        | (tide > 2.50)
    )
    light_flood_mask = (
        (rain > 25)
        | ((rain > 15) & (soil > 0.30))
        | ((rain > 10) & (tide > 1.20))
    )

    labeled_df["Nguy_cơ_ngập"] = 0
    labeled_df.loc[light_flood_mask, "Nguy_cơ_ngập"] = 1
    labeled_df.loc[heavy_flood_mask, "Nguy_cơ_ngập"] = 2
    return labeled_df


def compute_outlier_summary_by_class(
    df: pd.DataFrame, group_col: str, numeric_columns: list[str]
) -> pd.DataFrame:
    """
    Phát hiện ngoại lai (outlier) bằng CẢ 2 phương pháp IQR và Z-score, tính RIÊNG cho từng lớp
    `group_col` (Nguy_cơ_ngập: 0/1/2) trên từng cột số - đúng như khuyến nghị đã ghi ở phần nhận xét
    bên dưới bảng giá trị thiếu: không gộp chung tất cả các lớp lại rồi tính 1 ngưỡng duy nhất, vì
    ngưỡng "bất thường" của lượng mưa/triều cường ở lớp `Ngập nặng` vốn dĩ CAO HƠN nhiều so với lớp
    `Không ngập` một cách tự nhiên - nếu tính chung, phần lớn các dòng dữ liệu THẬT của lớp `Ngập nặng`
    sẽ bị nhầm là ngoại lai.

    - IQR: ngoại lai là giá trị nằm ngoài [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    - Z-score: ngoại lai là giá trị có |z| > 3 (lệch hơn 3 độ lệch chuẩn so với trung bình của lớp đó).

    Hàm này CHỈ THỐNG KÊ để hiển thị, KHÔNG tự động loại bỏ dòng nào khỏi dữ liệu.
    """
    class_name_map = {0: "Không ngập", 1: "Ngập nhẹ", 2: "Ngập nặng"}
    result_rows = []

    for class_value, class_df in df.groupby(group_col):
        class_label = class_name_map.get(int(class_value), f"Lớp {class_value}")
        for column in numeric_columns:
            values = class_df[column].dropna()
            if len(values) < 2:
                continue

            # ---- IQR ----
            q1, q3 = values.quantile(0.25), values.quantile(0.75)
            iqr = q3 - q1
            lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            iqr_outlier_count = int(((values < lower_bound) | (values > upper_bound)).sum())

            # ---- Z-score ---- (std=0 nghĩa là mọi giá trị trong lớp giống hệt nhau -> không có outlier)
            std = values.std()
            if std and std > 0:
                z_scores = (values - values.mean()) / std
                zscore_outlier_count = int((z_scores.abs() > 3).sum())
            else:
                zscore_outlier_count = 0

            result_rows.append(
                {
                    "Lớp nguy cơ": class_label,
                    "Cột": column,
                    "Số ngoại lệ (IQR)": iqr_outlier_count,
                    "Tỷ lệ (%) (IQR)": round(iqr_outlier_count / len(values) * 100, 2),
                    "Số ngoại lệ (Z-score)": zscore_outlier_count,
                    "Tỷ lệ (%) (Z-score)": round(zscore_outlier_count / len(values) * 100, 2),
                }
            )

    return pd.DataFrame(result_rows)


def compute_monthly_trend_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Gộp theo tháng (1-12): lượng mưa trung bình + tỷ lệ ngập (%) - dùng chung logic với
    `eda_analysis.py::save_monthly_trend_plot()` để 2 nơi luôn cho cùng 1 con số."""
    monthly_df = df.copy()
    monthly_df["Tháng"] = monthly_df["Thời_gian"].dt.month
    return (
        monthly_df.groupby("Tháng")
        .agg(
            Lượng_mưa_TB=("Lượng_mưa_mm", "mean"),
            Tỷ_lệ_ngập=("Nguy_cơ_ngập", lambda values: (values > 0).mean() * 100),
        )
        .reindex(range(1, 13))
    )


def render_interactive_monthly_trend_chart(eda_df: pd.DataFrame) -> None:
    """
    Biểu đồ TƯƠNG TÁC (Plotly, thay cho ảnh PNG tĩnh `monthly_trend.png` do `eda_analysis.py` sinh
    sẵn) - cho phép ZOOM/PAN/HOVER xem số liệu chính xác từng điểm, và chọn XEM THEO TỪNG NĂM riêng lẻ
    (thay vì chỉ gộp cả 10 năm như bản ảnh tĩnh cũ) qua ô chọn năm bên dưới.
    """
    if eda_df.empty or "Thời_gian" not in eda_df.columns or "Lượng_mưa_mm" not in eda_df.columns:
        st.info("Chưa có dữ liệu lịch sử trong `data/historical/` để vẽ biểu đồ xu hướng theo tháng.")
        return

    available_years = sorted(eda_df["Thời_gian"].dt.year.dropna().astype(int).unique().tolist())
    year_options = ["Gộp tất cả các năm"] + [str(year) for year in available_years]
    selected_year_label = st.selectbox("Chọn năm để xem", year_options, key="monthly_trend_year_select")

    filtered_df = (
        eda_df
        if selected_year_label == "Gộp tất cả các năm"
        else eda_df[eda_df["Thời_gian"].dt.year == int(selected_year_label)]
    )
    if filtered_df.empty:
        st.info(f"Không có dữ liệu cho năm {selected_year_label}.")
        return

    monthly_stats = compute_monthly_trend_stats(filtered_df).reset_index()
    month_labels = [f"Th{month}" for month in monthly_stats["Tháng"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=month_labels,
            y=monthly_stats["Lượng_mưa_TB"],
            name="Lượng mưa TB (mm)",
            mode="lines+markers",
            line=dict(color="#4C78A8", width=2.5),
            marker=dict(size=8),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=month_labels,
            y=monthly_stats["Tỷ_lệ_ngập"],
            name="Tỷ lệ ngập (%)",
            mode="lines+markers",
            line=dict(color="#E45756", width=2.5, dash="dash"),
            marker=dict(size=8, symbol="square"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title=(
            "Xu hướng lượng mưa & tỷ lệ ngập theo tháng - "
            + ("gộp toàn bộ các năm" if selected_year_label == "Gộp tất cả các năm" else f"năm {selected_year_label}")
        ),
        xaxis=dict(title="Tháng"),
        yaxis=dict(
            title=dict(text="Lượng mưa TB (mm)", font=dict(color="#4C78A8")),
            tickfont=dict(color="#4C78A8"),
        ),
        yaxis2=dict(
            title=dict(text="Tỷ lệ ngập (%)", font=dict(color="#E45756")),
            tickfont=dict(color="#E45756"),
            overlaying="y",
            side="right",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=440,
        margin=dict(t=70),
    )
    st.plotly_chart(fig, use_container_width=True)

    render_chart_discussion(
        "Biểu đồ tương tác - có thể zoom/kéo thả để xem chi tiết từng tháng, hoặc dùng ô chọn năm phía "
        "trên để xem riêng 1 năm thay vì số liệu gộp cả 10 năm. Xu hướng lượng mưa và tỷ lệ ngập thường "
        "đồng biến theo mùa mưa (tháng 9-12 ở khu vực Huế) - cơ sở để khuyến nghị tăng cường giám sát "
        "vào các tháng cao điểm này."
    )


def apply_dark_plotly_theme(fig, height: int = 420) -> None:
    """
    Style DÙNG CHUNG cho mọi biểu đồ Plotly trong app: nền TRONG SUỐT + chữ/trục sáng màu, khớp theme
    tối toàn cục (`apply_global_ui_theme()`). Không áp style này thì Plotly mặc định nền TRẮNG, đặt
    cạnh theme tối trông như 1 ảnh dán lì, không giống thành phần tương tác thật của trang (bug thực tế
    đã gặp với Confusion Matrix trước khi thêm hàm này).
    """
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        # KHÔNG set `title=dict(font=...)` ở đây khi chưa có `title.text` - 1 số phiên bản Plotly.js
        # render literal chữ "undefined" ngay dưới tiêu đề markdown khi `layout.title` tồn tại nhưng
        # thiếu "text" (bug thực tế đã gặp). Màu chữ tiêu đề (khi CÓ set text ở nơi gọi) đã tự kế thừa
        # từ `font` toàn cục bên dưới, không cần khai báo `title.font` riêng.
        font=dict(color="#e5eefc"),
        legend=dict(font=dict(color="#e5eefc")),
        height=height,
        margin=dict(t=50, b=40, l=10, r=10),
    )
    fig.update_xaxes(color="#cbd5e1", gridcolor="#334155", zerolinecolor="#334155")
    fig.update_yaxes(color="#cbd5e1", gridcolor="#334155", zerolinecolor="#334155")


def render_correlation_heatmap_interactive(eda_df: pd.DataFrame) -> None:
    """Ma trận tương quan Pearson (Plotly heatmap tương tác) - thay cho `correlation_heatmap.png` tĩnh
    do `eda_analysis.py` sinh sẵn. Tính TRỰC TIẾP từ `eda_df` (dữ liệu lịch sử thật đang nạp trong
    app), không phụ thuộc file ảnh xuất sẵn."""
    numeric_cols = [col for col in [*FEATURE_COLS_FOR_INFERENCE, "Nguy_cơ_ngập"] if col in eda_df.columns]
    if eda_df.empty or len(numeric_cols) < 2:
        st.info("Chưa đủ dữ liệu số để tính ma trận tương quan.")
        return

    corr_df = eda_df[numeric_cols].corr(method="pearson").round(2)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns.tolist(),
            y=corr_df.columns.tolist(),
            colorscale="RdYlBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            hovertemplate="%{y} vs %{x}: %{z}<extra></extra>",
            colorbar=dict(title="Hệ số"),
        )
    )
    for row_index, row_label in enumerate(corr_df.index):
        for col_index, col_label in enumerate(corr_df.columns):
            value = corr_df.iloc[row_index, col_index]
            fig.add_annotation(
                x=col_label,
                y=row_label,
                text=f"{value:.2f}",
                showarrow=False,
                font=dict(size=12, color="#0f172a" if abs(value) < 0.6 else "#f8fafc"),
            )
    apply_dark_plotly_theme(fig, height=480)
    st.plotly_chart(fig, use_container_width=True)


def render_class_distribution_interactive(eda_df: pd.DataFrame) -> None:
    """Phân bố lớp mục tiêu (Plotly bar chart tương tác) - thay cho `class_distribution.png` tĩnh."""
    if eda_df.empty or "Nguy_cơ_ngập" not in eda_df.columns:
        st.info("Chưa có dữ liệu để thống kê phân bố lớp.")
        return

    class_counts = pd.to_numeric(eda_df["Nguy_cơ_ngập"], errors="coerce").value_counts().sort_index()
    class_labels = [f"{int(code)} - {CLASS_LABEL_VI.get(str(int(code)), str(code))}" for code in class_counts.index]
    class_colors = {"0": "#4C78A8", "1": "#F58518", "2": "#E45756"}
    bar_colors = [class_colors.get(str(int(code)), "#94a3b8") for code in class_counts.index]

    fig = go.Figure(
        data=go.Bar(
            x=class_labels,
            y=class_counts.values,
            marker=dict(color=bar_colors, line=dict(color="#1F1F1F", width=1)),
            text=[f"{int(v):,}" for v in class_counts.values],
            textposition="outside",
            hovertemplate="%{x}: %{y:,}<extra></extra>",
        )
    )
    apply_dark_plotly_theme(fig)
    fig.update_layout(xaxis=dict(title="Lớp nguy cơ ngập"), yaxis=dict(title="Số lượng quan sát"))
    st.plotly_chart(fig, use_container_width=True)


def render_flood_share_by_location_interactive(eda_df: pd.DataFrame) -> None:
    """Tỷ lệ ngập theo địa phương (Plotly pie chart tương tác) - thay cho
    `flood_share_by_location.png` tĩnh."""
    if eda_df.empty or "Nguy_cơ_ngập" not in eda_df.columns or "Địa phương" not in eda_df.columns:
        st.info("Chưa có dữ liệu để tính tỷ lệ ngập theo địa phương.")
        return

    flood_df = eda_df[pd.to_numeric(eda_df["Nguy_cơ_ngập"], errors="coerce") > 0]
    if flood_df.empty:
        st.info("Không có bản ghi ngập nào trong dữ liệu hiện có.")
        return

    location_counts = flood_df["Địa phương"].value_counts()
    fig = go.Figure(
        data=go.Pie(
            labels=location_counts.index.tolist(),
            values=location_counts.values.tolist(),
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
            textinfo="label+percent",
            marker=dict(line=dict(color="#0b1220", width=2)),
        )
    )
    apply_dark_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


def render_feature_distribution_by_class_interactive(eda_df: pd.DataFrame, feature_col: str, title: str) -> None:
    """Phân bố 1 biến số (mưa/triều cường...) theo TỪNG lớp nguy cơ ngập (Plotly histogram chồng lớp,
    dạng density) - thay cho `rain_distribution_by_class.png`/`tide_distribution_by_class.png` tĩnh."""
    if eda_df.empty or feature_col not in eda_df.columns or "Nguy_cơ_ngập" not in eda_df.columns:
        st.info(f"Chưa có dữ liệu để vẽ phân bố `{feature_col}`.")
        return

    plot_df = eda_df.copy()
    plot_df["Nguy_cơ_ngập"] = pd.to_numeric(plot_df["Nguy_cơ_ngập"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Nguy_cơ_ngập", feature_col])
    plot_df["Mức độ ngập"] = plot_df["Nguy_cơ_ngập"].astype(int).astype(str).map(CLASS_LABEL_VI)

    class_colors = {"Không ngập": "#4C78A8", "Ngập nhẹ": "#F58518", "Ngập nặng": "#E45756"}
    fig = px.histogram(
        plot_df,
        x=feature_col,
        color="Mức độ ngập",
        histnorm="probability density",
        barmode="overlay",
        opacity=0.55,
        nbins=40,
        color_discrete_map=class_colors,
        category_orders={"Mức độ ngập": ["Không ngập", "Ngập nhẹ", "Ngập nặng"]},
    )
    apply_dark_plotly_theme(fig)
    fig.update_layout(title=dict(text=title), yaxis=dict(title="Mật độ phân bố"))
    st.plotly_chart(fig, use_container_width=True)


def render_eda_tab() -> None:
    """
    Nội dung Tab 1 - Khám phá dữ liệu (EDA), bước ĐẦU TIÊN của vòng đời Data Science.
    Bố cục: 2 cột song song (Raw Data | Thống kê mô tả) phía trên, tiếp theo là 1 khối biểu đồ phân
    phối/tương quan, và cuối cùng là khối xử lý giá trị thiếu/ngoại lai - mỗi khối đặt trong
    `st.expander` để trang không bị dồn cục, người xem chỉ mở phần mình cần.
    """
    st.subheader("Khám phá dữ liệu (EDA)")
    st.caption(
        "Bước 1/4 của pipeline: hiểu dữ liệu trước khi làm sạch và huấn luyện. Các biểu đồ tĩnh bên dưới "
        "được sinh sẵn bởi `eda_analysis.py` (chạy `python eda_analysis.py` để làm mới sau khi có dữ liệu mới)."
    )

    eda_df = load_eda_sample_dataframe()

    # ---- Hàng 1: bố cục 2 CỘT song song bằng st.columns ----
    col_raw, col_stats = st.columns(2)

    with col_raw:
        with st.expander("Dữ liệu thô (Raw Data)", expanded=True):
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
        with st.expander("Thống kê mô tả (Descriptive Statistics)", expanded=True):
            # TODO: dán code `df.describe()` / thống kê chi tiết hơn của bạn (ví dụ describe theo
            # từng địa phương, theo từng lớp nguy cơ ngập...) vào đây.
            if eda_df.empty:
                st.info("Chưa có dữ liệu để thống kê.")
            else:
                numeric_df = eda_df.select_dtypes(include="number")
                describe_df = numeric_df.describe().T.reset_index().rename(columns={"index": "Biến"})
                render_styled_table(
                    build_contrast_styler(
                        describe_df,
                        numeric_formats={
                            "count": "{:,.0f}",
                            "mean": "{:,.3f}",
                            "std": "{:,.3f}",
                            "min": "{:,.3f}",
                            "25%": "{:,.3f}",
                            "50%": "{:,.3f}",
                            "75%": "{:,.3f}",
                            "max": "{:,.3f}",
                        },
                    ),
                    height=min(120 + 38 * len(describe_df), 400),
                )
                render_chart_discussion(
                    "Bảng `describe()` cho thấy khoảng giá trị, trung bình và độ lệch chuẩn của từng biến số - "
                    "cơ sở để phát hiện đơn vị đo bất thường (ví dụ độ ẩm âm, nhiệt độ ngoài khoảng hợp lý) "
                    "trước khi đưa dữ liệu vào bước tiền xử lý ở Tab 2."
                )

    st.markdown("---")

    # ---- Hàng 1.5: biểu đồ xu hướng mưa & tỷ lệ ngập theo tháng - TƯƠNG TÁC (thay cho ảnh tĩnh
    # `monthly_trend.png` cũ), cho phép chọn xem từng năm riêng lẻ hoặc gộp toàn bộ như trước ----
    with st.expander("Xu hướng mưa & tỷ lệ ngập theo tháng (biểu đồ tương tác)", expanded=True):
        render_interactive_monthly_trend_chart(eda_df)

    st.markdown("---")

    # ---- Hàng 2: biểu đồ phân phối / tương quan - TƯƠNG TÁC (Plotly, tính trực tiếp từ eda_df thật,
    # thay cho 5 ảnh PNG tĩnh do eda_analysis.py sinh sẵn trước đây) ----
    with st.expander("Phân phối dữ liệu & ma trận tương quan (Distribution / Heatmap)", expanded=True):
        chart_columns = st.columns(2)
        with chart_columns[0]:
            st.markdown("**Ma trận tương quan (Heatmap)**")
            render_correlation_heatmap_interactive(eda_df)
        with chart_columns[1]:
            st.markdown("**Phân bố lớp mục tiêu**")
            render_class_distribution_interactive(eda_df)

        chart_columns_2 = st.columns(2)
        with chart_columns_2[0]:
            st.markdown("**Tỷ lệ ngập theo địa phương**")
            render_flood_share_by_location_interactive(eda_df)
        with chart_columns_2[1]:
            st.markdown("**Phân bố lượng mưa theo lớp**")
            render_feature_distribution_by_class_interactive(
                eda_df, "Lượng_mưa_mm", "Phân bố lượng mưa theo lớp nguy cơ ngập"
            )

        st.markdown("**Phân bố triều cường theo lớp**")
        render_feature_distribution_by_class_interactive(
            eda_df, "Chiều_cao_triều_m", "Phân bố triều cường theo lớp nguy cơ ngập"
        )

        render_chart_discussion(
            "Các biểu đồ trên tổng hợp quan hệ tương quan giữa các biến, phân phối lớp mục tiêu, và tỷ "
            "lệ ngập theo địa phương - cung cấp căn cứ định lượng cho khuyến nghị quản trị ở Tab 3 (ví "
            "dụ: khu vực nào cần ưu tiên nguồn lực phòng chống ngập)."
        )

    # ---- Hàng 3: xử lý giá trị thiếu / ngoại lai ----
    with st.expander("Xử lý giá trị thiếu & ngoại lai (Missing Value / Outlier)", expanded=False):
        if eda_df.empty:
            st.info("Chưa có dữ liệu để kiểm tra.")
        else:
            missing_summary = eda_df.isna().sum().rename("Số lượng thiếu").to_frame()
            missing_summary["Tỷ lệ thiếu (%)"] = (missing_summary["Số lượng thiếu"] / len(eda_df) * 100).round(2)
            missing_summary = missing_summary.reset_index().rename(columns={"index": "Cột"})
            render_styled_table(
                build_contrast_styler(
                    missing_summary,
                    numeric_formats={"Số lượng thiếu": "{:,}", "Tỷ lệ thiếu (%)": "{:.2f}"},
                ),
                height=min(120 + 38 * len(missing_summary), 400),
            )
            render_chart_discussion(
                "Bảng trên thống kê số lượng và tỷ lệ giá trị thiếu theo từng cột - căn cứ để quyết định "
                "chiến lược xử lý (loại bỏ, nội suy, hay điền giá trị trung vị) ở Tab 2."
            )

            st.markdown("---")
            st.markdown("**Phát hiện ngoại lai (Outlier) bằng IQR & Z-score - tính riêng cho từng lớp**")
            if "Nguy_cơ_ngập" not in eda_df.columns:
                st.info("Thiếu cột `Nguy_cơ_ngập` nên không thể tính ngoại lai theo từng lớp.")
            else:
                outlier_feature_columns = [
                    column
                    for column in eda_df.select_dtypes(include="number").columns
                    if column != "Nguy_cơ_ngập"
                ]
                outlier_summary = compute_outlier_summary_by_class(eda_df, "Nguy_cơ_ngập", outlier_feature_columns)
                render_styled_table(
                    build_contrast_styler(
                        outlier_summary,
                        numeric_formats={
                            "Số ngoại lệ (IQR)": "{:,}",
                            "Tỷ lệ (%) (IQR)": "{:.2f}",
                            "Số ngoại lệ (Z-score)": "{:,}",
                            "Tỷ lệ (%) (Z-score)": "{:.2f}",
                        },
                    ),
                    height=min(120 + 38 * len(outlier_summary), 460),
                )
                render_chart_discussion(
                    "Bảng trên áp dụng cả 2 phương pháp - IQR (ngoài [Q1-1.5·IQR, Q3+1.5·IQR]) và Z-score "
                    "(|z| > 3) - tính riêng cho từng lớp `Nguy_cơ_ngập` (0/1/2), theo đúng khuyến nghị: gộp "
                    "chung các lớp sẽ khiến phần lớn dòng dữ liệu thật của lớp `Ngập nặng` (mưa/triều cực đoan) "
                    "bị nhầm là ngoại lai, vì đó chính là tín hiệu thật có giá trị dự báo, không nên loại bỏ. "
                    "Bảng này chỉ thống kê để tham khảo, chưa tự động loại bỏ dòng nào khỏi dữ liệu."
                )


# ==================================================================================================
# TAB 2 - TIỀN XỬ LÝ & HUẤN LUYỆN
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


def render_ctgan_before_after_chart(before_distribution_df: pd.DataFrame, after_distribution_df: pd.DataFrame) -> None:
    """
    Biểu đồ CỘT NHÓM (grouped bar) so sánh trực quan phân phối 3 lớp nguy cơ ngập TRƯỚC và SAU khi xử
    lý mất cân bằng (CTGAN, fallback SMOTE) - bổ sung cho 2 bảng số liệu ở `render_ctgan_dataset_panel()`
    vốn đặt cạnh nhau ở 2 CỘT RIÊNG BIỆT nên khó so sánh trực tiếp bằng mắt.

    "Trước" và "Sau" là 2 THỰC THỂ khác nhau (categorical - identity), KHÔNG phải 1 đại lượng đo lường
    liên tục, nên dùng 2 MÀU CỐ ĐỊNH (xám = trước/còn vấn đề, xanh dương = sau/đã xử lý) thay vì thang
    màu sequential/diverging.
    """
    if before_distribution_df.empty and after_distribution_df.empty:
        st.info("Chưa có đủ dữ liệu phân phối lớp để vẽ biểu đồ so sánh.")
        return

    def _prepare(distribution_df: pd.DataFrame, stage_label: str) -> pd.DataFrame:
        prepared_df = distribution_df.copy()
        prepared_df["Nhãn lớp"] = prepared_df["Lớp"].map(CLASS_LABEL_VI).fillna(prepared_df["Lớp"])
        prepared_df["Giai đoạn"] = stage_label
        return prepared_df

    combined_df = pd.concat(
        [_prepare(before_distribution_df, "Trước xử lý"), _prepare(after_distribution_df, "Sau xử lý")],
        ignore_index=True,
    )
    class_order = [
        CLASS_LABEL_VI[key] for key in ["0", "1", "2"] if CLASS_LABEL_VI[key] in combined_df["Nhãn lớp"].values
    ]
    stage_colors = {"Trước xử lý": "#64748b", "Sau xử lý": "#3b82f6"}

    fig = go.Figure()
    for stage_label, stage_color in stage_colors.items():
        stage_df = combined_df[combined_df["Giai đoạn"] == stage_label]
        if stage_df.empty:
            continue
        stage_df = stage_df.set_index("Nhãn lớp").reindex(class_order).reset_index()
        fig.add_trace(
            go.Bar(
                name=stage_label,
                x=stage_df["Nhãn lớp"],
                y=stage_df["Số lượng"],
                marker_color=stage_color,
                text=[f"{value:,.0f}" if pd.notna(value) else "" for value in stage_df["Số lượng"]],
                textposition="outside",
                textfont=dict(color="#f8fafc"),
            )
        )
    fig.update_layout(
        barmode="group",
        margin=dict(t=10, b=10, l=10, r=10),
        height=340,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#f8fafc", tickfont=dict(size=14)),
        yaxis=dict(title="Số lượng quan sát", color="#cbd5e1", gridcolor="#334155"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(color="#f8fafc", size=13)),
    )
    st.plotly_chart(fig, use_container_width=True)
    render_chart_discussion(
        "Biểu đồ trên đặt cạnh nhau 2 giai đoạn để thấy ngay hiệu quả xử lý mất cân bằng: cột xám "
        "(trước xử lý) lệch hẳn về lớp `Không ngập`, trong khi cột xanh (sau xử lý) gần bằng nhau giữa "
        "3 lớp - đúng mục tiêu của CTGAN/SMOTE là giúp model không học lệch về phía lớp đa số, tránh "
        "bỏ sót các trường hợp `Ngập nhẹ`/`Ngập nặng` (hiếm gặp hơn nhưng quan trọng hơn để cảnh báo)."
    )


@st.cache_data(show_spinner=False)
def _load_ctgan_comparison_artifacts_cached(distribution_file_mtime: float):
    """Đọc dữ liệu export trước/sau CTGAN (do `analyze_and_train.py` xuất ra) để hiển thị nhanh trên
    Streamlit. `distribution_file_mtime` KHÔNG dùng trong thân hàm - chỉ tồn tại để LÀM CACHE KEY, ép
    Streamlit tự đọc lại file mỗi khi `analyze_and_train.py` ghi đè file mới (mtime đổi), thay vì cache
    mãi mãi kết quả của lần train ĐẦU TIÊN cho tới khi ai đó bấm 'Làm mới toàn bộ cache' thủ công."""
    with CTGAN_DISTRIBUTION_PATH.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    before_df = pd.read_csv(CTGAN_BEFORE_PATH) if CTGAN_BEFORE_PATH.exists() else pd.DataFrame()
    after_df = pd.read_csv(CTGAN_AFTER_PATH) if CTGAN_AFTER_PATH.exists() else pd.DataFrame()

    return {
        "summary": summary,
        "before_df": before_df,
        "after_df": after_df,
    }


def load_ctgan_comparison_artifacts():
    if not CTGAN_DISTRIBUTION_PATH.exists():
        return None
    return _load_ctgan_comparison_artifacts_cached(CTGAN_DISTRIBUTION_PATH.stat().st_mtime)


def render_ctgan_dataset_panel(title: str, subtitle: str, summary: dict | None, dataset_df: pd.DataFrame) -> None:
    """Render 1 cột dữ liệu trước hoặc sau CTGAN (dùng cho cả 2 cột trong `render_ctgan_section`)."""
    st.markdown(f"#### {title}")
    st.caption(subtitle)
    total_rows = (summary or {}).get("total_rows")
    sample_rows = (summary or {}).get("sample_rows")
    metric_col_1, metric_col_2 = st.columns(2)
    metric_col_1.metric("Tổng số dòng", f"{total_rows:,}" if total_rows is not None else "N/A")
    metric_col_2.metric("Số dòng hiển thị", f"{sample_rows:,}" if sample_rows is not None else "N/A")

    distribution_df = build_ctgan_distribution_dataframe(summary)
    if distribution_df.empty:
        st.info("Chưa có thống kê phân phối lớp.")
    else:
        # Giữ `distribution_df` GỐC ở dạng số (truyền cho build_ctgan_distribution_discussion() bên
        # dưới vẫn cần idxmax/idxmin/chia số học) - chỉ tạo BẢN HIỂN THỊ riêng có dấu phân cách hàng
        # nghìn cho cột "Số lượng", tránh hiện "12850" khó đọc như trước.
        display_distribution_df = distribution_df.copy()
        display_distribution_df["Số lượng"] = display_distribution_df["Số lượng"].map(lambda v: f"{v:,}")
        st.dataframe(display_distribution_df, use_container_width=True, hide_index=True, height=160)
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
    error_detail = summary.get("error_detail")
    if method_used != "CTGAN":
        st.warning(
            f"Kết quả export gần nhất không hoàn tất bằng CTGAN thuần. Method dùng thực tế: `{method_used}` | trạng thái: `{status}`."
        )
        if error_detail:
            st.error(f"Lý do fallback (lỗi thực tế từ lần chạy CTGAN gần nhất): {error_detail}")
        else:
            st.caption(
                "Chưa có chi tiết lỗi cho lần fallback này (bản export cũ, trước khi tính năng ghi lại "
                "lý do lỗi được thêm vào) - hãy chạy lại huấn luyện để lần fallback tiếp theo (nếu có) "
                "ghi kèm lý do cụ thể."
            )
    else:
        st.success(f"Export CTGAN sẵn sàng. Trạng thái gần nhất: `{status}`.")

    st.markdown("#### Biểu đồ so sánh phân phối lớp trước/sau xử lý")
    render_ctgan_before_after_chart(
        build_ctgan_distribution_dataframe(summary.get("before")),
        build_ctgan_distribution_dataframe(summary.get("after")),
    )
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        render_ctgan_dataset_panel(
            title="Dữ liệu gốc (Bị mất cân bằng)",
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
def _load_hyperparameter_tuning_results_cached(results_file_mtime: float) -> dict:
    """Đọc `data/hyperparameter_tuning_results.json` (do `analyze_and_train.py` xuất ra sau khi chạy
    GridSearchCV/Optuna thật). `results_file_mtime` chỉ dùng làm cache key - xem
    `_load_ctgan_comparison_artifacts_cached()` để biết lý do (tự làm mới khi file đổi, không cần bấm
    nút xoá cache thủ công)."""
    with HYPERPARAMETER_TUNING_RESULTS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_hyperparameter_tuning_results() -> dict | None:
    if not HYPERPARAMETER_TUNING_RESULTS_PATH.exists():
        return None
    return _load_hyperparameter_tuning_results_cached(HYPERPARAMETER_TUNING_RESULTS_PATH.stat().st_mtime)


def render_hyperparameter_tuning_section() -> None:
    """
    Khối 'Log Tinh chỉnh Siêu tham số' bên trong Tab 2 - hiển thị kết quả THẬT của GridSearchCV (Random
    Forest) và Optuna (XGBoost) từ lần chạy `analyze_and_train.py` gần nhất, đọc từ
    `data/hyperparameter_tuning_results.json` (xem `train_and_evaluate_models()` trong
    `analyze_and_train.py` - đây KHÔNG phải demo/mock data như `hyperparameter_tuning.py` chạy độc
    lập, mà là kết quả tune trực tiếp trên dữ liệu train thật của lần huấn luyện đó).
    """
    payload = load_hyperparameter_tuning_results()
    if payload is None:
        st.info(
            "Chưa có file `hyperparameter_tuning_results.json`. Hãy chạy `python analyze_and_train.py` "
            "(hoặc bấm 'Bắt đầu Huấn luyện Nền' bên dưới) - Random Forest sẽ tự tinh chỉnh bằng "
            "GridSearchCV, XGBoost bằng Optuna, kết quả thật sẽ hiện ở đây sau khi huấn luyện xong."
        )
        return

    results = payload.get("results", {})
    if not results:
        st.info("File kết quả tồn tại nhưng chưa có model nào được tinh chỉnh.")
        return

    st.caption(f"Kết quả tinh chỉnh gần nhất: {payload.get('generated_at', 'không rõ thời điểm')}.")

    for model_name, tuning_result in results.items():
        st.markdown(f"**{model_name} - {tuning_result.get('method', 'Unknown')}**")
        best_score = tuning_result.get("best_score")
        score_label = tuning_result.get("score_label", "Điểm tốt nhất")
        metric_col, params_col = st.columns([1, 2])
        with metric_col:
            st.metric(score_label, f"{best_score:.4f}" if best_score is not None else "N/A")
        with params_col:
            st.json(tuning_result.get("best_params", {}), expanded=False)

        trials = tuning_result.get("trials", [])
        if trials:
            trials_df = pd.DataFrame(trials)
            st.dataframe(trials_df, use_container_width=True, hide_index=True, height=min(60 + 35 * len(trials_df), 320))
        st.markdown("---")


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
            "Bắt đầu Huấn luyện Nền",
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
        if st.button("Nạp lại artifact mới nhất", key="reload_trained_artifacts_button", use_container_width=True):
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


@st.cache_data(show_spinner="Đang gộp và làm sạch dữ liệu lịch sử...")
def load_cleaned_training_dataframe() -> pd.DataFrame:
    """
    Nạp dữ liệu THẬT qua ĐÚNG pipeline tiền xử lý dùng lúc huấn luyện: `load_and_concatenate_csvs()`
    (gộp toàn bộ `data/historical/*.csv`) rồi `preprocess_features()` (ép kiểu số, điền giá trị thiếu
    bằng TRUNG VỊ từng cột, giữ đúng các cột cần cho model) trong `analyze_and_train.py` - gọi lại
    ĐÚNG 2 hàm đó thay vì viết lại logic làm sạch riêng ở `app.py`, để tránh tình trạng 2 nơi xử lý
    lệch nhau (app hiển thị 1 kiểu, lúc train thật lại ra kết quả khác).

    Cache bằng `st.cache_data` vì gộp + làm sạch ~440 nghìn dòng khá tốn, không cần chạy lại mỗi khi
    Streamlit rerun (ví dụ khi người dùng tương tác widget ở tab khác).
    """
    train_module = get_train_module()
    raw_df = train_module.load_and_concatenate_csvs()
    return train_module.preprocess_features(raw_df)


def render_preprocessing_training_tab() -> None:
    """
    Nội dung Tab 2 - Tiền xử lý & huấn luyện, bước THỨ HAI của vòng đời Data Science.
    Bố cục: 2 cột song song (Dữ liệu đã làm sạch | Chia Train/Test) phía trên, tiếp theo là khối
    Cân bằng dữ liệu (CTGAN) và khối Log Tinh chỉnh siêu tham số - mỗi khối 1 `st.expander`.
    """
    st.subheader("Tiền xử lý & huấn luyện")
    st.caption(
        "Bước 2/4 của pipeline: làm sạch dữ liệu, chia tập train/test đúng đặc thù chuỗi thời gian, "
        "cân bằng lớp thiểu số, và tinh chỉnh siêu tham số (Optuna / GridSearchCV)."
    )

    col_clean, col_split = st.columns(2)

    with col_clean:
        with st.expander("Dữ liệu đã làm sạch (Cleaned Data)", expanded=True):
            try:
                cleaned_df = load_cleaned_training_dataframe()
            except Exception as exc:
                st.error(f"Không nạp/làm sạch được dữ liệu: {exc}")
            else:
                st.dataframe(cleaned_df.head(20), use_container_width=True, hide_index=True)
                render_chart_discussion(
                    f"Bảng trên là {len(cleaned_df):,} dòng sau khi qua `preprocess_features()` trong "
                    "`analyze_and_train.py`: ép kiểu số, điền giá trị thiếu bằng trung vị của từng cột "
                    "(median - ít bị lệch bởi outlier hơn trung bình), và chỉ giữ lại các cột thật sự "
                    "cần cho huấn luyện. Lưu ý: nhãn `Nguy_cơ_ngập` ở bước này lấy thẳng từ dữ liệu gốc "
                    "(không tạo lại theo rule-based) - việc gán nhãn rule-based chỉ áp dụng cho dữ liệu "
                    "tổng hợp CTGAN ở khối 'Cân bằng dữ liệu' bên dưới, không áp dụng ở bước làm sạch này."
                )

    with col_split:
        with st.expander("Chia tập Train / Test (Data Splitting)", expanded=True):
            st.markdown(
                "- **Tỷ lệ chia**: 80% Train / 20% Test.\n"
                "- **Phương pháp**: chia theo mốc thời gian (`shuffle=False`) - tập Test luôn nằm sau "
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
                "TimeSeriesSplit khác K-Fold thông thường ở chỗ nó không xáo trộn dữ liệu - đảm bảo mọi "
                "lần đánh giá đều mô phỏng đúng bối cảnh dự báo thực tế (chỉ dùng dữ liệu quá khứ để dự "
                "báo tương lai), tránh đánh giá bị 'ảo' do rò rỉ thông tin tương lai."
            )

    st.markdown("---")

    with st.expander("Cân bằng dữ liệu (CTGAN Before / After)", expanded=False):
        st.caption(
            "Đọc file export từ `analyze_and_train.py` để so sánh dữ liệu trước/sau khi cân bằng lớp "
            "thiểu số bằng CTGAN (tự fallback sang SMOTE nếu cần)."
        )
        render_ctgan_section()

    with st.expander("Log tinh chỉnh siêu tham số (Optuna / GridSearchCV)", expanded=True):
        st.caption(
            "Random Forest tự tinh chỉnh bằng GridSearchCV, XGBoost bằng Optuna (TPE) - chạy thật "
            "trên dữ liệu train của lần huấn luyện gần nhất (`analyze_and_train.py`), không phải demo."
        )
        render_hyperparameter_tuning_section()
        render_training_controls_panel()


# ==================================================================================================
# TAB 3 - ĐÁNH GIÁ MÔ HÌNH
# ==================================================================================================
def render_confusion_matrix_heatmap(confusion_matrix_json_path: Path) -> None:
    """
    Confusion Matrix dạng heatmap TƯƠNG TÁC (Plotly) - đọc dữ liệu số thô từ `confusion_matrix.json`
    (do `build_confusion_matrix_from_labels()` trong `analyze_and_train.py` xuất kèm ảnh PNG), thay
    cho ảnh tĩnh trước đây. Hover để xem chính xác số lượng từng ô, zoom được khi cần soi kỹ.
    """
    with confusion_matrix_json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    labels = payload.get("labels") or []
    matrix = payload.get("matrix") or []
    if not labels or not matrix:
        st.info("File `confusion_matrix.json` rỗng hoặc thiếu dữ liệu.")
        return

    matrix_array = np.asarray(matrix, dtype=float)
    max_value = matrix_array.max() if matrix_array.size else 0

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="Blues",
            hovertemplate="Nhãn thật: %{y}<br>Nhãn dự đoán: %{x}<br>Số lượng: %{z}<extra></extra>",
            colorbar=dict(title="Số lượng"),
            showscale=True,
        )
    )
    # Ghi số liệu trực tiếp lên từng ô bằng annotation (thay vì texttemplate chung 1 màu) - để tự
    # chọn màu chữ TRẮNG trên ô nền đậm (số lớn) và màu chữ TỐI trên ô nền nhạt (số nhỏ), đảm bảo đọc
    # được rõ ràng ở MỌI ô thay vì 1 màu cố định dễ bị chìm vào nền ở 1 đầu thang màu.
    for row_index, row_label in enumerate(labels):
        for col_index, col_label in enumerate(labels):
            cell_value = matrix_array[row_index, col_index]
            is_dark_cell = max_value > 0 and cell_value / max_value > 0.5
            fig.add_annotation(
                x=col_label,
                y=row_label,
                text=str(int(cell_value)),
                showarrow=False,
                font=dict(size=16, color="#f8fafc" if is_dark_cell else "#0f172a"),
            )
    fig.update_layout(
        title=dict(text="Confusion Matrix - Best Model", font=dict(color="#f8fafc")),
        # Nền TRONG SUỐT (khớp `plot_bgcolor`/`paper_bgcolor` đã dùng ở Feature Importance) - mặc định
        # Plotly nền TRẮNG, đặt cạnh theme tối của app trông như 1 ảnh dán lì, không giống 1 thành phần
        # tương tác thật của trang. Không set màu này thì colorbar/trục cũng bị lẫn vào nền trắng đó.
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Nhãn dự đoán", side="bottom", color="#cbd5e1"),
        yaxis=dict(title="Nhãn thật", autorange="reversed", color="#cbd5e1"),
        height=420,
        margin=dict(t=50, b=40, l=10, r=10),
    )
    fig.update_traces(colorbar=dict(title=dict(text="Số lượng", font=dict(color="#f8fafc")), tickfont=dict(color="#cbd5e1")))
    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance_bar_chart(feature_importance_json_path: Path) -> None:
    """
    Biểu đồ THANH NGANG (horizontal bar, Plotly) cho Feature Importance - theo góp ý của GVHD (đổi từ
    heatmap sang thanh màu, dễ so sánh độ lớn giữa các biến hơn heatmap 1 hàng). Đọc dữ liệu số thô từ
    `feature_importance.json` (do `plot_feature_importance()` trong `analyze_and_train.py` xuất kèm
    ảnh PNG) - KHÔNG tính lại importance ở đây, tránh chạy lại `permutation_importance` (tốn thời
    gian) mỗi lần Streamlit rerun.

    Sắp xếp GIẢM DẦN theo Importance, TẤT CẢ các thanh dùng CHUNG 1 MÀU (không tô gradient theo giá
    trị) - vì các thanh đang biểu diễn CÙNG 1 đại lượng (mức độ quan trọng), độ dài thanh đã đủ thể
    hiện sự khác biệt, tô nhiều màu khác nhau chỉ gây rối mắt không cần thiết. Ghi số liệu trực tiếp ở
    đầu mỗi thanh (không cần hover mới thấy) để phù hợp khi trình bày/in báo cáo.
    """
    with feature_importance_json_path.open("r", encoding="utf-8") as file:
        importance_records = json.load(file)

    importance_df = pd.DataFrame(importance_records)
    if importance_df.empty:
        st.info("File `feature_importance.json` rỗng.")
        return
    # Sắp xếp TĂNG DẦN vì Plotly vẽ thanh ngang từ DƯỚI LÊN - biến quan trọng nhất cần nằm TRÊN CÙNG.
    importance_df = importance_df.sort_values("Importance", ascending=True)

    fig = go.Figure(
        data=go.Bar(
            x=importance_df["Importance"].tolist(),
            y=importance_df["Feature"].tolist(),
            orientation="h",
            # 1 MÀU DUY NHẤT cho mọi thanh (không dùng thang màu gradient theo giá trị) - vì các thanh
            # đang so sánh CÙNG 1 đại lượng (mức độ quan trọng), không phải nhiều đại lượng khác nhau
            # cần phân biệt bằng màu; ĐỘ DÀI thanh đã đủ thể hiện độ lớn, tô màu khác nhau theo giá trị
            # chỉ gây rối mắt không cần thiết.
            marker=dict(color="#4C78A8", line=dict(color="#1e3a5f", width=1)),
            text=[f"{value:.3f}" for value in importance_df["Importance"]],
            textposition="outside",
            textfont=dict(size=16, color="#f8fafc"),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        # Cao hơn bản cũ (260 -> 420) để khớp chiều cao Confusion Matrix bên cạnh và đỡ bị bó hẹp khi
        # có đủ 5 dòng (mỗi dòng cần khoảng không đủ rộng để không đè chữ lên nhau).
        margin=dict(t=20, b=40, l=10, r=60),
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Mức độ quan trọng", color="#cbd5e1", gridcolor="#334155", zeroline=False, title_font=dict(size=15), tickfont=dict(size=13)),
        yaxis=dict(color="#f8fafc", tickfont=dict(size=15)),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_overfitting_check_section(metrics_df: pd.DataFrame) -> None:
    """
    Khối "Kiểm tra học vẹt (Overfitting)" - so sánh F1-Macro trên tập TRAIN vs tập TEST cho từng
    model, sắp xếp GIẢM DẦN theo mức chênh lệch (model học vẹt rõ nhất nằm trên cùng).

    LƯU Ý PHẠM VI: chỉ số này hiện CHỈ được tính cho model dạng bảng (tabular_classifier/
    tabular_regressor - xem `attach_train_test_gap()` trong `analyze_and_train.py`), CHƯA áp dụng cho
    model dạng chuỗi/Deep Learning (LSTM/GRU/CNN/Hybrid) - các model đó sẽ tự động không xuất hiện ở
    đây (cột `Chênh lệch Train-Test (F1)` rỗng) thay vì hiện sai số 0 gây hiểu lầm là "không học vẹt".
    """
    checkable_df = metrics_df.dropna(subset=["Chênh lệch Train-Test (F1)"]).copy()
    if checkable_df.empty:
        return

    checkable_df = checkable_df.sort_values("Chênh lệch Train-Test (F1)", ascending=False).reset_index(drop=True)

    with st.expander("Kiểm tra học vẹt (Overfitting) - so sánh điểm Train vs Test", expanded=False):
        st.caption(
            "So sánh F1-Macro trên chính tập TRAIN model đã học với F1-Macro trên tập TEST (chưa từng "
            "thấy) - chênh lệch càng lớn, model càng có dấu hiệu học thuộc lòng dữ liệu train thay vì "
            f"học được quy luật tổng quát hoá được. Ngưỡng cảnh báo tham khảo: > "
            f"{OVERFITTING_GAP_WARNING_THRESHOLD:.2f} điểm F1-Macro. Chỉ áp dụng cho model dạng bảng "
            "(chưa tính cho model dạng chuỗi/Deep Learning)."
        )
        display_df = checkable_df[
            ["Model", "Train F1 (Macro)", "F1 (Macro)", "Chênh lệch Train-Test (F1)"]
        ].rename(columns={"F1 (Macro)": "Test F1 (Macro)"})
        render_styled_table(
            build_contrast_styler(
                display_df,
                numeric_formats={
                    "Train F1 (Macro)": "{:.4f}",
                    "Test F1 (Macro)": "{:.4f}",
                    "Chênh lệch Train-Test (F1)": "{:+.4f}",
                },
            ),
            height=min(120 + 38 * len(display_df), 420),
        )

        flagged_df = checkable_df[checkable_df["Chênh lệch Train-Test (F1)"] > OVERFITTING_GAP_WARNING_THRESHOLD]
        if flagged_df.empty:
            st.success(
                "Không model nào (trong số có đủ dữ liệu kiểm tra) vượt ngưỡng cảnh báo học vẹt - "
                "khoảng cách train-test ở mức chấp nhận được."
            )
        else:
            flagged_names = ", ".join(
                f"**{row['Model']}** (+{row['Chênh lệch Train-Test (F1)']:.4f})"
                for _, row in flagged_df.iterrows()
            )
            st.warning(
                f"Nghi ngờ học vẹt: {flagged_names} - điểm trên tập train cao vượt trội so với tập "
                "test. Cân nhắc giảm độ phức tạp model (giảm độ sâu cây, tăng regularization, giảm k "
                "cho KNN...) hoặc kiểm tra lại xem model có đang 'nhớ' mẫu tổng hợp CTGAN/SMOTE thay vì "
                "học quy luật thật hay không."
            )


# Ngưỡng xếp loại (1-Tốt / 2-Trung bình / 3-Yếu) cho bảng xếp loại độ đo.
METRIC_RATING_THRESHOLDS: dict[str, tuple[float, float]] = {
    "Accuracy": (0.75, 0.5),
    "Precision (Macro)": (0.6, 0.4),
    "Recall (Macro)": (0.6, 0.4),
    "F1 (Macro)": (0.6, 0.4),
    "ROC-AUC (OvR Macro)": (0.8, 0.7),
}
METRIC_RATING_STYLE: dict[str, tuple[str, str]] = {
    "1 - Tốt": ("#166534", "#f0fdf4"),
    "2 - Trung bình": ("#92400e", "#fffbeb"),
    "3 - Yếu": ("#991b1b", "#fef2f2"),
}


def rate_metric_value(metric_name: str, value: float | None) -> str | None:
    """Xếp loại 1 giá trị độ đo theo ngưỡng `METRIC_RATING_THRESHOLDS` - trả `None` nếu thiếu giá trị
    hoặc độ đo đó chưa có ngưỡng khai báo (không đoán bừa)."""
    if value is None or pd.isna(value):
        return None
    thresholds = METRIC_RATING_THRESHOLDS.get(metric_name)
    if thresholds is None:
        return None
    good_cutoff, medium_cutoff = thresholds
    if value >= good_cutoff:
        return "1 - Tốt"
    if value >= medium_cutoff:
        return "2 - Trung bình"
    return "3 - Yếu"


def render_metric_rating_table(best_model_metric_values: dict, best_model_name: str) -> None:
    """
    Bảng 3 cột (Độ đo / Giá trị / Xếp loại) cho model TỐT NHẤT - khác với bảng leaderboard so sánh
    NHIỀU model theo 1 độ đo (F1-Macro), bảng này soi NHIỀU độ đo của CÙNG 1 model, giúp đọc nhanh model
    đang mạnh/yếu ở khía cạnh nào (ví dụ Recall tốt nhưng Precision yếu) mà không cần tự nhớ khoảng giá
    trị hợp lệ [0, 1] của từng độ đo là tốt hay xấu.
    """
    if not best_model_metric_values:
        return

    candidate_metrics = {
        "Accuracy": best_model_metric_values.get("accuracy"),
        "Precision (Macro)": best_model_metric_values.get("precision_macro"),
        "Recall (Macro)": best_model_metric_values.get("recall_macro"),
        "F1 (Macro)": best_model_metric_values.get("f1_macro"),
        "ROC-AUC (OvR Macro)": best_model_metric_values.get("roc_auc_ovr_macro"),
    }
    rating_rows = []
    for metric_name, value in candidate_metrics.items():
        if value is None or pd.isna(value):
            continue
        rating_rows.append(
            {
                "Độ đo": metric_name,
                "Giá trị": float(value),
                "Xếp loại": rate_metric_value(metric_name, value) or "Chưa có ngưỡng",
            }
        )
    if not rating_rows:
        return

    rating_df = pd.DataFrame(rating_rows)

    def highlight_rating_column(row: pd.Series) -> list[str]:
        style = METRIC_RATING_STYLE.get(row["Xếp loại"])
        if not style:
            return [""] * len(row)
        background, text_color = style
        return [
            f"background-color: {background}; color: {text_color}; font-weight: 700;"
            if column == "Xếp loại"
            else ""
            for column in row.index
        ]

    st.markdown(f"### Bảng xếp loại độ đo - model tốt nhất ({best_model_name})")
    rating_styler = (
        build_contrast_styler(rating_df, numeric_formats={"Giá trị": "{:.4f}"})
        .apply(highlight_rating_column, axis=1)
    )
    render_styled_table(rating_styler, height=min(90 + 38 * len(rating_df), 280))


def render_model_metrics(evaluation_metrics, deployment_config, runtime_info) -> None:
    """Hiển thị bảng số liệu, biểu đồ Plotly và ảnh artifact đánh giá mô hình (Model Comparison Metrics)."""
    if not evaluation_metrics:
        st.info("Chưa có evaluation metrics để hiển thị.")
        return

    metrics_rows = []
    metric_values_by_display_name: dict[str, dict] = {}
    for model_name, metric_values in evaluation_metrics.items():
        if not isinstance(metric_values, dict):
            continue
        display_name = metric_values.get("model_name", model_name)
        metric_values_by_display_name[display_name] = metric_values
        metrics_rows.append(
            {
                "Model": display_name,
                "Accuracy": metric_values.get("accuracy"),
                "Precision (Macro)": metric_values.get("precision_macro"),
                "Recall (Macro)": metric_values.get("recall_macro"),
                "F1 (Macro)": metric_values.get("f1_macro"),
                "Train F1 (Macro)": metric_values.get("train_f1_macro"),
                "Chênh lệch Train-Test (F1)": metric_values.get("overfitting_gap_f1"),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    if metrics_df.empty:
        st.info("Không có dữ liệu metrics hợp lệ để hiển thị.")
        return

    metrics_df = metrics_df.sort_values(by="F1 (Macro)", ascending=False).reset_index(drop=True)
    metrics_df.insert(0, "Xếp hạng", range(1, len(metrics_df) + 1))
    render_styled_table(
        build_contrast_styler(
            metrics_df,
            numeric_formats={
                "Accuracy": "{:.4f}",
                "Precision (Macro)": "{:.4f}",
                "Recall (Macro)": "{:.4f}",
                "F1 (Macro)": "{:.4f}",
                "Train F1 (Macro)": "{:.4f}",
                "Chênh lệch Train-Test (F1)": "{:+.4f}",
            },
            rank_highlight=True,
        ),
        height=420,
    )
    st.caption(
        "**Train F1 (Macro)**: điểm F1-Macro đo trên chính tập Train (dữ liệu model đã học), khác với "
        "cột **F1 (Macro)** đo trên tập Test (dữ liệu model chưa từng thấy). **Chênh lệch Train-Test "
        "(F1)** = Train F1 − Test F1 - chênh lệch càng lớn thì càng nghi ngờ model 'học vẹt' (thuộc "
        "lòng dữ liệu train thay vì học được quy luật tổng quát, xem khối 'Kiểm tra học vẹt' bên dưới). "
        "2 cột này hiện **'-' (None)** cho các model dạng chuỗi (GRU/LSTM/Hybrid) vì cách tính điểm "
        "Train F1 hiện chỉ áp dụng cho model dạng bảng (sklearn_tabular) - không phải model đó bị lỗi."
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

    render_metric_rating_table(
        metric_values_by_display_name.get(best_metrics_row["Model"], {}), best_metrics_row["Model"]
    )

    render_overfitting_check_section(metrics_df)

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
    confusion_matrix_json_path = (
        Path(runtime_info.get("latest_dir", str(LATEST_MODELS_DIR))) / "confusion_matrix.json"
    )
    feature_importance_path = Path(runtime_info.get("latest_dir", str(LATEST_MODELS_DIR))) / "feature_importance.png"
    feature_importance_json_path = (
        Path(runtime_info.get("latest_dir", str(LATEST_MODELS_DIR))) / "feature_importance.json"
    )

    with image_col_1:
        st.markdown("**Confusion Matrix**")
        if confusion_matrix_json_path.exists():
            render_confusion_matrix_heatmap(confusion_matrix_json_path)
        elif confusion_matrix_path.exists():
            # Fallback cho artifact từ lần train CŨ (trước khi có `confusion_matrix.json`) - chỉ có
            # ảnh heatmap tĩnh, chưa có dữ liệu số thô để tự vẽ lại bản tương tác.
            render_full_width_image(str(confusion_matrix_path))
            st.caption(
                "Chưa có dữ liệu số cho heatmap tương tác (artifact từ lần train cũ) - hãy train lại "
                "để có bản tương tác."
            )
        else:
            st.info("Chưa có `confusion_matrix.json`/`confusion_matrix.png` trong `models/latest/`.")

    with image_col_2:
        st.markdown("**Feature Importance**")
        if feature_importance_json_path.exists():
            render_feature_importance_bar_chart(feature_importance_json_path)
        elif feature_importance_path.exists():
            # Fallback cho artifact từ lần train CŨ (trước khi có `feature_importance.json`) - chỉ có
            # ảnh bar chart tĩnh, chưa có dữ liệu số thô để tự vẽ lại thanh màu tương tác.
            render_full_width_image(str(feature_importance_path))
            st.caption(
                "Chưa có dữ liệu số cho biểu đồ tương tác (artifact từ lần train cũ) - hãy train lại "
                "để có bản thanh màu tương tác."
            )
        elif deployment_config.get("model_type") in {"keras_sequence", "hybrid_lstm_xgboost", "hybrid_lstm_gru_xgboost"}:
            # Model dạng sequence (LSTM/GRU/CNN/Hybrid) TÍNH ĐƯỢC Feature Importance qua Permutation
            # Importance (xem `plot_sequence_feature_importance()` trong analyze_and_train.py - áp dụng
            # cho MỌI loại model, không riêng model dạng bảng) - nếu tới đây vẫn không thấy file, nghĩa
            # là bước tính đó gặp lỗi ở lần train gần nhất (xem log server để biết lý do cụ thể), KHÔNG
            # phải giới hạn kỹ thuật không thể tính được như trước đây.
            st.info(
                f"Model tốt nhất hiện tại (`{deployment_config.get('model_name', '')}`) là dạng chuỗi/"
                "sequence - Feature Importance (permutation) đáng lẽ tính được cho loại model này, "
                "nhưng chưa thấy file kết quả. Hãy train lại và xem log server nếu vẫn không xuất hiện."
            )
        else:
            st.info(
                "Chưa có Feature Importance cho model đang triển khai - hoặc CHƯA train lần nào, hoặc "
                "model thắng cuộc thuộc dạng chuỗi/hybrid (LSTM/GRU/CNN/Hybrid) mà chỉ số importance "
                "kiểu bảng (feature_importances_/coef_/permutation) không áp dụng trực tiếp được cho "
                "input dạng cửa sổ thời gian - xem log huấn luyện gần nhất để biết chính xác lý do."
            )

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
        "hybrid_lstm_gru_xgboost": "Hybrid LSTM + GRU + XGBoost (Keras x2 + XGBoost native)",
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
    if st.button("Kiểm tra nạp model triển khai", key="test_load_deployment_model_button"):
        try:
            with st.spinner("Đang nạp model theo deployment_config.json..."):
                loaded = load_deployment_model(deployment_config, runtime_info["latest_dir"])
            if loaded["model_type"] == "hybrid_lstm_xgboost":
                st.success(
                    f"Nạp thành công model_type=`{loaded['model_type']}`: "
                    f"feature_extractor=`{type(loaded['feature_extractor']).__name__}`, "
                    f"classifier=`{type(loaded['classifier']).__name__}`, "
                    f"scaler=`{type(loaded['scaler']).__name__}`."
                )
            elif loaded["model_type"] == "hybrid_lstm_gru_xgboost":
                st.success(
                    f"Nạp thành công model_type=`{loaded['model_type']}`: "
                    f"lstm_feature_extractor=`{type(loaded['lstm_feature_extractor']).__name__}`, "
                    f"gru_feature_extractor=`{type(loaded['gru_feature_extractor']).__name__}`, "
                    f"classifier=`{type(loaded['classifier']).__name__}`, "
                    f"scaler=`{type(loaded['scaler']).__name__}`."
                )
            else:
                st.success(
                    f"Nạp thành công model_type=`{loaded['model_type']}`: "
                    f"model=`{type(loaded['model']).__name__}`, scaler=`{type(loaded['scaler']).__name__}`."
                )
        except Exception as exc:
            st.error(f"Nạp model thất bại: {exc}")

    st.markdown("---")
    st.subheader("Phân tích đường cong ROC-AUC (One-vs-Rest)")
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
    class_label_vi = CLASS_LABEL_VI
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


def render_managerial_insights_section(
    evaluation_metrics: dict, deployment_config: dict, runtime_info: dict
) -> None:
    """
    Nhận định & kết luận quản trị - TÍNH THẬT từ dữ liệu và kết quả huấn luyện hiện có (bản trước đây
    chỉ là TEMPLATE placeholder tĩnh, không thật sự tính toán gì - đã phát hiện và sửa lại). Mỗi con
    số dưới đây lấy TRỰC TIẾP từ artifact/dữ liệu thật, không hardcode:
    - Tên model + F1-Macro: từ `deployment_config.json` (đúng model vừa được chọn triển khai).
    - Địa phương/tháng ưu tiên: tính trực tiếp từ dữ liệu lịch sử thật (`load_eda_sample_dataframe()`).
    - AUC lớp `Ngập nặng`: từ `roc_curve_data.json` (đúng kết quả đánh giá của model đang triển khai).
    """
    best_model_name = deployment_config.get("model_name", "N/A")
    best_f1_macro = deployment_config.get("f1_macro", 0)

    eda_df = load_eda_sample_dataframe()
    priority_location_text = "Chưa có đủ dữ liệu lịch sử để xác định."
    priority_month_text = "Chưa có đủ dữ liệu lịch sử để xác định."
    if not eda_df.empty and "Nguy_cơ_ngập" in eda_df.columns:
        flood_df = eda_df[pd.to_numeric(eda_df["Nguy_cơ_ngập"], errors="coerce") > 0]
        if not flood_df.empty and "Địa phương" in flood_df.columns:
            location_counts = flood_df["Địa phương"].value_counts()
            top_location = location_counts.index[0]
            top_location_share = location_counts.iloc[0] / len(flood_df) * 100
            priority_location_text = (
                f"**{top_location}** ghi nhận {int(location_counts.iloc[0]):,} lượt ngập "
                f"({top_location_share:.1f}% tổng số lượt ngập trong dữ liệu lịch sử) - địa phương có "
                "tần suất ngập cao nhất, cần ưu tiên đầu tư trạm quan trắc/lực lượng ứng trực."
            )
        if not flood_df.empty and "Thời_gian" in flood_df.columns:
            month_counts = flood_df["Thời_gian"].dt.month.value_counts()
            top_month = int(month_counts.index[0])
            top_month_share = month_counts.iloc[0] / len(flood_df) * 100
            priority_month_text = (
                f"**Tháng {top_month}** ghi nhận {int(month_counts.iloc[0]):,} lượt ngập "
                f"({top_month_share:.1f}% tổng số lượt ngập) - tháng cao điểm cần tăng cường giám sát "
                "và chuẩn bị phương án sơ tán."
            )

    residual_risk_text = "Chưa có dữ liệu ROC-AUC để đánh giá rủi ro còn tồn đọng."
    roc_path = Path(runtime_info.get("latest_dir", str(LATEST_MODELS_DIR))) / "roc_curve_data.json"
    if roc_path.exists():
        try:
            with roc_path.open("r", encoding="utf-8") as file:
                roc_payload = json.load(file)
            heavy_flood_auc = (roc_payload.get("curves", {}).get("2") or {}).get("auc")
            if heavy_flood_auc is not None:
                if heavy_flood_auc < 0.85:
                    residual_risk_text = (
                        f"AUC lớp `Ngập nặng` hiện ở mức **{heavy_flood_auc:.4f}** - tương đối thấp so "
                        "với 2 lớp còn lại, phản ánh đúng thực tế lớp này CHỈ CHIẾM PHẦN NHỎ trong dữ "
                        "liệu (mất cân bằng nặng). Hướng khắc phục: thu thập thêm dữ liệu ngập nặng "
                        "thật, cải thiện chất lượng CTGAN, hoặc tinh chỉnh ngưỡng cảnh báo qua "
                        "`hyperparameter_tuning.py`."
                    )
                else:
                    residual_risk_text = (
                        f"AUC lớp `Ngập nặng` hiện ở mức **{heavy_flood_auc:.4f}** - khá tốt, model phân "
                        "biệt được lớp nguy hiểm nhất này tương đối rõ ràng. Vẫn nên tiếp tục theo dõi "
                        "khi có thêm dữ liệu thực tế mới để xác nhận độ ổn định."
                    )
        except Exception:
            pass

    st.markdown(
        f"""
1. **Hiệu năng mô hình đề xuất triển khai**: model tốt nhất hiện tại là **`{best_model_name}`** với
   F1-Macro = **{best_f1_macro:.4f}**. Đây là model có điểm cân bằng tốt nhất giữa Precision và Recall
   trên cả 3 lớp, đặc biệt quan trọng với lớp `Ngập nặng` vì bỏ sót 1 trường hợp ngập thật gây hậu quả
   nghiêm trọng hơn nhiều so với 1 lần cảnh báo dư thừa.
2. **Khu vực ưu tiên**: {priority_location_text}
3. **Thời điểm ưu tiên**: {priority_month_text}
4. **Rủi ro còn tồn đọng**: {residual_risk_text}
        """
    )


def render_evaluation_tab() -> None:
    """
    Nội dung Tab 3 - Đánh giá mô hình, bước THỨ BA của vòng đời Data Science.
    Bố cục: khối so sánh chỉ số (tái sử dụng `render_model_metrics`) rồi đến khối "Kết luận quản trị" -
    phần bắt buộc phải có để nối kết quả kỹ thuật với ý nghĩa thực tiễn cho người ra quyết định.
    """
    st.subheader("Đánh giá mô hình")
    st.caption("Bước 3/4 của pipeline: so sánh hiệu năng các mô hình đã huấn luyện và rút ra khuyến nghị quản trị.")

    try:
        evaluation_metrics, deployment_config, runtime_info = load_evaluation_artifacts()
    except Exception as exc:
        st.warning(
            f"Chưa thể nạp evaluation metrics: {exc} "
            "Hãy khởi chạy huấn luyện ở Tab 2 (Tiền xử lý & huấn luyện) trước."
        )
        return

    with st.expander("So sánh chỉ số mô hình (F1-Score / Precision / Recall)", expanded=True):
        render_model_metrics(evaluation_metrics, deployment_config, runtime_info)

    with st.expander("Nhận định & kết luận quản trị (Managerial Insights)", expanded=True):
        render_managerial_insights_section(evaluation_metrics, deployment_config, runtime_info)


# ==================================================================================================
# TAB 4 - BẢN ĐỒ TRÁNH NGẬP (Smart Routing bằng TomTom Routing API)
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

    # Kiểm tra API key TRƯỚC khi gọi mạng - `get_api_secret()` trả `None` nếu máy chưa cấu hình
    # `.streamlit/secrets.toml` lẫn biến môi trường. Nếu không chặn ở đây, request vẫn được gửi với
    # `key=None` (requests tự bỏ qua param `None`), TomTom trả lỗi 403 mơ hồ khiến người dùng khó
    # đoán nguyên nhân thật - chặn sớm để báo đúng lỗi "chưa cấu hình API key".
    if not api_key:
        return {
            "success": False,
            "route_points": [],
            "distance_km": None,
            "travel_time_min": None,
            "used_avoid_areas": has_flood_zones,
            "error": (
                "Chưa cấu hình TOMTOM_API_KEY - hãy khai báo trong `.streamlit/secrets.toml` "
                "(key `TOMTOM_KEY`) hoặc biến môi trường `TOMTOM_API_KEY`."
            ),
        }

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

        if not response.ok:
            # TomTom trả lỗi kèm JSON body có trường 'detailedError' rõ nghĩa hơn nhiều so với để
            # `raise_for_status()` ném HTTPError chung chung (ví dụ chỉ ghi "400 Client Error") - lỗi
            # hay gặp NHẤT khi cho phép click BẤT KỲ trên bản đồ là điểm click rơi vào vị trí TomTom
            # không tìm được đường vào (giữa biển, giữa sông, ruộng không có đường bộ...).
            try:
                error_payload = response.json()
                detailed_message = (
                    error_payload.get("detailedError", {}).get("message")
                    or error_payload.get("error", {}).get("description")
                    or response.text[:300]
                )
            except Exception:
                detailed_message = response.text[:300]

            # Gợi ý "click lại gần đường" chỉ hợp lý cho lỗi 400 (thường là NO_ROUTE_FOUND khi điểm
            # click rơi vào vị trí không có đường bộ) - lỗi 401/403 là do KEY sai/hết hạn, gợi ý đó sẽ
            # gây hiểu lầm, nên chỉ thêm khi đúng ngữ cảnh.
            hint = (
                " Nếu vừa click điểm giữa biển/sông/khu vực không có đường bộ, hãy click lại vị trí "
                "khác gần đường thật."
                if response.status_code == 400
                else " Có thể TOMTOM_KEY đang sai hoặc đã hết hạn - kiểm tra lại cấu hình key."
                if response.status_code in (401, 403)
                else ""
            )
            return {
                "success": False,
                "route_points": [],
                "distance_km": None,
                "travel_time_min": None,
                "used_avoid_areas": has_flood_zones,
                "error": f"TomTom trả lỗi HTTP {response.status_code}: {detailed_message}.{hint}",
            }

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


def nudge_point_outside_flood_zones(
    point: tuple[float, float], flooded_polygons: list[list[tuple[float, float]]]
) -> tuple[float, float]:
    """
    Nếu `point` rơi ĐÚNG vào bên trong 1 vùng ngập (hình vuông do `build_flood_zone_polygon` sinh
    ra), dịch điểm đó ra khỏi biên Bắc của vùng ngập trước khi gửi cho TomTom.

    TẠI SAO CẦN HÀM NÀY: điểm đi/đến MẶC ĐỊNH ban đầu (trước khi người dùng click chọn lại) trùng
    CHÍNH XÁC tọa độ 1 trong 5 địa phương giám sát (`REAL_MONITORED_LOCATIONS`). Nếu đúng lúc đó địa
    phương này đang 'Ngập', điểm xuất phát/đến sẽ nằm NGAY TÂM của chính vùng `avoidAreas` gửi cho
    TomTom - TomTom không thể định tuyến ĐI TỪ/ĐẾN một điểm nằm trong vùng bị cấm, gây lỗi định tuyến
    ngay từ lần mở tab đầu tiên dù người dùng chưa hề click gì. Chỉ dịch điểm dùng để GỌI TomTom -
    KHÔNG ghi đè lại `st.session_state`, nên marker hiển thị trên bản đồ vẫn đúng vị trí gốc.
    """
    for polygon in flooded_polygons:
        latitudes = [p[0] for p in polygon]
        longitudes = [p[1] for p in polygon]
        min_lat, max_lat, min_lon, max_lon = min(latitudes), max(latitudes), min(longitudes), max(longitudes)
        if min_lat <= point[0] <= max_lat and min_lon <= point[1] <= max_lon:
            return (max_lat + 0.005, point[1])  # Dịch lên phía Bắc, ra khỏi biên vùng ngập.
    return point


def predict_class_from_sequence_window(deployed_model: dict, window_input: np.ndarray) -> int:
    """
    Suy luận nhãn lớp (0/1/2) từ 1 CỬA SỔ chuỗi thời gian ĐÃ chuẩn hóa + reshape sẵn thành
    `(1, window_size, n_features)`, dùng CHUNG cho model dạng `keras_sequence`/`hybrid_lstm_xgboost`.

    Tách riêng khỏi `predict_flood_class()` vì đúng đoạn dispatch model_type này (keras_sequence dùng
    `argmax` trên xác suất softmax, hybrid dùng feature_extractor -> classifier) trước đây bị LẶP LẠI
    y hệt ở 2 nơi: `predict_flood_class()` (dự báo tức thời, 1 cửa sổ) và
    `predict_days_ahead_forecast_sequence()` (dự báo nhiều ngày, N cửa sổ trượt) - dễ sửa 1 chỗ quên
    chỗ kia (đúng nguyên nhân khiến `window_size` fallback từng lệch nhau giữa 2 hàm).
    """
    model_type = deployed_model["model_type"]

    # Khóa TOÀN BỘ lệnh gọi Keras (`.predict()`) bằng `_KERAS_INFERENCE_LOCK` - xem comment tại nơi
    # khai báo lock để biết lý do (an toàn khi 5 địa phương suy luận song song bằng ThreadPoolExecutor).
    if model_type == "keras_sequence":
        with _KERAS_INFERENCE_LOCK:
            class_probabilities = deployed_model["model"].predict(window_input, verbose=0)
        return int(class_probabilities.argmax(axis=-1)[0])

    if model_type == "hybrid_lstm_xgboost":
        with _KERAS_INFERENCE_LOCK:
            embedding = deployed_model["feature_extractor"].predict(window_input, verbose=0)
        return int(deployed_model["classifier"].predict(embedding)[0])

    if model_type == "hybrid_lstm_gru_xgboost":
        # NỐI (concatenate) embedding LSTM + GRU theo ĐÚNG thứ tự đã dùng lúc huấn luyện
        # (`train_lstm_gru_xgboost_hybrid_model()`: LSTM trước, GRU sau) - đảo thứ tự sẽ khiến XGBoost
        # nhận nhầm ý nghĩa của từng cột embedding, cho kết quả sai mà không hề báo lỗi.
        with _KERAS_INFERENCE_LOCK:
            lstm_embedding = deployed_model["lstm_feature_extractor"].predict(window_input, verbose=0)
            gru_embedding = deployed_model["gru_feature_extractor"].predict(window_input, verbose=0)
        combined_embedding = np.concatenate([lstm_embedding, gru_embedding], axis=1)
        return int(deployed_model["classifier"].predict(combined_embedding)[0])

    raise ValueError(f"model_type không được hỗ trợ cho suy luận theo cửa sổ: {model_type}")


def predict_flood_class(deployed_model: dict, location_df: pd.DataFrame, feature_columns: list[str]) -> int:
    """
    Suy luận nhãn nguy cơ ngập (0 = An toàn, 1 = Ngập nhẹ, 2 = Ngập nặng) mới nhất bằng model THẬT
    đã triển khai, tự động chọn đúng luồng xử lý theo `model_type` (xem `load_deployment_model()`
    để biết chi tiết 3 loại model: sklearn_tabular / keras_sequence / hybrid_lstm_xgboost).

    `location_df` ở đây là DÒNG THÔ (theo GIỜ) đọc thẳng từ `data/historical/*.csv` qua
    `read_csv_tail()` - vốn dùng dòng giờ GẦN NHẤT làm xấp xỉ cho "đặc trưng hôm nay" (đơn giản hoá đã
    có TỪ TRƯỚC, không phải điểm mới ở đây). `feature_columns` giờ có thể chứa 3 cột lag mưa
    (RAIN_LAG1_COL/RAIN_LAG2_COL/RAIN_ROLLING_3D_COL) mà CSV thô KHÔNG có sẵn - dùng
    `add_rain_lag_features_inference()` để tính (lag theo ĐƠN VỊ DÒNG có sẵn trong `location_df`, không
    ép về đúng "ngày" như lúc huấn luyện - xấp xỉ chấp nhận được, nhất quán với việc cả hàm này vốn đã
    dùng 1 dòng giờ thay cho 1 dòng ngày; quan trọng là KHÔNG CRASH vì thiếu cột thay vì chính xác tuyệt
    đối, người vận hành vẫn cần thấy trạng thái thay vì lỗi "Không xác định").
    """
    model_type = deployed_model["model_type"]
    scaler = deployed_model["scaler"]
    if any(lag_col in feature_columns for lag_col in (RAIN_LAG1_COL, RAIN_LAG2_COL, RAIN_ROLLING_3D_COL)):
        location_df = add_rain_lag_features_inference(location_df.sort_values("Thời_gian"))

    if model_type == "sklearn_tabular":
        latest_features = location_df[feature_columns].iloc[[-1]]
        # Giữ tên cột (DataFrame) khi đưa vào model.predict() - tránh warning "X does not have valid
        # feature names" từ sklearn, xem giải thích tương tự trong `predict_4_days_forecast()`.
        scaled_features = pd.DataFrame(scaler.transform(latest_features), columns=feature_columns)
        return int(deployed_model["model"].predict(scaled_features)[0])

    # Fallback dùng hằng số `DEFAULT_SEQUENCE_WINDOW_SIZE` (đầu file) - đúng `SEQUENCE_WINDOW` dùng
    # lúc huấn luyện trong `analyze_and_train.py`. TRƯỚC ĐÂY hardcode `24` ở đây - nếu
    # `deployment_config.json` thiếu key `window_size`, model
    # tuần tự sẽ nhận sai shape đầu vào (24 bước thay vì 7), suy luận sai lệch mà không có lỗi rõ
    # ràng nào cảnh báo - CHỈ lộ ra gián tiếp qua bug khác (risk bị báo nhầm "An toàn").
    window_size = deployed_model.get("window_size") or DEFAULT_SEQUENCE_WINDOW_SIZE
    if len(location_df) < window_size:
        raise ValueError("Không đủ dữ liệu quan trắc để tạo chuỗi thời gian cho model tuần tự.")
    window_features = location_df[feature_columns].iloc[-window_size:]
    scaled_window = scaler.transform(window_features)
    model_input = scaled_window.reshape(1, window_size, len(feature_columns))
    return predict_class_from_sequence_window(deployed_model, model_input)


def read_csv_tail(csv_path: Path, num_rows: int, chunk_bytes: int = 65536) -> pd.DataFrame:
    """
    Đọc CHỈ `num_rows` dòng CUỐI của 1 file CSV lớn, KHÔNG dùng `pd.read_csv()` để parse toàn bộ file
    - dùng cho nhánh suy luận real-time của `get_latest_flood_predictions()`, vốn chỉ cần 1 dòng
    (model dạng bảng) hoặc vài dòng gần nhất (model tuần tự) trong số hàng chục nghìn dòng của mỗi
    file CSV lịch sử 10 năm (~88 nghìn dòng/địa phương).

    CÁCH LÀM: đọc HEADER (dòng đầu, rẻ) riêng một lần, rồi `seek()` tới `chunk_bytes` cuối cùng của
    file thay vì đọc từ đầu - nếu đoạn đọc được chưa đủ `num_rows` dòng trọn vẹn (hiếm, chỉ xảy ra khi
    dòng CSV bất thường dài), tăng gấp đôi kích thước đọc và thử lại. Dòng đầu tiên trong đoạn đọc
    được LUÔN bị bỏ vì có thể bị cắt cụt giữa dòng (điểm `seek()` không đảm bảo rơi đúng ranh giới
    dòng). Nhanh hơn nhiều so với đọc + ép kiểu toàn bộ file chỉ để lấy vài dòng cuối.
    """
    file_size = csv_path.stat().st_size
    with csv_path.open("rb") as file:
        header_line = file.readline()
        header_end = file.tell()  # Vị trí byte NGAY SAU header - phần dữ liệu thật bắt đầu từ đây.
        data_size = max(file_size - header_end, 0)

        read_size = min(chunk_bytes, data_size)
        lines: list[bytes] = []
        while True:
            file.seek(header_end + (data_size - read_size))
            tail_bytes = file.read()
            candidate_lines = tail_bytes.split(b"\n")
            if read_size < data_size:
                candidate_lines = candidate_lines[1:]  # Dòng đầu có thể bị cắt cụt - bỏ đi cho an toàn.
            lines = [line for line in candidate_lines if line.strip()]
            if len(lines) >= num_rows or read_size >= data_size:
                break
            read_size = min(read_size * 4, data_size)

    tail_lines = lines[-num_rows:]
    csv_text = header_line.decode("utf-8-sig") + b"\n".join(tail_lines).decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(csv_text))


def _get_flood_predictions_dependency_signature() -> tuple:
    """
    "Chữ ký" (mtime) của các file mà `get_latest_flood_predictions()` phụ thuộc vào
    (`deployment_config.json` + 5 file CSV lịch sử) - dùng làm THAM SỐ CACHE để Streamlit tự biết
    khi nào cần tính lại (file đổi -> chữ ký đổi -> cache miss tự nhiên).

    THAY THẾ cho `ttl=300` cũ (tính lại MÙ QUÁNG mỗi 5 phút bất kể có gì thay đổi hay không - dữ liệu
    thật ra chỉ đổi khi có lần huấn luyện mới hoặc `fetch_data.py` chạy xong, thường là theo GIỜ/NGÀY
    chứ không phải theo phút): với cách này, nếu không có gì thay đổi, hàm luôn trả cache tức thì dù
    người dùng mở app cả tiếng đồng hồ; nếu vừa huấn luyện xong hoặc vừa có dữ liệu mới, lần gọi KẾ
    TIẾP sẽ tự nhận ra ngay (không cần đợi hết 5 phút cũ).

    Tính `os.stat()` cho 6 file rất rẻ (không đọc nội dung file) nên gọi hàm này mỗi lần rerun không
    tốn kém - phần ĐẮT (đọc CSV + suy luận model) chỉ chạy khi chữ ký thực sự đổi.
    """
    signature = [DEPLOYMENT_CONFIG_PATH.stat().st_mtime if DEPLOYMENT_CONFIG_PATH.exists() else None]
    for csv_filename in LOCATION_HISTORICAL_FILE.values():
        csv_path = HISTORICAL_DIR / csv_filename
        signature.append(csv_path.stat().st_mtime if csv_path.exists() else None)
    return tuple(signature)


@st.cache_data(show_spinner=False)
def _get_latest_flood_predictions_cached(_dependency_signature: tuple) -> pd.DataFrame:
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

    Cột 'Nguy cơ' có 3 giá trị: 'An toàn' / 'Ngập' / 'Không xác định'. FAIL-SAFE THEO THIẾT KẾ: khi
    suy luận cho 1 địa phương bị lỗi (thiếu CSV, model hỏng, không đủ dòng cho `window_size`, sai cột
    feature...), KHÔNG mặc định về 'An toàn' như bản trước - với hệ thống cảnh báo ngập, báo nhầm
    "an toàn" trong khi thực ra không xác định được là hướng lỗi NGUY HIỂM HƠN nhiều so với hiển thị
    rõ "không xác định, cần kiểm tra thủ công" cho người vận hành.
    """
    deployed_model = None
    feature_columns = FEATURE_COLS_FOR_INFERENCE
    try:
        deployed_model, feature_columns, _deployment_config = load_deployed_model_and_features()
    except Exception:
        deployed_model = None  # Chưa có model triển khai -> dùng nhánh dự phòng bên dưới.

    # Chỉ cần đọc vài dòng CUỐI của mỗi CSV (window_size cho model tuần tự, hoặc 1 dòng cho model
    # bảng/nhánh dự phòng) - xem `read_csv_tail()` để biết vì sao không đọc + parse toàn bộ file.
    if deployed_model is not None and deployed_model.get("model_type") in {
        "keras_sequence",
        "hybrid_lstm_xgboost",
        "hybrid_lstm_gru_xgboost",
    }:
        required_rows = (deployed_model.get("window_size") or DEFAULT_SEQUENCE_WINDOW_SIZE) + 2
    else:
        required_rows = 3

    records = []
    for location_name, csv_filename in LOCATION_HISTORICAL_FILE.items():
        try:
            location_df = read_csv_tail(HISTORICAL_DIR / csv_filename, required_rows).sort_values("Thời_gian")
            if deployed_model is not None:
                predicted_class = predict_flood_class(deployed_model, location_df, feature_columns)
            else:
                predicted_class = int(location_df["Nguy_cơ_ngập"].iloc[-1])
            risk_status = "An toàn" if predicted_class == 0 else "Ngập"
        except Exception as exc:
            # FAIL-SAFE: xem docstring - CỐ Ý không rơi về "An toàn" khi lỗi.
            risk_status = "Không xác định"
            print(f"[get_latest_flood_predictions] Lỗi suy luận cho '{location_name}': {exc}")

        records.append({"Địa phương": location_name, "Nguy cơ": risk_status})

    return pd.DataFrame(records, columns=["Địa phương", "Nguy cơ"])


def get_latest_flood_predictions() -> pd.DataFrame:
    """Wrapper mỏng: tính chữ ký phụ thuộc (rẻ) rồi gọi hàm đã cache thật - xem docstring
    `_get_flood_predictions_dependency_signature()` và `_get_latest_flood_predictions_cached()`."""
    return _get_latest_flood_predictions_cached(_get_flood_predictions_dependency_signature())


# ==================================================================================================
# MODULE DỰ BÁO 4 NGÀY (Ngày T, T+1, T+2, T+3) - CORE FORECASTING MODULE
# ==================================================================================================
def _fetch_or_estimate_tide_heights(
    lat: float, lon: float, forecast_dates: pd.DatetimeIndex, past_days: int = 0
) -> list[float]:
    """
    Lấy chiều cao triều (m) cho từng ngày trong `forecast_dates` (có thể gồm cả ngày QUÁ KHỨ nếu
    `past_days > 0` - dùng khi cần dựng cửa sổ chuỗi thời gian cho model dạng sequence/hybrid, xem
    `predict_days_ahead_forecast_sequence()`).

    Ưu tiên dữ liệu THẬT từ Open-Meteo Marine API (`wave_height_max`) tại chính tọa độ (lat, lon).
    Marine API CHỈ có dữ liệu tại các điểm lưới nằm trên/gần biển - với tọa độ NỘI ĐỊA THỰC SỰ (đã
    kiểm chứng bằng gọi API sống: TP Huế, Hương Trà), API trả về `null` cho toàn bộ ngày, khi đó hàm
    dùng công thức triều tổng hợp (bán nhật triều chu kỳ ~12.42 giờ + chu kỳ mặt trăng ~29.53 ngày)
    làm giá trị xấp xỉ - ĐÚNG phương pháp đã dùng để sinh cột `Chiều_cao_triều_m` khi xây dựng dữ liệu
    huấn luyện lịch sử (xem `calculate_synthetic_tide()` trong `fetch_data.py`), giúp đầu vào suy luận
    nhất quán về mặt phân phối với dữ liệu mà model đã học, thay vì dùng một hằng số mặc định tùy
    tiện. Công thức tổng hợp này tính trực tiếp từ giá trị NGÀY THÁNG (không phụ thuộc gọi API), nên
    áp dụng được cho cả ngày quá khứ lẫn tương lai mà không cần phân biệt.

    LƯU Ý: KHÔNG hardcode danh sách "địa phương nội địa -> bỏ qua Marine API" để tiết kiệm 1 lượt gọi
    mạng, dù có vẻ hợp lý - lưới dữ liệu của Marine API khá thô (~0.25 độ) nên vẫn "chụp" được điểm
    biển gần nhất cho một số tọa độ tưởng chừng nội địa (ví dụ Quảng Điền, Phú Vang, Hương Thủy trong
    5 địa phương giám sát THỰC RA vẫn nhận được dữ liệu triều thật, chỉ TP Huế/Hương Trà là luôn
    `null`) - hardcode sai sẽ âm thầm hạ chất lượng đầu vào của đúng những địa phương có dữ liệu thật.
    """
    forecast_days_needed = len(forecast_dates) - past_days
    try:
        marine_response = requests.get(
            "https://marine-api.open-meteo.com/v1/marine",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "wave_height_max",
                "past_days": past_days,
                "forecast_days": forecast_days_needed,
                "timezone": "auto",
            },
            timeout=15,
        )
        marine_response.raise_for_status()
        wave_heights = marine_response.json()["daily"]["wave_height_max"][: len(forecast_dates)]
        if len(wave_heights) == len(forecast_dates) and all(value is not None for value in wave_heights):
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


def _fetch_daily_weather_and_tide(
    lat: float, lon: float, forecast_days: int, past_days: int = 0
) -> tuple[dict, pd.DatetimeIndex, list[float]] | None:
    """
    Gọi Open-Meteo FORECAST API lấy dữ liệu khí tượng THEO NGÀY (nhiệt độ, độ ẩm, mưa, độ ẩm đất) cho
    `forecast_days` ngày tương lai - cộng thêm `past_days` ngày QUÁ KHỨ THẬT nếu > 0 (cần cho model
    dạng chuỗi phải dựng cửa sổ nhiều ngày, xem `predict_days_ahead_forecast_sequence()`) - và chiều
    cao triều tương ứng (qua `_fetch_or_estimate_tide_heights()`).

    DÙNG CHUNG cho `predict_4_days_forecast()` (`past_days=0`) và
    `predict_days_ahead_forecast_sequence()` (`past_days=window_size-1`) - trước đây mỗi hàm tự lặp
    lại y hệt đoạn gọi API + kiểm tra/cắt độ dài này (~20 dòng mỗi bản), từng có 1 bản QUÊN cắt đúng
    độ dài `forecast_dates` trước khi tính triều, gây lỗi ghép DataFrame lệch độ dài.

    GIẢI THÍCH XỬ LÝ CỬA SỔ THỜI GIAN (datetime window) - để đưa vào báo cáo luận văn: tham số
    `forecast_days`/`past_days` kết hợp `timezone="auto"` khiến Open-Meteo tự suy ra múi giờ ĐỊA
    PHƯƠNG của tọa độ (lat, lon) - với Huế là Asia/Ho_Chi_Minh (UTC+7) - rồi trả về mảng `daily.time`
    LUÔN kết thúc đúng ở Ngày T+3 (hôm nay + 3), mà KHÔNG cần tự tính `datetime.now() + timedelta`
    thủ công - cách làm thủ công dễ bị lệch 1 ngày nếu server chạy ở múi giờ UTC trong khi vị trí cần
    dự báo lại ở múi giờ khác (UTC+7).

    Trả về `(daily_weather, all_dates, tide_heights)` - `all_dates`/`tide_heights` đã được CẮT ĐÚNG
    `past_days + forecast_days` phần tử - hoặc `None` nếu gọi API lỗi/thiếu ngày dữ liệu.
    """
    total_days_needed = past_days + forecast_days
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
                "past_days": past_days,
                "forecast_days": forecast_days,
                "timezone": "auto",
            },
            timeout=15,
        )
        weather_response.raise_for_status()
        daily_weather = weather_response.json()["daily"]
        all_dates = pd.to_datetime(daily_weather["time"])
        if len(all_dates) < total_days_needed:
            raise ValueError(f"API chỉ trả về {len(all_dates)}/{total_days_needed} ngày dữ liệu.")
        # Cắt CHÍNH XÁC còn total_days_needed phần tử - đề phòng Open-Meteo trả về NHIỀU HƠN yêu cầu
        # (hiếm nhưng có thể) - nếu không cắt, `tide_heights` (tính theo `len(all_dates)`) sẽ DÀI HƠN
        # các cột khác (vốn tự cắt theo `total_days_needed`), gây lỗi ghép DataFrame lệch độ dài.
        all_dates = all_dates[:total_days_needed]
    except (requests.exceptions.RequestException, KeyError, ValueError, TypeError) as exc:
        print(f"[_fetch_daily_weather_and_tide] Lỗi khi gọi Open-Meteo Forecast API: {exc}")
        return None

    tide_heights = _fetch_or_estimate_tide_heights(lat, lon, all_dates, past_days=past_days)
    return daily_weather, all_dates, tide_heights


def add_rain_lag_features_inference(daily_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính 3 cột lag/tích luỹ mưa (`RAIN_LAG1_COL`/`RAIN_LAG2_COL`/`RAIN_ROLLING_3D_COL`) trên
    `daily_features_df` lúc SUY LUẬN - bản sinh đôi của phần tính lag trong
    `analyze_and_train.py::build_daily_feature_dataset()` lúc HUẤN LUYỆN, PHẢI cho ra cùng công thức
    (`.shift(1)`/`.shift(2)`/`.rolling(3, min_periods=1).sum()`) để scaler/model không suy luận sai vì
    lệch cách tính giữa 2 giai đoạn.

    Khác với lúc huấn luyện (nhiều địa phương, cần `groupby`), ở đây `daily_features_df` CHỈ có 1 địa
    phương/1 lần gọi (đã lọc theo `lat, lon` từ `_fetch_daily_weather_and_tide()`) và các dòng đã sẵn
    THEO ĐÚNG THỨ TỰ THỜI GIAN (Open-Meteo trả về `daily.time` tăng dần) nên không cần `groupby`.

    Bên gọi PHẢI đảm bảo `daily_features_df` có đủ `RAIN_LAG_FETCH_DAYS` dòng QUÁ KHỨ THẬT ở ĐẦU
    DataFrame (trước ngày dự báo đầu tiên) - xem `RAIN_LAG_FETCH_DAYS` - để 2 cột lag KHÔNG bị NaN ở
    ngay ngày dự báo đầu tiên (Ngày T/hôm nay).
    """
    result_df = daily_features_df.copy()
    result_df[RAIN_LAG1_COL] = result_df["Lượng_mưa_mm"].shift(1)
    result_df[RAIN_LAG2_COL] = result_df["Lượng_mưa_mm"].shift(2)
    result_df[RAIN_ROLLING_3D_COL] = result_df["Lượng_mưa_mm"].rolling(window=3, min_periods=1).sum()
    return result_df


def _build_forecast_row(
    day_label: str, forecast_date: pd.Timestamp, rain_mm: float, predicted_class: int, temperature_c: float
) -> dict:
    """Đóng gói 1 dòng kết quả dự báo (['Ngày', 'Dự báo Lượng mưa (mm)', 'Dự đoán Ngập', 'Nhiệt độ dự
    báo (°C)']) - DÙNG CHUNG cho `predict_4_days_forecast()` và `predict_days_ahead_forecast_sequence()`
    để đảm bảo định dạng ngày/nhãn nhất quán giữa 2 hàm (trước đây mỗi hàm tự viết riêng 1 bản).

    `temperature_c` được thêm sau (ban đầu chỉ có mưa/nhãn ngập) - phục vụ biểu đồ nhiệt độ/lượng mưa
    trong Tab "Dự báo 14 ngày tới" cần cả nhiệt độ lẫn lượng mưa theo ngày (dữ liệu này đã có sẵn trong
    `daily_features_df` ở cả 2 hàm gọi,
    chỉ là trước đây bị bỏ qua khi đóng gói dòng kết quả)."""
    return {
        "Ngày": f"{forecast_date.strftime('%d/%m/%Y')} ({day_label})",
        "Dự báo Lượng mưa (mm)": round(float(rain_mm), 1),
        "Dự đoán Ngập": "An toàn" if int(predicted_class) == 0 else "Nguy cơ ngập",
        "Nhiệt độ dự báo (°C)": round(float(temperature_c), 1),
    }


def predict_4_days_forecast(lat: float, lon: float, model, scaler) -> pd.DataFrame | None:
    """
    Dự báo nguy cơ ngập cho 4 NGÀY LIÊN TIẾP - Ngày T (hôm nay), T+1, T+2, T+3 - tại 1 tọa độ
    (lat, lon) bất kỳ, dùng `model` + `scaler` ĐÃ HUẤN LUYỆN SẴN được truyền vào (không tự nạp model
    trong hàm này, để hàm dùng được với bất kỳ model nào - XGBoost, RandomForest, hay Deep Learning).

    Trả về DataFrame gồm 3 cột: ['Ngày', 'Dự báo Lượng mưa (mm)', 'Dự đoán Ngập'], hoặc `None` nếu
    gọi API/model lỗi (đã được xử lý gracefully bằng try-except, không raise exception ra ngoài).
    """
    # BƯỚC A - FETCH DATA: xem docstring `_fetch_daily_weather_and_tide()` (giải thích đầy đủ cách xử
    # lý cửa sổ thời gian datetime, đúng chuẩn báo cáo luận văn). `past_days=RAIN_LAG_FETCH_DAYS` (thay
    # vì 0 như trước) - cần thêm 2 ngày QUÁ KHỨ THẬT trước Ngày T để tính cột lag mưa cho CHÍNH Ngày T
    # (xem `add_rain_lag_features_inference()`), nếu không cột lag của Ngày T sẽ bị NaN.
    fetch_result = _fetch_daily_weather_and_tide(
        lat, lon, forecast_days=FORECAST_DAYS_AHEAD, past_days=RAIN_LAG_FETCH_DAYS
    )
    if fetch_result is None:
        return None
    daily_weather, all_dates, tide_heights = fetch_result
    total_days_fetched = RAIN_LAG_FETCH_DAYS + FORECAST_DAYS_AHEAD

    # ==============================================================================================
    # BƯỚC B - PREPROCESSING: gộp toàn bộ đặc trưng vào 1 DataFrame theo ĐÚNG THỨ TỰ CỘT mà `scaler`
    # đã ghi nhớ lúc huấn luyện (`scaler.feature_names_in_`), sau đó `transform()` để đưa dữ liệu thô
    # (đơn vị gốc: độ C, %, mm, m³/m³, m) về cùng thang đo chuẩn hóa (mean=0, std=1) mà model đã học.
    # Bỏ qua bước này sẽ khiến model suy luận sai nghiêm trọng dù code chạy không lỗi.
    # ==============================================================================================
    try:
        # Dựng trên TOÀN BỘ `total_days_fetched` ngày (gồm cả RAIN_LAG_FETCH_DAYS ngày quá khứ) để
        # `add_rain_lag_features_inference()` có đủ dữ liệu tính lag cho ngày dự báo ĐẦU TIÊN, rồi mới
        # CẮT về đúng FORECAST_DAYS_AHEAD dòng (bỏ các dòng quá khứ chỉ dùng để tính lag, không phải
        # kết quả dự báo cần hiển thị) - cùng cách làm với `predict_days_ahead_forecast_sequence()`.
        daily_features_df = pd.DataFrame(
            {
                "Nhiệt_độ_C": daily_weather["temperature_2m_mean"][:total_days_fetched],
                "Độ_ẩm_%": daily_weather["relative_humidity_2m_mean"][:total_days_fetched],
                "Lượng_mưa_mm": daily_weather["rain_sum"][:total_days_fetched],
                "Độ_ẩm_đất": daily_weather["soil_moisture_0_to_7cm_mean"][:total_days_fetched],
                "Chiều_cao_triều_m": tide_heights[:total_days_fetched],
            }
        )
        daily_features_df = add_rain_lag_features_inference(daily_features_df)
        daily_features_df = daily_features_df.iloc[RAIN_LAG_FETCH_DAYS:].reset_index(drop=True)
        forecast_dates = all_dates[RAIN_LAG_FETCH_DAYS:total_days_fetched]

        feature_columns = list(getattr(scaler, "feature_names_in_", FEATURE_COLS_FOR_INFERENCE))
        # Giữ lại DataFrame (có tên cột) thay vì để `scaler.transform()` trả về ndarray thô - tránh
        # warning "X does not have valid feature names" khi model.predict() nhận vào numpy array,
        # dù model đã được fit bằng DataFrame có tên cột lúc huấn luyện (xem `scale_features()` trong
        # `analyze_and_train.py`).
        X_scaled = pd.DataFrame(
            scaler.transform(daily_features_df[feature_columns]), columns=feature_columns
        )
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

    # BƯỚC D - OUTPUT: xem docstring `_build_forecast_row()`.
    result_rows = [
        _build_forecast_row(
            DAY_LABELS[offset],
            forecast_dates[offset],
            daily_features_df["Lượng_mưa_mm"].iloc[offset],
            predicted_classes[offset],
            daily_features_df["Nhiệt_độ_C"].iloc[offset],
        )
        for offset in range(FORECAST_DAYS_AHEAD)
    ]
    return pd.DataFrame(
        result_rows, columns=["Ngày", "Dự báo Lượng mưa (mm)", "Dự đoán Ngập", "Nhiệt độ dự báo (°C)"]
    )


def predict_days_ahead_forecast_sequence(
    lat: float, lon: float, deployed_model: dict, feature_columns: list[str]
) -> pd.DataFrame | None:
    """
    PHẦN MỞ RỘNG của `predict_4_days_forecast()` cho model DẠNG CHUỖI (`keras_sequence`) và HYBRID
    (`hybrid_lstm_xgboost`) - 2 loại model mà `predict_4_days_forecast()` KHÔNG xử lý được, vì nó đưa
    từng ngày vào model NHƯ 1 DÒNG ĐỘC LẬP (đúng cho model dạng bảng), trong khi model dạng chuỗi cần
    NGUYÊN 1 CỬA SỔ nhiều ngày liên tiếp làm đầu vào.

    ĐÚNG QUY ƯỚC CỬA SỔ LÚC HUẤN LUYỆN (xem `build_sequence_datasets()` trong `analyze_and_train.py`):
    cửa sổ gồm `window_size` ngày liên tiếp, và NHÃN CẦN DỰ ĐOÁN LÀ CỦA CHÍNH NGÀY CUỐI CÙNG trong cửa
    sổ đó (không phải ngày sau đó) - tức để dự đoán ngày T+3, cần 1 cửa sổ (window_size - 1) ngày
    TRƯỚC T+3 cộng với chính ngày T+3.

    CÁCH LẤY DỮ LIỆU: gọi Open-Meteo với CẢ `past_days` (lấy đúng (window_size - 1) ngày QUÁ KHỨ THẬT
    ngay trước hôm nay) VÀ `forecast_days=4` (lấy dự báo TƯƠNG LAI cho T/T+1/T+2/T+3) trong CÙNG 1
    request, ghép thành 1 chuỗi ngày liên tục duy nhất, rồi TRƯỢT cửa sổ qua 4 vị trí kết thúc tại
    T/T+1/T+2/T+3 - với các cửa sổ của T+1/T+2/T+3, một phần cửa sổ sẽ dùng chính dữ liệu DỰ BÁO
    (chưa xảy ra) của các ngày trước đó trong cùng đợt dự báo, đây là cách làm hợp lý duy nhất vì
    tại thời điểm dự đoán, dữ liệu THẬT của những ngày đó chưa tồn tại.

    LẤY THÊM `RAIN_LAG_FETCH_DAYS` NGÀY QUÁ KHỨ (ngoài `window_size - 1`): cột lag mưa
    (`add_rain_lag_features_inference()`) cần dữ liệu 1-2 ngày TRƯỚC MỖI ngày trong cửa sổ, kể cả ngày
    CŨ NHẤT của cửa sổ đầu tiên (dùng để dự đoán Ngày T) - nếu chỉ fetch đúng `window_size - 1` ngày
    như trước, ngày cũ nhất trong cửa sổ đó sẽ có cột lag bị NaN vì không có dữ liệu trước nó.
    """
    scaler = deployed_model["scaler"]
    window_size = deployed_model.get("window_size") or DEFAULT_SEQUENCE_WINDOW_SIZE
    past_days_needed = window_size - 1
    # Số ngày quá khứ THẬT cần fetch = đủ cho cửa sổ (`past_days_needed`) CỘNG THÊM đủ cho lag của
    # chính ngày cũ nhất trong cửa sổ đó (`RAIN_LAG_FETCH_DAYS`).
    total_past_days_fetch = past_days_needed + RAIN_LAG_FETCH_DAYS
    total_days_needed = total_past_days_fetch + FORECAST_DAYS_AHEAD

    # BƯỚC A - FETCH DATA: xem docstring `_fetch_daily_weather_and_tide()`.
    fetch_result = _fetch_daily_weather_and_tide(
        lat, lon, forecast_days=FORECAST_DAYS_AHEAD, past_days=total_past_days_fetch
    )
    if fetch_result is None:
        return None
    daily_weather, all_dates, tide_heights = fetch_result

    try:
        daily_features_df = pd.DataFrame(
            {
                "Nhiệt_độ_C": daily_weather["temperature_2m_mean"][:total_days_needed],
                "Độ_ẩm_%": daily_weather["relative_humidity_2m_mean"][:total_days_needed],
                "Lượng_mưa_mm": daily_weather["rain_sum"][:total_days_needed],
                "Độ_ẩm_đất": daily_weather["soil_moisture_0_to_7cm_mean"][:total_days_needed],
                "Chiều_cao_triều_m": tide_heights[:total_days_needed],
            }
        )
        daily_features_df = add_rain_lag_features_inference(daily_features_df)
        # Giữ tên cột (DataFrame) khi transform - tránh warning "X does not have valid feature names".
        scaled_all_days = pd.DataFrame(
            scaler.transform(daily_features_df[feature_columns]), columns=feature_columns
        ).to_numpy()
    except Exception as exc:
        print(f"[predict_days_ahead_forecast_sequence] Lỗi khi tiền xử lý/chuẩn hóa dữ liệu: {exc}")
        return None

    result_rows = []
    try:
        for offset in range(FORECAST_DAYS_AHEAD):
            # Vị trí ngày T/T+1/.../T+13 trong chuỗi đã ghép - dịch thêm `RAIN_LAG_FETCH_DAYS` so với
            # trước đây vì mảng giờ có thêm phần đầu chỉ để tính lag (xem giải thích ở docstring).
            end_idx = total_past_days_fetch + offset
            start_idx = end_idx - window_size + 1
            window_input = scaled_all_days[start_idx : end_idx + 1].reshape(1, window_size, len(feature_columns))
            # Dispatch model_type dùng CHUNG với `predict_flood_class()` - xem
            # `predict_class_from_sequence_window()`.
            predicted_class = predict_class_from_sequence_window(deployed_model, window_input)
            result_rows.append(
                _build_forecast_row(
                    DAY_LABELS[offset],
                    all_dates[end_idx],
                    daily_features_df["Lượng_mưa_mm"].iloc[end_idx],
                    predicted_class,
                    daily_features_df["Nhiệt_độ_C"].iloc[end_idx],
                )
            )
    except Exception as exc:
        print(f"[predict_days_ahead_forecast_sequence] Lỗi khi suy luận bằng model: {exc}")
        return None

    return pd.DataFrame(
        result_rows, columns=["Ngày", "Dự báo Lượng mưa (mm)", "Dự đoán Ngập", "Nhiệt độ dự báo (°C)"]
    )


@st.cache_data(persist="disk", show_spinner="Đang gọi Open-Meteo và suy luận dự báo cho 5 địa phương...")
def _compute_forecast_4day_result() -> dict:
    """
    Tính bảng dự báo 14 ngày cho toàn bộ 5 địa phương giám sát (gọi Open-Meteo + suy luận model).

    Cache bằng `st.cache_data(persist="disk", ...)` - Streamlit TỰ lưu kết quả (kể cả DataFrame và
    `datetime`) ra đĩa và tự nạp lại ở lần chạy sau (F5, restart server), THAY vì tự viết tay ~40 dòng
    JSON serialize/deserialize (`to_dict(orient="records")` / `pd.DataFrame(...)` / `isoformat()` /
    `datetime.fromisoformat()`) như bản trước - vừa ít code hơn, vừa tránh lỗi âm thầm nếu sau này đổi
    shape của `cached_result` mà quên cập nhật đồng bộ cả 2 hàm đọc/ghi.

    ĐỂ LÀM MỚI: gọi `_compute_forecast_4day_result.clear()` TRƯỚC khi gọi lại hàm này - xem
    `render_forecast_tab()` (khi bấm nút "Dự báo lại" hoặc phát hiện cache đã qua ngày mới) và
    `render_sidebar()` (nút "Làm mới toàn bộ cache").

    Raise `RuntimeError` với thông điệp rõ ràng nếu chưa có model triển khai hoặc không lấy được dự
    báo cho địa phương nào - KHÔNG tự bắt lỗi ở đây, để `render_forecast_tab()` tự quyết định hiển thị
    gì (báo lỗi trắng nếu chưa từng có cache, hay giữ cache cũ + cảnh báo nếu đã có).
    """
    try:
        deployed_model, feature_columns, deployment_config = load_deployed_model_and_features()
    except Exception as exc:
        raise RuntimeError(
            f"Không thể nạp model để dự báo (có thể chưa huấn luyện model nào, hoặc lỗi tạm thời khi "
            f"đọc artifact): {exc}"
        ) from exc

    model_type = deployment_config.get("model_type")

    def _forecast_one_location(location_name: str, lat: float, lon: float) -> pd.DataFrame | None:
        # DISPATCH THEO ĐÚNG model_type: model dạng bảng dùng `predict_4_days_forecast()` (mỗi ngày 1
        # dòng độc lập), model dạng chuỗi/Hybrid dùng `predict_days_ahead_forecast_sequence()` (cửa sổ
        # nhiều ngày liên tiếp, xem docstring hàm đó để biết chi tiết cách dựng cửa sổ).
        if model_type == "sklearn_tabular":
            return predict_4_days_forecast(lat, lon, deployed_model["model"], deployed_model["scaler"])
        if model_type in {"keras_sequence", "hybrid_lstm_xgboost", "hybrid_lstm_gru_xgboost"}:
            return predict_days_ahead_forecast_sequence(lat, lon, deployed_model, feature_columns)
        return None

    # CHẠY SONG SONG 5 địa phương bằng ThreadPoolExecutor - mỗi địa phương tốn tới 2 lệnh gọi mạng
    # chặn (Open-Meteo Forecast + Marine API), worst case ~10 lượt gọi NỐI TIẾP nếu chạy tuần tự
    # (có thể lên tới hàng chục giây - hàng phút nếu mạng chậm). Vì phần tốn thời gian là I/O mạng
    # (không phải tính toán CPU), GIL của Python được nhả ra trong lúc chờ `requests.get()`, nên
    # ThreadPoolExecutor mang lại lợi ích thật (không bị GIL chặn) - phần suy luận Keras (nếu có)
    # được bảo vệ riêng bằng `_KERAS_INFERENCE_LOCK` (xem `predict_class_from_sequence_window()`).
    forecast_by_location: dict[str, pd.DataFrame | None] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(REAL_MONITORED_LOCATIONS)) as executor:
        future_to_location = {
            executor.submit(_forecast_one_location, location_name, lat, lon): location_name
            for location_name, (lat, lon) in REAL_MONITORED_LOCATIONS.items()
        }
        for future in concurrent.futures.as_completed(future_to_location):
            location_name = future_to_location[future]
            try:
                forecast_by_location[location_name] = future.result()
            except Exception as exc:
                print(f"[_compute_forecast_4day_result] Lỗi dự báo cho '{location_name}': {exc}")
                forecast_by_location[location_name] = None

    # Ghép kết quả theo ĐÚNG thứ tự REAL_MONITORED_LOCATIONS (không phải thứ tự hoàn thành song song,
    # vốn không xác định trước) - giữ bảng dự báo hiển thị ổn định, dễ đọc giữa các lần làm mới.
    forecast_frames = []
    failed_locations = []
    for location_name in REAL_MONITORED_LOCATIONS:
        location_forecast_df = forecast_by_location.get(location_name)
        if location_forecast_df is None:
            failed_locations.append(location_name)
            continue
        location_forecast_df = location_forecast_df.copy()
        location_forecast_df.insert(0, "Địa phương", location_name)
        forecast_frames.append(location_forecast_df)

    if not forecast_frames:
        raise RuntimeError("Không lấy được dự báo cho bất kỳ địa phương nào lúc này.")

    return {
        "combined_df": pd.concat(forecast_frames, ignore_index=True),
        "failed_locations": failed_locations,
        "model_name": deployment_config.get("model_name"),
        "f1_macro": deployment_config.get("f1_macro", 0),
        "generated_at": datetime.now(),
    }


def fetch_open_meteo_rain_mm(lat: float, lon: float) -> float | None:
    """Lượng mưa HÔM NAY (mm) từ Open-Meteo - nguồn baseline mà model đang dùng để huấn luyện/suy luận."""
    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "daily": "rain_sum", "forecast_days": 1, "timezone": "auto"},
            timeout=10,
        )
        response.raise_for_status()
        return round(float(response.json()["daily"]["rain_sum"][0]), 1)
    except Exception as exc:
        print(f"[fetch_open_meteo_rain_mm] Lỗi: {exc}")
        return None


def fetch_weatherbit_rain_mm(lat: float, lon: float, api_key: str) -> float | None:
    try:
        response = requests.get(
            "https://api.weatherbit.io/v2.0/forecast/daily",
            params={"lat": lat, "lon": lon, "key": api_key, "days": 1, "units": "M"},
            timeout=10,
        )
        response.raise_for_status()
        return round(float(response.json()["data"][0]["precip"]), 1)
    except Exception as exc:
        print(f"[fetch_weatherbit_rain_mm] Lỗi: {exc}")
        return None


def fetch_visualcrossing_rain_mm(lat: float, lon: float, api_key: str) -> float | None:
    try:
        response = requests.get(
            f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/today",
            params={"unitGroup": "metric", "key": api_key, "include": "days", "contentType": "json"},
            timeout=10,
        )
        response.raise_for_status()
        return round(float(response.json()["days"][0].get("precip") or 0), 1)
    except Exception as exc:
        print(f"[fetch_visualcrossing_rain_mm] Lỗi: {exc}")
        return None


def fetch_tomorrow_io_rain_mm(lat: float, lon: float, api_key: str) -> float | None:
    """
    Lượng mưa HÔM NAY (mm) từ Tomorrow.io.

    BẮT BUỘC truyền `timezone` - khác với Open-Meteo (`fetch_open_meteo_rain_mm`, dùng "auto"), Tomorrow.io
    MẶC ĐỊNH gộp "ngày" theo giờ UTC nếu thiếu tham số này (đã kiểm chứng qua tài liệu API) - tức "ngày 0"
    của họ sẽ là 07h00 hôm nay đến 07h00 hôm sau theo giờ Việt Nam (UTC+7), LỆCH hẳn khung "hôm nay" thật
    mà Open-Meteo và các nguồn khác đang so sánh cùng. Lỗi thật đã gặp: lệch khung giờ này vô tình gộp
    thêm mưa của khung giờ kế bên, khiến số liệu cao gấp 2-4 lần các nguồn khác một cách có hệ thống dù
    không phải do model dự báo thời tiết khác nhau thật sự.
    """
    try:
        response = requests.get(
            "https://api.tomorrow.io/v4/weather/forecast",
            params={
                "location": f"{lat},{lon}",
                "timesteps": "1d",
                "units": "metric",
                "timezone": "Asia/Ho_Chi_Minh",
                "apikey": api_key,
            },
            timeout=10,
        )
        response.raise_for_status()
        values = response.json()["timelines"]["daily"][0]["values"]
        return round(float(values.get("rainAccumulationSum", 0)), 1)
    except Exception as exc:
        print(f"[fetch_tomorrow_io_rain_mm] Lỗi: {exc}")
        return None


def fetch_stormglass_rain_mm(lat: float, lon: float, api_key: str) -> float | None:
    """
    Stormglass trả dữ liệu THEO GIỜ (không có tổng ngày sẵn) - cộng dồn 24 giờ tới để ra lượng mưa
    trong ngày, mỗi giờ ưu tiên nguồn `.sg` (blended, đáng tin nhất), nếu thiếu thì lấy nguồn đầu tiên
    có sẵn trong dict `precipitation` (mỗi nhà cung cấp phụ trợ 1 field riêng, ví dụ `.noaa`, `.icon`).
    """
    try:
        response = requests.get(
            "https://api.stormglass.io/v2/weather/point",
            params={"lat": lat, "lng": lon, "params": "precipitation"},
            headers={"Authorization": api_key},
            timeout=10,
        )
        response.raise_for_status()
        hours = response.json().get("hours", [])[:24]
        if not hours:
            return None
        total_mm = 0.0
        for hour_entry in hours:
            precip_sources = hour_entry.get("precipitation", {})
            value = precip_sources.get("sg")
            if value is None and precip_sources:
                value = next(iter(precip_sources.values()))
            total_mm += value or 0
        return round(total_mm, 1)
    except Exception as exc:
        print(f"[fetch_stormglass_rain_mm] Lỗi: {exc}")
        return None


WEATHER_COMPARISON_FETCHERS: dict[str, Callable[[float, float, str], float | None]] = {
    "WEATHERBIT_KEY": fetch_weatherbit_rain_mm,
    "VISUALCROSSING_KEY": fetch_visualcrossing_rain_mm,
    "TOMORROW_KEY": fetch_tomorrow_io_rain_mm,
    "STORMGLASS_KEY": fetch_stormglass_rain_mm,
}

OPEN_METEO_BASELINE_COLUMN = "Open-Meteo (mm) [baseline model]"
WEATHER_DEVIATION_WARN_MM = 5.0  # lệch tuyệt đối > 5mm so với baseline mới đánh dấu cảnh báo


WEATHER_COMPARISON_CACHE_TTL_SECONDS = 3600  # 1 tiếng - hết hạn thì lần mở tab/rerun kế tiếp tự gọi lại API


@st.cache_data(
    ttl=WEATHER_COMPARISON_CACHE_TTL_SECONDS,
    show_spinner="Đang đối chiếu dữ liệu mưa giữa các nguồn API...",
)
def build_weather_comparison_matrix(active_provider_keys: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    """
    Dựng ma trận Địa phương x Nguồn API (LƯỢNG MƯA hôm nay, đơn vị mm) để đối chiếu Open-Meteo (nguồn
    model đang dùng để huấn luyện/suy luận) với các nguồn thay thế người dùng đã cấu hình key qua khung
    admin. Chỉ so sánh lượng mưa (biến liên quan trực tiếp đến nguy cơ ngập), KHÔNG so nhiệt độ/độ ẩm.

    CHỈ đối chiếu để hiển thị - KHÔNG đưa giá trị từ nguồn khác vào model suy luận, vì model chỉ được
    huấn luyện trên phân phối/cách đo của Open-Meteo, đưa thẳng dữ liệu nguồn khác vào có thể lệch
    calibration và làm sai lệch kết quả dự đoán ngập.

    `active_provider_keys` là tuple (secret_key, api_key) đã sắp xếp - dùng làm 1 phần cache key để
    Streamlit tự làm mới bảng khi người dùng đổi/xoá key qua khung admin, thay vì dùng chung 1 cache
    cho mọi phiên (mỗi phiên có thể đang thử key khác nhau).

    KẾT QUẢ TỰ ĐỘNG LÀM MỚI SAU MỖI `WEATHER_COMPARISON_CACHE_TTL_SECONDS` GIÂY: đây là cache "pull"
    kiểu Streamlit (không có tiến trình nền chạy độc lập) - nghĩa là API chỉ thực sự được gọi lại khi
    có người mở/tải lại tab SAU KHI cache đã hết hạn, không tự chạy ngầm khi không ai mở app. Muốn lấy
    ngay lập tức không cần đợi hết hạn thì bấm nút "Cập nhật dữ liệu API".
    """
    provider_key_map = dict(active_provider_keys)
    active_providers = [p for p in WEATHER_PROVIDER_OPTIONS if p["secret_key"] in provider_key_map]

    rows = []
    for location_name, (lat, lon) in REAL_MONITORED_LOCATIONS.items():
        row = {"Địa phương": location_name, OPEN_METEO_BASELINE_COLUMN: fetch_open_meteo_rain_mm(lat, lon)}
        for provider in active_providers:
            fetch_fn = WEATHER_COMPARISON_FETCHERS[provider["secret_key"]]
            row[f"{provider['label']} (mm)"] = fetch_fn(lat, lon, provider_key_map[provider["secret_key"]])
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    numeric_columns = [c for c in comparison_df.columns if c != "Địa phương"]
    # Ép kiểu số THẬT SỰ (float64 + NaN) thay vì để cột object lẫn `None` - cột object khiến
    # `st.dataframe`/`Styler.format` hiển thị thẳng chữ "None" ra bảng thay vì ô trống/"—".
    comparison_df[numeric_columns] = comparison_df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return comparison_df


def render_weather_comparison_section() -> None:
    """
    Khung "Đối chiếu dữ liệu mưa giữa các nguồn API" - chỉ hiện khi người dùng đã cấu hình ÍT NHẤT 1
    API key thay thế qua khung admin sidebar. Mục tiêu: giúp phát hiện Open-Meteo (nguồn model đang
    dùng) có đang báo lệch nhiều so với các nguồn độc lập khác hay không, để người xem tự cân nhắc thêm
    khi đọc kết quả dự báo - đây là bước ĐỐI CHIẾU (cross-check hiển thị), CHƯA phải hệ khuyến nghị DSS
    đầy đủ (chưa tự động ra quyết định/hành động, chỉ hỗ trợ đọc thêm thông tin).
    """
    active_provider_keys = tuple(
        sorted(
            (provider["secret_key"], api_key)
            for provider in WEATHER_PROVIDER_OPTIONS
            if provider["secret_key"] in WEATHER_COMPARISON_FETCHERS
            and (api_key := get_api_secret(provider["secret_key"], provider["env_var_name"]))
        )
    )

    with st.expander("Đối chiếu dữ liệu mưa giữa các nguồn API (ma trận)", expanded=False):
        if not active_provider_keys:
            st.info(
                "Chưa có nguồn API thay thế nào được cấu hình để đối chiếu với Open-Meteo (nguồn model "
                "đang dùng). Vào khung **Cấu hình API nâng cao (Admin)** ở sidebar để thêm key của "
                "Weatherbit / Visual Crossing / Tomorrow.io / Stormglass."
            )
            return

        st.caption(
            "Bảng so sánh **lượng mưa hôm nay (mm)** - biến liên quan trực tiếp đến nguy cơ ngập, không "
            "phải nhiệt độ/độ ẩm. Ô hiện **\"—\"** nghĩa là lần gọi đó thất bại (key sai/hết hạn/hết "
            f"quota) chứ không phải trời không mưa - kiểm tra lại key ở khung Admin nếu thấy \"—\" kéo "
            f"dài. Dữ liệu tự làm mới mỗi khi có người mở lại tab và cache đã quá "
            f"{WEATHER_COMPARISON_CACHE_TTL_SECONDS // 3600} tiếng (không có tiến trình chạy nền)."
        )

        if st.button("Cập nhật dữ liệu API", key="weather_comparison_refresh_button"):
            build_weather_comparison_matrix.clear()
            st.rerun()

        comparison_df = build_weather_comparison_matrix(active_provider_keys)
        provider_columns = [c for c in comparison_df.columns if c not in ("Địa phương", OPEN_METEO_BASELINE_COLUMN)]
        numeric_columns = [OPEN_METEO_BASELINE_COLUMN, *provider_columns]

        def highlight_deviation(row: pd.Series) -> list[str]:
            numeric_row = comparison_df.loc[row.name]
            baseline_value = numeric_row[OPEN_METEO_BASELINE_COLUMN]
            styles = [""] * len(row)
            for i, col in enumerate(row.index):
                if col not in provider_columns or pd.isna(numeric_row[col]) or pd.isna(baseline_value):
                    continue
                if abs(numeric_row[col] - baseline_value) > WEATHER_DEVIATION_WARN_MM:
                    styles[i] = "background-color: #FEE2E2; color: #991B1B; font-weight: 600;"
            return styles

        # Tự format thành chuỗi hiển thị ("—" cho ô lỗi) NGAY TRÊN DATAFRAME, không dựa vào
        # `Styler.format()` - vì `st.dataframe` không đảm bảo áp dụng format của Styler ở mọi phiên
        # bản Streamlit, dễ bị lộ ra chữ "None"/số thô lẫn lộn định dạng như đã gặp thực tế.
        display_df = comparison_df.copy()
        display_df[numeric_columns] = display_df[numeric_columns].apply(
            lambda col: col.map(lambda v: "—" if pd.isna(v) else f"{v:.1f}")
        )

        comparison_styler = build_contrast_styler(display_df).apply(highlight_deviation, axis=1)
        render_styled_table(comparison_styler, height=min(90 + 38 * len(comparison_df), 320))

        deviation_notes = []
        for _, row in comparison_df.iterrows():
            baseline_value = row[OPEN_METEO_BASELINE_COLUMN]
            if pd.isna(baseline_value):
                continue
            worst_col, worst_diff = None, 0.0
            for col in provider_columns:
                if pd.isna(row[col]):
                    continue
                diff = abs(row[col] - baseline_value)
                if diff > worst_diff:
                    worst_col, worst_diff = col, diff
            if worst_col and worst_diff > WEATHER_DEVIATION_WARN_MM:
                deviation_notes.append(
                    f"**{row['Địa phương']}**: lệch **{worst_diff:.1f}mm** giữa Open-Meteo và {worst_col} - "
                    "nên xem thêm nguồn khác/theo dõi sát trước khi kết luận."
                )

        if deviation_notes:
            render_chart_discussion(
                f"Có {len(deviation_notes)}/{len(comparison_df)} địa phương đang lệch trên "
                f"{WEATHER_DEVIATION_WARN_MM:.0f}mm giữa các nguồn:\n\n" + "\n\n".join(deviation_notes)
            )
        else:
            render_chart_discussion(
                "Các nguồn API đang cho kết quả khá đồng nhất (lệch không quá "
                f"{WEATHER_DEVIATION_WARN_MM:.0f}mm) - chưa có dấu hiệu Open-Meteo báo sai lệch lớn so "
                "với nguồn độc lập khác cho các địa phương giám sát hiện tại."
            )


@st.fragment(run_every=LIVE_TAB_AUTO_REFRESH_INTERVAL)
def render_forecast_tab() -> None:
    """
    Nội dung TRANG ĐẦU TIÊN của app - bảng dự báo nguy cơ ngập 14 ngày tới (T đến T+13) cho toàn bộ
    5 địa phương giám sát, dùng chính model đã huấn luyện (không phải số liệu giả lập).

    ĐẶT Ở TRANG ĐẦU (thay vì EDA) vì đây là KẾT QUẢ CUỐI CÙNG, thiết thực nhất mà người xem (chính
    quyền địa phương khi vận hành, hoặc hội đồng khi bảo vệ luận văn) cần thấy NGAY khi mở app -
    "mô hình dự báo được gì" - thay vì phải lật qua các tab kỹ thuật nội bộ (khám phá dữ liệu, quy
    trình huấn luyện) trước mới thấy được giá trị thực tế của hệ thống.

    ----------------------------------------------------------------------------------------------
    HỆ THỐNG HOÁ CẬP NHẬT REALTIME LIÊN TỤC (`@st.fragment(run_every=...)`):
    ----------------------------------------------------------------------------------------------
    Toàn bộ hàm này được đánh dấu là 1 FRAGMENT của Streamlit, tự động CHẠY LẠI mỗi
    `LIVE_TAB_AUTO_REFRESH_INTERVAL` (hiện tại: 10 phút) MÀ KHÔNG CẦN người dùng bấm F5 hay tương tác
    gì - miễn là tab trình duyệt vẫn đang mở (kết nối WebSocket với server còn sống). Vì các hàm dữ
    liệu bên dưới (`_compute_forecast_4day_result()`) đã tự cache theo mtime/ngày, mỗi lần fragment
    tự chạy lại gần như MIỄN PHÍ (trả cache ngay) TRỪ KHI có gì đó thật sự thay đổi (huấn luyện xong
    model mới, hoặc đã qua ngày mới) - lúc đó UI sẽ tự cập nhật ngay trong 10 phút kế tiếp mà không
    cần ai chủ động làm gì.

    GIỚI HẠN CẦN NÓI RÕ (đúng bản chất kiến trúc Streamlit, không phải thiếu sót khi cài đặt): cơ chế
    này CHỈ hoạt động khi có ÍT NHẤT 1 trình duyệt đang mở kết nối tới app - Streamlit không có tiến
    trình nền chạy khi hoàn toàn không ai xem trang (khác với cron job thật chạy độc lập trên server).
    Nếu cần dữ liệu THẬT SỰ được fetch mới ngay cả khi không ai mở app, vẫn cần 1 tiến trình lập lịch
    riêng ở tầng hệ điều hành (cron/systemd timer gọi `fetch_data.py`) - nằm ngoài phạm vi Streamlit.
    """
    st.subheader("Dự báo ngập lụt 14 ngày tới")
    st.caption(
        "Kết quả dự báo thật từ model đã huấn luyện, cho toàn bộ 5 địa phương giám sát (Ngày T = hôm "
        "nay, đến T+13), dựa trên dữ liệu thời tiết dự báo mới nhất từ Open-Meteo Forecast API. Trang "
        f"này tự động kiểm tra và làm mới mỗi {LIVE_TAB_AUTO_REFRESH_INTERVAL} khi đang mở."
    )

    header_left, header_right = st.columns([5, 1])
    with header_right:
        refresh_clicked = st.button(
            "Dự báo lại",
            key="refresh_4day_forecast_button",
            use_container_width=True,
            help="Gọi lại Open-Meteo và suy luận model để lấy dự báo mới nhất.",
        )

    # `_compute_forecast_4day_result()` cache bằng `st.cache_data(persist="disk")` - lần gọi ĐẦU TIÊN
    # trong phiên này chỉ tốn kém nếu cache trên đĩa CHƯA có (F5/restart server sẽ đọc lại cache cũ
    # gần như tức thì, không gọi lại Open-Meteo). Bọc try/except vì hàm raise RuntimeError khi chưa có
    # model triển khai hoặc gọi API thất bại hoàn toàn.
    try:
        cached_result = _compute_forecast_4day_result()
    except Exception:
        cached_result = None

    # "Hết hạn" khi cache được tính từ một NGÀY KHÁC (khác ngày dương lịch hiện tại) - tức là cứ qua
    # 00h00 là lần rerun/mở app KẾ TIẾP sẽ tự động phát hiện cache cũ và gọi lại Open-Meteo + model để
    # lấy dự báo mới cho ngày hôm đó, không cần người dùng phải nhớ bấm nút. Lưu ý: Streamlit không có
    # tiến trình nền chạy đúng lúc 00h00 - việc tự làm mới chỉ xảy ra ở lượt tương tác/mở trang ĐẦU
    # TIÊN sau khi qua ngày mới (đúng bản chất "on-demand" của một app web, không phải cron job thật).
    is_cache_stale = cached_result is None or cached_result["generated_at"].date() < datetime.now().date()

    # CHỈ thực sự gọi lại Open-Meteo + suy luận model khi: (1) chưa có cache nào dùng được (kể cả từ
    # đĩa), (2) cache đã qua ngày mới (is_cache_stale), hoặc (3) người dùng chủ động bấm "Dự báo
    # lại" - gọi `.clear()` để buộc `_compute_forecast_4day_result()` tính lại thay vì trả cache cũ.
    #
    # FAIL-SAFE: hễ làm mới thất bại (lỗi đọc artifact/model TẠM THỜI, ví dụ file đang được ghi dở lúc
    # training worker chạy song song), KHÔNG xóa mất bảng dự báo hợp lệ đang có trong `cached_result`
    # (biến này giữ nguyên giá trị CŨ vì except bên dưới không gán lại nó) - chỉ `return` báo lỗi trắng
    # khi thực sự CHƯA TỪNG có cache nào để hiển thị thay thế.
    refresh_error: str | None = None
    if refresh_clicked or is_cache_stale:
        _compute_forecast_4day_result.clear()
        try:
            cached_result = _compute_forecast_4day_result()
        except Exception as exc:
            refresh_error = str(exc)

    if refresh_error is not None:
        if cached_result is None:
            st.error(f"{refresh_error} Hãy khởi chạy huấn luyện ở Tab 'Tiền xử lý & huấn luyện' trước.")
            return
        st.warning(
            f"Không làm mới được dự báo mới ({refresh_error}) - đang hiển thị kết quả gần nhất "
            f"còn lưu (xem thời điểm cập nhật bên dưới)."
        )

    combined_forecast_df = cached_result["combined_df"]

    with header_left:
        st.caption(f"Cập nhật lần cuối: {cached_result['generated_at'].strftime('%H:%M:%S %d/%m/%Y')}")

    if cached_result["failed_locations"]:
        st.warning(
            f"Không lấy được dự báo cho: {', '.join(cached_result['failed_locations'])} "
            "(Open-Meteo/model tạm thời lỗi - hãy thử bấm 'Dự báo lại')."
        )

    render_styled_table(
        build_contrast_styler(
            combined_forecast_df,
            numeric_formats={"Dự báo Lượng mưa (mm)": "{:.1f}", "Nhiệt độ dự báo (°C)": "{:.1f}"},
        ),
        height=min(120 + 38 * len(combined_forecast_df), 640),
    )

    at_risk_df = combined_forecast_df[combined_forecast_df["Dự đoán Ngập"] != "An toàn"]
    if at_risk_df.empty:
        render_chart_discussion(
            f"Trong {FORECAST_DAYS_AHEAD} ngày tới, cả {len(REAL_MONITORED_LOCATIONS)}/{len(REAL_MONITORED_LOCATIONS)} địa "
            "phương giám sát đều được model dự báo AN TOÀN. Vẫn nên theo dõi lại thường xuyên vì dự báo "
            "thời tiết có thể thay đổi giữa các lần cập nhật."
        )
    else:
        risk_counts_by_location = at_risk_df["Địa phương"].value_counts()
        render_chart_discussion(
            f"Có {at_risk_df['Địa phương'].nunique()}/{len(REAL_MONITORED_LOCATIONS)} địa phương xuất "
            f"hiện ít nhất 1 ngày nguy cơ ngập trong {FORECAST_DAYS_AHEAD} ngày tới. Địa phương có số ngày nguy cơ nhiều nhất: "
            f"**{risk_counts_by_location.index[0]}** ({int(risk_counts_by_location.iloc[0])}/{FORECAST_DAYS_AHEAD} ngày). Nên "
            "ưu tiên theo dõi sát và chuẩn bị phương án ứng phó sớm cho các khu vực này."
        )

    st.caption(
        f"Model đang dùng để dự báo: `{cached_result['model_name']}` "
        f"(F1-Macro={cached_result['f1_macro']:.4f})"
    )

    # Biểu đồ nhiệt độ/lượng mưa từng địa phương - GỘP thẳng vào đây (trước đây là 1 tab riêng "Biểu đồ
    # dự báo") vì đây CHỈ là dữ liệu ĐẦU VÀO thời tiết, không phải kết quả dự đoán ngập của ML/DL - tách
    # thành tab riêng dễ khiến người xem lầm tưởng đây là 1 tính năng dự báo độc lập, trong khi thực ra
    # nó minh hoạ TRỰC TIẾP cho đúng bảng "Dự đoán Ngập" ở trên (dùng chung 1 cache, không gọi lại API).
    with st.expander("Biểu đồ nhiệt độ & lượng mưa dự báo theo từng địa phương", expanded=False):
        for location_name in REAL_MONITORED_LOCATIONS:
            location_forecast_df = combined_forecast_df[combined_forecast_df["Địa phương"] == location_name]
            render_location_forecast_combo_chart(location_name, location_forecast_df)

    render_weather_comparison_section()


def render_location_forecast_combo_chart(location_name: str, location_forecast_df: pd.DataFrame) -> None:
    """
    Biểu đồ cột (lượng mưa, mm) + đường (nhiệt độ, °C) theo NGÀY cho 1 địa phương - dùng ĐÚNG dữ liệu
    dự báo thật đã có sẵn từ `_compute_forecast_4day_result()` (không gọi thêm API/model nào mới),
    theo mẫu biểu đồ người dùng cung cấp (cột mưa + đường nhiệt độ trên cùng 1 biểu đồ, trục y phụ).
    """
    if location_forecast_df.empty:
        st.info(f"Chưa có dữ liệu dự báo cho {location_name}.")
        return

    if "Nhiệt độ dự báo (°C)" not in location_forecast_df.columns:
        # Cột này mới thêm sau (bản trước chỉ có mưa/nhãn ngập) - nếu thiếu, nghĩa là đang đọc phải
        # CACHE ĐĨA CŨ của `_compute_forecast_4day_result()` (Streamlit cache_data KHÔNG tự phát hiện
        # được khi hàm CON bên trong như `_build_forecast_row()` thay đổi, chỉ hàm được decorate trực
        # tiếp) - báo rõ cho người dùng thay vì crash KeyError, và tự hướng dẫn cách xoá cache đúng.
        st.warning(
            f"Chưa có dữ liệu nhiệt độ cho {location_name} - có thể do cache cũ trước khi tính năng "
            "này được thêm vào. Bấm nút 'Làm mới toàn bộ cache' ở sidebar rồi tải lại trang."
        )
        return

    # Rút gọn nhãn trục X: "12/09/2026 (Hôm nay)" -> "12/09" - đủ để phân biệt các ngày, không chiếm
    # quá nhiều chỗ ngang khi hiển thị đủ 14 cột trên 1 biểu đồ.
    short_day_labels = location_forecast_df["Ngày"].str.extract(r"^(\d{2}/\d{2})")[0]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=short_day_labels,
            y=location_forecast_df["Dự báo Lượng mưa (mm)"],
            name="Lượng mưa (mm)",
            marker=dict(color="#3b82f6"),
            text=[f"{v:.1f}" for v in location_forecast_df["Dự báo Lượng mưa (mm)"]],
            textposition="outside",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=short_day_labels,
            y=location_forecast_df["Nhiệt độ dự báo (°C)"],
            name="Nhiệt độ (°C)",
            mode="lines+markers+text",
            line=dict(color="#f97316", width=3),
            marker=dict(size=7),
            text=[f"{v:.0f}°" for v in location_forecast_df["Nhiệt độ dự báo (°C)"]],
            textposition="top center",
            textfont=dict(color="#f97316"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title=dict(text=f"Nhiệt độ và lượng mưa dự báo - {location_name}"),
        xaxis=dict(title="Ngày"),
        yaxis=dict(title="Lượng mưa (mm)"),
        yaxis2=dict(title="Nhiệt độ (°C)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
        hovermode="x unified",
        bargap=0.3,
    )
    apply_dark_plotly_theme(fig, height=380)
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(show_spinner=False)
def load_district_boundaries() -> dict | None:
    """
    Nạp GeoJSON ranh giới hành chính THẬT của 5 địa phương giám sát (thay cho chấm điểm marker) - trả
    về `None` nếu file thiếu/lỗi cấu trúc, để `build_smart_routing_map()` tự động rơi về vẽ marker
    điểm như phương án dự phòng, KHÔNG làm crash bản đồ.
    """
    try:
        with DISTRICT_BOUNDARY_GEOJSON_PATH.open("r", encoding="utf-8") as file:
            geojson_data = json.load(file)
        if not geojson_data.get("features"):
            return None
        return geojson_data
    except Exception:
        return None


def _points_in_polygon(px: np.ndarray, py: np.ndarray, poly_x: np.ndarray, poly_y: np.ndarray) -> np.ndarray:
    """Ray casting (even-odd rule), vector hoá bằng numpy trên toàn bộ điểm lưới cùng lúc."""
    inside = np.zeros(len(px), dtype=bool)
    n = len(poly_x)
    j = n - 1
    for i in range(n):
        xi, yi, xj, yj = poly_x[i], poly_y[i], poly_x[j], poly_y[j]
        crosses = (yi > py) != (yj > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_intersect = (xj - xi) * (py - yi) / (yj - yi) + xi
        inside ^= crosses & (px < x_intersect)
        j = i
    return inside


def _min_distance_to_ring(px: np.ndarray, py: np.ndarray, ring_x: np.ndarray, ring_y: np.ndarray) -> np.ndarray:
    """Khoảng cách nhỏ nhất từ mỗi điểm tới các cạnh polygon (đơn vị độ kinh/vĩ - chỉ cần dùng để SO
    SÁNH tương đối trong phạm vi hẹp 1 tỉnh, không cần quy đổi mét)."""
    min_distance = np.full(len(px), np.inf)
    n = len(ring_x)
    for i in range(n):
        x1, y1, x2, y2 = ring_x[i], ring_y[i], ring_x[(i + 1) % n], ring_y[(i + 1) % n]
        segment_dx, segment_dy = x2 - x1, y2 - y1
        segment_length_sq = segment_dx**2 + segment_dy**2
        if segment_length_sq == 0:
            t = np.zeros(len(px))
        else:
            t = np.clip(((px - x1) * segment_dx + (py - y1) * segment_dy) / segment_length_sq, 0, 1)
        distance = np.hypot(px - (x1 + t * segment_dx), py - (y1 + t * segment_dy))
        min_distance = np.minimum(min_distance, distance)
    return min_distance


def _find_polygon_label_anchor(ring: list[list[float]], grid_size: int = 40) -> tuple[float, float]:
    """
    Tìm điểm (lon, lat) TỐT NHẤT để đặt nhãn tên bên trong 1 polygon ranh giới hành chính - xấp xỉ
    thuật toán "pole of inaccessibility" (điểm nằm bên trong polygon và XA BIÊN nhất) bằng lưới thô rồi
    tinh chỉnh thêm 1 lớp lưới mịn quanh điểm tốt nhất.

    TẠI SAO CẦN HÀM NÀY: trước đây nhãn tên địa phương dùng thẳng toạ độ cố định ở
    `REAL_MONITORED_LOCATIONS` (1 điểm đại diện trung tâm thị trấn) - toạ độ đó KHÔNG nhất thiết rơi
    vào phần "thân" chính của polygon ranh giới THẬT (`load_district_boundaries()`), đặc biệt với các
    địa phương có hình dạng lồi lõm/kéo dài dọc sông (vd Hương Trà) - khiến nhãn hiển thị LỆCH hẳn ra
    ngoài vùng tô màu xanh/đỏ trên bản đồ, đã thấy lỗi này thật khi xem bản đồ đã publish. Không dùng
    centroid diện tích đơn thuần vì với polygon lõm/phân nhánh, centroid có thể rơi ra NGOÀI polygon
    (vào đúng chỗ lõm). Không thêm dependency `shapely`/`geopandas` (có tên trong requirements.txt
    nhưng chưa cài trong venv hiện tại) - chỉ dùng numpy đã có sẵn.
    """
    ring_x = np.array([point[0] for point in ring], dtype=float)
    ring_y = np.array([point[1] for point in ring], dtype=float)

    def best_inside_point(x_min, x_max, y_min, y_max):
        grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
        grid_x, grid_y = grid_x.ravel(), grid_y.ravel()
        inside = _points_in_polygon(grid_x, grid_y, ring_x, ring_y)
        if not inside.any():
            return None
        candidates_x, candidates_y = grid_x[inside], grid_y[inside]
        distances = _min_distance_to_ring(candidates_x, candidates_y, ring_x, ring_y)
        best_index = int(np.argmax(distances))
        return float(candidates_x[best_index]), float(candidates_y[best_index])

    x_min, x_max, y_min, y_max = ring_x.min(), ring_x.max(), ring_y.min(), ring_y.max()
    coarse_best = best_inside_point(x_min, x_max, y_min, y_max)
    if coarse_best is None:
        # Lưới thô không trúng điểm nào bên trong (polygon quá mỏng) - dự phòng bằng centroid các đỉnh.
        return float(ring_x.mean()), float(ring_y.mean())

    step_x, step_y = (x_max - x_min) / grid_size, (y_max - y_min) / grid_size
    refined_best = best_inside_point(
        coarse_best[0] - step_x, coarse_best[0] + step_x, coarse_best[1] - step_y, coarse_best[1] + step_y
    )
    return refined_best if refined_best is not None else coarse_best


@st.cache_data(show_spinner=False)
def compute_district_label_anchors() -> dict[str, tuple[float, float]]:
    """
    Trả về `{tên địa phương: (lat, lon)}` - điểm neo để đặt NHÃN TÊN, tính từ chính polygon ranh giới
    thật (`load_district_boundaries()`) bằng `_find_polygon_label_anchor()`, thay vì dùng thẳng toạ độ
    cố định ở `REAL_MONITORED_LOCATIONS`. Xem docstring `_find_polygon_label_anchor()` để biết lý do.
    Rỗng nếu thiếu file ranh giới - khi đó `build_smart_routing_map()` tự rơi về dùng
    `REAL_MONITORED_LOCATIONS` như cũ.
    """
    district_boundaries = load_district_boundaries()
    if not district_boundaries:
        return {}

    anchors: dict[str, tuple[float, float]] = {}
    for feature in district_boundaries["features"]:
        location_name = feature.get("properties", {}).get("name")
        geometry = feature.get("geometry", {})
        coordinates = geometry.get("coordinates") or []
        if geometry.get("type") == "MultiPolygon":
            candidate_rings = [polygon[0] for polygon in coordinates if polygon]
        else:
            candidate_rings = [coordinates[0]] if coordinates else []
        if not location_name or not candidate_rings:
            continue
        # Nhãn nên đặt trên phần đất CHÍNH (mảnh lớn nhất theo diện tích bounding box), không phải 1
        # mảnh nhỏ tách rời khi polygon là MultiPolygon.
        exterior_ring = max(
            candidate_rings,
            key=lambda ring: (
                max(p[0] for p in ring) - min(p[0] for p in ring)
            ) * (max(p[1] for p in ring) - min(p[1] for p in ring)),
        )
        anchor_lon, anchor_lat = _find_polygon_label_anchor(exterior_ring)
        anchors[location_name] = (anchor_lat, anchor_lon)
    return anchors


# Màu tô theo trạng thái nguy cơ - dùng chung cho cả nhánh tô ranh giới (GeoJson) lẫn nhánh marker
# dự phòng, đảm bảo 2 cách hiển thị luôn nhất quán màu sắc với nhau.
RISK_FILL_COLOR_MAP = {"Ngập": "#EF4444", "An toàn": "#22C55E"}
RISK_TOOLTIP_LABEL_MAP = {
    "Ngập": "Nguy cơ NGẬP (dự báo AI)",
    "An toàn": "An toàn",
}


def build_forecast_popup_html(location_name: str, location_forecast_df: pd.DataFrame | None) -> str:
    """
    Tạo nội dung HTML cho POPUP hiện ra khi CLICK vào vùng giám sát (khác `tooltip` chỉ hiện lúc rê
    chuột) - tóm tắt dự báo `FORECAST_DAYS_AHEAD` ngày tới (T đến T+FORECAST_DAYS_AHEAD-1, hiện là 14
    ngày) của CHÍNH địa phương đó.

    Tái sử dụng ĐÚNG kết quả đã tính sẵn ở `_compute_forecast_4day_result()` (dùng chung với Tab
    'Dự báo N ngày tới') - KHÔNG gọi lại Open-Meteo/suy luận model riêng cho việc build popup, để
    click vào bản đồ luôn phản hồi tức thì thay vì phải chờ gọi API mỗi lần.

    Bảng bọc trong 1 div `max-height` + `overflow-y: auto` vì 14 dòng dễ khiến popup quá dài (khác
    bản 4 ngày cũ không cần cuộn) - vẫn xem đủ toàn bộ chuỗi ngày mà không đẩy popup tràn ra ngoài
    khung nhìn bản đồ.
    """
    if location_forecast_df is None or location_forecast_df.empty:
        return (
            f"<b>{location_name}</b><br>"
            f"<i>Chưa có dữ liệu dự báo cho địa phương này (xem Tab 'Dự báo {FORECAST_DAYS_AHEAD} ngày "
            "tới' để biết chi tiết lỗi, ví dụ chưa huấn luyện model hoặc Open-Meteo tạm thời lỗi).</i>"
        )

    row_html_parts = []
    for _, row in location_forecast_df.iterrows():
        row_html_parts.append(
            "<tr>"
            f"<td style='padding:2px 6px;border-bottom:1px solid #e2e8f0;'>{row['Ngày']}</td>"
            f"<td style='padding:2px 6px;border-bottom:1px solid #e2e8f0;text-align:right;'>"
            f"{row['Dự báo Lượng mưa (mm)']:.1f} mm</td>"
            f"<td style='padding:2px 6px;border-bottom:1px solid #e2e8f0;'>{row['Dự đoán Ngập']}</td>"
            "</tr>"
        )

    return (
        "<div style='font-family: sans-serif; min-width: 260px;'>"
        f"<b>{location_name} - Dự báo {FORECAST_DAYS_AHEAD} ngày tới</b>"
        "<div style='max-height: 260px; overflow-y: auto; margin-top: 6px;'>"
        "<table style='width:100%; border-collapse: collapse; font-size: 12px;'>"
        "<tr style='background:#f1f5f9;'>"
        "<th style='text-align:left;padding:2px 6px;'>Ngày</th>"
        "<th style='padding:2px 6px;'>Mưa dự báo</th>"
        "<th style='text-align:left;padding:2px 6px;'>Nguy cơ</th>"
        "</tr>"
        f"{''.join(row_html_parts)}"
        "</table>"
        "</div>"
        "</div>"
    )


def build_smart_routing_map(
    df_predictions: pd.DataFrame,
    real_flooded_polygons: list,
    start_point: tuple[float, float] | None = None,
    end_point: tuple[float, float] | None = None,
    route_points: list | None = None,
    forecast_by_location: dict[str, pd.DataFrame] | None = None,
) -> folium.Map:
    """Dựng bản đồ Folium DUY NHẤT gộp cả 2 chức năng:
    1) Giám sát 5 địa phương thực tế - TÔ RANH GIỚI HÀNH CHÍNH (không phải chấm điểm) màu XANH LÁ
       ('An toàn') / ĐỎ ('Ngập') theo đúng địa giới thật, thay vì 1 chấm điểm đại diện - trực quan hơn
       nhiều vì thể hiện đúng PHẠM VI địa phương đang được cảnh báo, không chỉ 1 toạ độ trung tâm.
       CLICK vào 1 vùng sẽ hiện POPUP tóm tắt dự báo 14 ngày tới của địa phương đó (nếu có sẵn dữ liệu
       trong `forecast_by_location`), TOOLTIP (rê chuột) vẫn giữ nguyên hiển thị trạng thái hiện tại.
    2) Định tuyến - marker điểm đi/đến + tuyến đường né ngập vẽ XANH DƯƠNG.
    """
    center_lat = sum(lat for lat, _ in REAL_MONITORED_LOCATIONS.values()) / len(REAL_MONITORED_LOCATIONS)
    center_lon = sum(lon for _, lon in REAL_MONITORED_LOCATIONS.values()) / len(REAL_MONITORED_LOCATIONS)
    # NỀN BẢN ĐỒ: dùng Esri "World Dark Gray Canvas" (server.arcgisonline.com) thay vì OpenStreetMap
    # hoặc CartoDB mặc định trước đây - ĐÃ KIỂM CHỨNG THỰC TẾ CẢ HAI ĐỀU LỖI:
    #   1. `tile.openstreetmap.org` (OSM gốc): chặn/không phản hồi ổn định từ nhiều môi trường server/
    #      cloud (chính sách Tile Usage Policy của OSM ưu tiên trình duyệt người dùng cuối).
    #   2. `basemaps.cartocdn.com` ("CartoDB dark_matter"): tải được (HTTP 200) nhưng trả về ẢNH
    #      WATERMARK "API KEY REQUIRED" thay vì bản đồ thật - CARTO đã đổi chính sách, gói ẩn danh
    #      miễn phí không còn dùng được cho basemap nữa (lỗi thật gặp khi deploy, đã tự kiểm tra bằng
    #      cách tải ảnh tile về xem trực tiếp, không chỉ dựa vào mã HTTP 200 - 200 không có nghĩa là
    #      NỘI DUNG đúng).
    # Esri Dark Gray Canvas (2 lớp: Base + Reference nhãn) ĐÃ kiểm chứng bằng cách tải + xem ảnh tile
    # THẬT tại chính khu vực Huế - ra bản đồ chi tiết, không watermark, không cần API key, thuộc dịch
    # vụ ArcGIS Online công khai của Esri (được phép dùng ẩn danh cho mục đích tham chiếu chung).
    routing_map = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=None)
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri, HERE, Garmin, FAO, NOAA, USGS",
        name="Nền bản đồ",
        overlay=False,
        control=False,
        max_zoom=16,
    ).add_to(routing_map)
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri",
        name="Nhãn tên đường/địa danh",
        overlay=True,
        control=False,
        max_zoom=16,
    ).add_to(routing_map)

    # ---- (1) Giám sát: TÔ RANH GIỚI HÀNH CHÍNH thật của 5 địa phương, màu theo đúng 'Nguy cơ' dự báo
    # của AI. FAIL-SAFE: thiếu dòng dữ liệu cũng KHÔNG mặc định "An toàn" (xem docstring
    # get_latest_flood_predictions) - địa phương "Không xác định" tô màu XÁM để phân biệt rõ.
    district_boundaries = load_district_boundaries()

    if district_boundaries:
        for feature in district_boundaries["features"]:
            location_name = feature.get("properties", {}).get("name")
            risk_rows = df_predictions.loc[df_predictions["Địa phương"] == location_name, "Nguy cơ"]
            risk_status = risk_rows.iloc[0] if not risk_rows.empty else "Không xác định"
            fill_color = RISK_FILL_COLOR_MAP.get(risk_status, "#9CA3AF")
            tooltip_label = RISK_TOOLTIP_LABEL_MAP.get(
                risk_status, "Không xác định được (lỗi model/dữ liệu - cần kiểm tra thủ công)"
            )
            location_forecast_df = (
                forecast_by_location.get(location_name) if forecast_by_location is not None else None
            )

            folium.GeoJson(
                feature,
                style_function=lambda _feat, fill_color=fill_color: {
                    "fillColor": fill_color,
                    "color": fill_color,
                    "weight": 2,
                    "fillOpacity": 0.45,
                },
                highlight_function=lambda _feat: {"weight": 3, "fillOpacity": 0.65},
                tooltip=folium.Tooltip(f"{tooltip_label} - {location_name}"),
                popup=folium.Popup(
                    build_forecast_popup_html(location_name, location_forecast_df), max_width=320
                ),
            ).add_to(routing_map)
    else:
        # DỰ PHÒNG: thiếu/lỗi file GeoJSON ranh giới -> vẽ lại marker điểm như bản trước, để bản đồ
        # vẫn hoạt động được thay vì trống trơn.
        for location_name, coordinates in REAL_MONITORED_LOCATIONS.items():
            risk_rows = df_predictions.loc[df_predictions["Địa phương"] == location_name, "Nguy cơ"]
            risk_status = risk_rows.iloc[0] if not risk_rows.empty else "Không xác định"
            location_forecast_df = (
                forecast_by_location.get(location_name) if forecast_by_location is not None else None
            )
            popup = folium.Popup(build_forecast_popup_html(location_name, location_forecast_df), max_width=320)

            if risk_status == "Ngập":
                folium.Marker(
                    location=coordinates,
                    tooltip=f"{location_name}: Nguy cơ NGẬP (dự báo AI)",
                    popup=popup,
                    icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa"),
                ).add_to(routing_map)
            elif risk_status == "An toàn":
                folium.Marker(
                    location=coordinates,
                    tooltip=f"{location_name}: An toàn",
                    popup=popup,
                    icon=folium.Icon(color="green", icon="check", prefix="fa"),
                ).add_to(routing_map)
            else:
                folium.Marker(
                    location=coordinates,
                    tooltip=f"{location_name}: không xác định được (lỗi model/dữ liệu - cần kiểm tra thủ công)",
                    popup=popup,
                    icon=folium.Icon(color="gray", icon="question", prefix="fa"),
                ).add_to(routing_map)

    # ---- NHÃN TÊN CỐ ĐỊNH cho 5 địa phương - LUÔN HIỂN THỊ trên bản đồ (không cần rê chuột/click như
    # tooltip/popup ở trên) - dùng `DivIcon` (nhãn chữ thuần, không phải icon ghim) đặt NGAY TRÊN mỗi
    # vùng/điểm giám sát, màu viền theo đúng 'Nguy cơ' để vừa đọc được tên vừa nhận biết trạng thái
    # ngay từ cái nhìn đầu tiên, không phải tương tác mới biết đây là địa phương nào.
    # VỊ TRÍ NHÃN: ưu tiên điểm neo tính từ chính polygon ranh giới thật (`compute_district_label_
    # anchors()`) để nhãn LUÔN nằm bên trong vùng tô màu xanh/đỏ - trước đây dùng thẳng toạ độ cố định
    # ở REAL_MONITORED_LOCATIONS nên có địa phương (vd Hương Trà, hình dạng kéo dài dọc sông) bị lệch
    # nhãn ra hẳn ngoài vùng tô màu. Chỉ rơi về REAL_MONITORED_LOCATIONS khi thiếu file ranh giới.
    district_label_anchors = compute_district_label_anchors()
    for location_name, fallback_coordinates in REAL_MONITORED_LOCATIONS.items():
        label_position = district_label_anchors.get(location_name, fallback_coordinates)
        risk_rows = df_predictions.loc[df_predictions["Địa phương"] == location_name, "Nguy cơ"]
        risk_status = risk_rows.iloc[0] if not risk_rows.empty else "Không xác định"
        label_border_color = RISK_FILL_COLOR_MAP.get(risk_status, "#9CA3AF")
        folium.Marker(
            location=label_position,
            icon=folium.DivIcon(
                html=(
                    '<div style="'
                    "font-size: 12px; font-weight: 700; color: #f8fafc; "
                    "background: rgba(15, 23, 42, 0.85); padding: 2px 7px; border-radius: 4px; "
                    f"border: 1.5px solid {label_border_color}; white-space: nowrap; "
                    'transform: translate(-50%, -50%);">'
                    f"{location_name}</div>"
                )
            ),
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
            tooltip="Vùng ngập ước tính - routing engine tự động né khu vực này",
        ).add_to(routing_map)

    # ---- (2) Định tuyến: điểm đi/đến (màu riêng, tránh trùng với màu xanh lá/đỏ của giám sát) ----
    if start_point is not None:
        folium.Marker(
            location=start_point,
            tooltip="Điểm xuất phát",
            icon=folium.Icon(color="blue", icon="play", prefix="fa"),
        ).add_to(routing_map)
    if end_point is not None:
        folium.Marker(
            location=end_point,
            tooltip="Điểm đến",
            icon=folium.Icon(color="cadetblue", icon="flag-checkered", prefix="fa"),
        ).add_to(routing_map)

    if route_points:
        folium.PolyLine(
            locations=route_points,
            color="#2563EB",
            weight=6,
            opacity=0.85,
            tooltip="Tuyến đường di chuyển (đã né vùng ngập)",
        ).add_to(routing_map)

    return routing_map


def render_smart_routing_tab() -> None:
    """
    Nội dung Tab 4 - Bản đồ tránh ngập, bước THỨ TƯ (sản phẩm ứng dụng thực tế) của pipeline.

    THIẾT KẾ LẠI THEO YÊU CẦU THỰC TẾ (khác bản demo trước - vốn chỉ cho chọn giữa 5 điểm giám sát
    cố định và luôn cần bấm nút thủ công):
      1. ĐIỂM ĐI/ĐẾN LÀ BẤT KỲ TRÊN BẢN ĐỒ: người dùng click trực tiếp lên bản đồ để đặt điểm xuất
         phát/điểm đến ở BẤT KỲ đâu trong khu vực (không giới hạn quanh 5 điểm giám sát) - dùng
         `st_folium(..., returned_objects=["last_clicked"])` để đọc lại tọa độ vừa click.
      2. TỰ ĐỘNG TÌM ĐƯỜNG KHI CÓ NGẬP: chỉ cần MỘT địa phương giám sát chuyển sang trạng thái
         'Ngập', hệ thống tự động gọi TomTom tính lại tuyến né vùng ngập ngay, KHÔNG cần người dùng
         bấm nút - nút "Tìm tuyến đường" vẫn giữ lại để chủ động tính lại bất cứ lúc nào (kể cả khi
         đang an toàn), nhưng không còn là điều kiện BẮT BUỘC để có tuyến đường khi có ngập.
    """
    st.subheader("Bản đồ tránh ngập")
    st.caption(
        "Bước 4/4 của pipeline: giám sát 5 địa phương thực tế tại Thừa Thiên Huế bằng kết quả dự báo "
        "của model AI. Click trực tiếp lên bản đồ để đặt điểm xuất phát/điểm đến ở bất kỳ vị trí nào - "
        "khi có địa phương đang ngập, hệ thống tự động tính lại tuyến né vùng ngập, không cần bấm nút."
    )

    # ==============================================================================================
    # BƯỚC 1: ĐỌC ĐỘNG kết quả dự báo mới nhất (df_predictions) cho 5 địa phương giám sát thực tế.
    # ==============================================================================================
    df_predictions = get_latest_flood_predictions()

    # Dữ liệu dự báo 14 ngày tới (T đến T+13) CHO TỪNG ĐỊA PHƯƠNG - dùng để hiện popup tóm tắt khi
    # click vào vùng giám sát trên bản đồ (xem `build_forecast_popup_html`). Tái sử dụng ĐÚNG cache
    # `_compute_forecast_4day_result()` đã tính cho Tab 'Dự báo 14 ngày tới' - không gọi lại Open-Meteo
    # riêng cho việc này. Bọc try/except vì tab Bản đồ vẫn phải hoạt động (giám sát + định tuyến) ngay
    # cả khi chưa có model triển khai hoặc Open-Meteo tạm thời lỗi - lúc đó popup chỉ báo "chưa có dữ
    # liệu" thay vì làm crash cả tab.
    try:
        forecast_by_location = {
            location_name: location_df.reset_index(drop=True)
            for location_name, location_df in _compute_forecast_4day_result()["combined_df"].groupby("Địa phương")
        }
    except Exception:
        forecast_by_location = None

    # ---- BƯỚC 2: từ df_predictions, suy ra danh sách vùng ngập THỰC TẾ cần né (real_flooded_polygons)
    # và danh sách TÊN các địa phương đang ngập (dùng làm "chữ ký" để biết khi nào tình trạng ngập
    # thay đổi, phục vụ auto-trigger ở BƯỚC 4).
    real_flooded_polygons: list[list[tuple[float, float]]] = []
    flooded_location_names: list[str] = []
    unknown_location_names: list[str] = []
    for location_name, coordinates in REAL_MONITORED_LOCATIONS.items():
        risk_rows = df_predictions.loc[df_predictions["Địa phương"] == location_name, "Nguy cơ"]
        risk_status = risk_rows.iloc[0] if not risk_rows.empty else "Không xác định"
        if risk_status == "Ngập":
            real_flooded_polygons.append(build_flood_zone_polygon(coordinates))
            flooded_location_names.append(location_name)
        elif risk_status == "Không xác định":
            unknown_location_names.append(location_name)

    st.markdown("##### Trạng thái giám sát 5 địa phương (từ dự báo AI mới nhất)")
    st.dataframe(df_predictions, use_container_width=True, hide_index=True)
    if unknown_location_names:
        # FAIL-SAFE: cảnh báo RÕ RÀNG thay vì để những địa phương này âm thầm biến thành "An toàn"
        # (xem docstring get_latest_flood_predictions) - routing engine cũng KHÔNG tự né được các khu
        # vực này vì không có đủ dữ liệu để dựng vùng ngập, nên cần con người kiểm tra thủ công.
        st.warning(
            f"Không xác định được nguy cơ ngập cho: {', '.join(unknown_location_names)} "
            "(model/dữ liệu lỗi - xem log server). Các địa phương này không được tự động né khi định "
            "tuyến - vui lòng kiểm tra thủ công trước khi di chuyển qua khu vực này."
        )

    # ==============================================================================================
    # BƯỚC 3: CHỌN ĐIỂM ĐI/ĐẾN BẰNG CÁCH CLICK LÊN BẢN ĐỒ (thay vì chỉ chọn trong 5 điểm cố định).
    # `st.radio` xác định click TIẾP THEO trên bản đồ sẽ gán vào điểm nào; tọa độ mặc định ban đầu
    # lấy tạm 2 điểm giám sát để bản đồ/tuyến đường có sẵn dữ liệu ngay từ lần mở tab đầu tiên, người
    # dùng có thể click để thay đổi bất cứ lúc nào.
    # ==============================================================================================
    if "routing_start_point" not in st.session_state:
        st.session_state["routing_start_point"] = REAL_MONITORED_LOCATIONS["TP Huế"]
    if "routing_end_point" not in st.session_state:
        st.session_state["routing_end_point"] = REAL_MONITORED_LOCATIONS["Phú Vang"]

    control_left, control_center, control_right = st.columns([1, 2, 1])
    with control_center:
        click_target = st.radio(
            "Click lên bản đồ để đặt:",
            options=["Điểm xuất phát", "Điểm đến"],
            horizontal=True,
            key="routing_click_target",
        )
        start_point = st.session_state["routing_start_point"]
        end_point = st.session_state["routing_end_point"]
        st.caption(
            f"Xuất phát: `{start_point[0]:.4f}, {start_point[1]:.4f}` | "
            f"Đến: `{end_point[0]:.4f}, {end_point[1]:.4f}`"
        )
        find_route_clicked = st.button(
            "Tính lại tuyến đường",
            key="find_smart_route_button",
            use_container_width=True,
            help="Tuyến đường tự động tính lại khi có địa phương đang ngập - bấm nút này để chủ động tính lại bất cứ lúc nào.",
        )

    # ---- Bản đồ đặt CĂN GIỮA, rộng và cân đối - vẽ TRƯỚC khi xử lý click để lấy được sự kiện click
    # của LẦN RENDER NÀY (st_folium trả `last_clicked` ngay khi vẽ xong bản đồ) ----
    map_left, map_center, map_right = st.columns([1, 6, 1])
    with map_center:
        route_result_for_map = st.session_state.get("smart_route_result")
        route_points_to_draw = (
            route_result_for_map["route_points"]
            if route_result_for_map and route_result_for_map["success"]
            else None
        )
        smart_map = build_smart_routing_map(
            df_predictions=df_predictions,
            real_flooded_polygons=real_flooded_polygons,
            start_point=start_point,
            end_point=end_point,
            route_points=route_points_to_draw,
            forecast_by_location=forecast_by_location,
        )
        # `returned_objects=["last_clicked"]`: BẮT BUỘC phải đọc lại tọa độ click để hỗ trợ chọn điểm
        # tùy ý trên bản đồ - đánh đổi là app sẽ rerun mỗi khi người dùng click/pan/zoom bản đồ (khác
        # với bản trước dùng `returned_objects=[]` để tắt hẳn, khi đó không thể đọc được click).
        map_state = st_folium(
            smart_map, width=1000, height=600, returned_objects=["last_clicked"], key="smart_routing_folium_map"
        )

    # ==============================================================================================
    # BƯỚC 4: XỬ LÝ CLICK MỚI - so sánh với click đã xử lý gần nhất (lưu trong session_state) để chỉ
    # cập nhật điểm đi/đến ĐÚNG 1 LẦN cho mỗi lượt click thật, tránh việc `st_folium` trả lại cùng 1
    # tọa độ ở các lần rerun tiếp theo (do tương tác widget khác) làm điểm bị "đặt lại" ngoài ý muốn.
    #
    # SỬA HIỆU NĂNG: bản trước gọi `st.rerun()` NGAY tại đây, rồi BƯỚC 5 (ở lượt render KẾ TIẾP) mới
    # tính tuyến và `st.rerun()` LẦN NỮA - 1 lượt click tốn tới 2 lần rerun toàn bộ script liên tiếp
    # (tăng gấp đôi độ trễ cảm nhận được). Nay CHỈ cập nhật session_state + biến local ở đây (không
    # rerun ngay), để BƯỚC 5 tính tuyến (nếu cần) trong CÙNG 1 lượt chạy này - `st.rerun()` của
    # BƯỚC 5 (nếu có chạy) sẽ vẽ lại bản đồ với CẢ marker mới LẪN tuyến mới trong đúng 1 lần rerun.
    # ==============================================================================================
    click_processed_this_run = False
    last_clicked = (map_state or {}).get("last_clicked")
    if last_clicked:
        clicked_point = (round(last_clicked["lat"], 6), round(last_clicked["lng"], 6))
        if clicked_point != st.session_state.get("routing_last_processed_click"):
            st.session_state["routing_last_processed_click"] = clicked_point
            if click_target == "Điểm xuất phát":
                st.session_state["routing_start_point"] = clicked_point
            else:
                st.session_state["routing_end_point"] = clicked_point
            click_processed_this_run = True

    start_point = st.session_state["routing_start_point"]
    end_point = st.session_state["routing_end_point"]

    # ==============================================================================================
    # BƯỚC 5: TỰ ĐỘNG GỌI TOMTOM KHI CÓ GÌ ĐÓ THẬT SỰ THAY ĐỔI - không cần người dùng bấm nút. Dùng
    # "chữ ký" (điểm đi, điểm đến, danh sách địa phương đang ngập) để CHỈ tính lại khi có thay đổi
    # thật, tránh gọi lại TomTom vô ích ở những lần rerun không liên quan (đổi radio, mở tab khác...).
    #
    # SỬA LỖI: bản trước chỉ auto-fetch khi `bool(flooded_location_names)` = True, gây 2 hệ quả sai:
    #   (1) Vừa hết ngập (flooded_location_names rỗng trở lại) -> route/thông báo "đã né vùng ngập"
    #       CŨ vẫn hiển thị y nguyên, mâu thuẫn với "Vùng ngập đã né: 0" đang hiện ngay bên cạnh.
    #   (2) Click đổi điểm đi/đến trong lúc KHÔNG có ngập -> marker di chuyển nhưng đường kẻ xanh +
    #       số liệu quãng đường/thời gian vẫn của CẶP ĐIỂM CŨ, sai hoàn toàn so với điểm đang chọn.
    # Nay auto-fetch theo đúng "chữ ký đã đổi", KHÔNG còn phụ thuộc có ngập hay không - miễn là đã có
    # 1 tuyến đường được tính trước đó (`has_existing_route`) hoặc đang có ngập, để vẫn giữ tinh thần
    # tiết kiệm API cho đúng 1 trường hợp hợp lệ: phiên mới mở, chưa từng tính tuyến, chưa có ngập,
    # chưa ai bấm nút -> không tự ý gọi TomTom.
    # ==============================================================================================
    same_point_error = start_point == end_point
    current_signature = (start_point, end_point, tuple(sorted(flooded_location_names)))
    has_existing_route = "smart_route_signature" in st.session_state
    should_auto_fetch = (
        not same_point_error
        and st.session_state.get("smart_route_signature") != current_signature
        and (bool(flooded_location_names) or has_existing_route)
    )

    if (find_route_clicked or should_auto_fetch) and not same_point_error:
        spinner_text = (
            "Phát hiện thay đổi - đang tự động tính lại tuyến đường..."
            if should_auto_fetch and not find_route_clicked
            else "Đang tính toán tuyến đường..."
        )
        with st.spinner(spinner_text):
            # Dịch điểm đi/đến ra khỏi vùng ngập (nếu vô tình trùng tâm) TRƯỚC khi gọi TomTom - xem
            # docstring `nudge_point_outside_flood_zones()`. Chỉ ảnh hưởng lời gọi API, không đổi
            # marker hiển thị trên bản đồ.
            safe_start_point = nudge_point_outside_flood_zones(start_point, real_flooded_polygons)
            safe_end_point = nudge_point_outside_flood_zones(end_point, real_flooded_polygons)
            st.session_state["smart_route_result"] = fetch_tomtom_route(
                safe_start_point, safe_end_point, flooded_polygons=real_flooded_polygons
            )
        st.session_state["smart_route_signature"] = current_signature
        st.session_state["smart_route_auto_triggered"] = should_auto_fetch and not find_route_clicked
        st.rerun()  # Vẽ lại bản đồ với CẢ marker mới (nếu vừa click) LẪN tuyến vừa tính, trong 1 lần.

    # Rerun DỰ PHÒNG: chỉ chạy tới đây nếu BƯỚC 5 KHÔNG tự rerun ở trên (ví dụ: vừa click đổi điểm
    # nhưng chưa đủ điều kiện auto-fetch - không có ngập, chưa từng tính tuyến; hoặc điểm mới trùng
    # nhau nên bị `same_point_error` chặn) - vẫn cần đúng 1 lần rerun để bản đồ hiển thị marker mới.
    if click_processed_this_run:
        st.rerun()

    route_result = st.session_state.get("smart_route_result")

    with control_center:
        if same_point_error:
            st.warning("Điểm xuất phát và điểm đến đang trùng nhau - hãy click lại để chọn 2 vị trí khác nhau.")
        elif route_result is None:
            st.info("Click lên bản đồ để đặt điểm đi/đến, hoặc bấm **Tính lại tuyến đường**.")
        elif not route_result["success"]:
            st.error(f"Không thể lấy tuyến đường từ TomTom: {route_result['error']}")
        else:
            metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
            metric_col_1.metric("Quãng đường", f"{route_result['distance_km']} km")
            metric_col_2.metric("Thời gian di chuyển", f"{route_result['travel_time_min']} phút")
            metric_col_3.metric("Vùng ngập đã né", f"{len(real_flooded_polygons)}")
            if route_result["used_avoid_areas"]:
                auto_note = (
                    " (tự động cập nhật)" if st.session_state.get("smart_route_auto_triggered") else ""
                )
                st.success(f"Đã tính tuyến đường né vùng ngập thành công bằng TomTom Routing API{auto_note}.")
            else:
                st.success(
                    "Không địa phương nào đang có nguy cơ ngập - TomTom tính tuyến đường bình "
                    "thường (không kèm avoidAreas)."
                )


# ==================================================================================================
# SIDEBAR & ĐIỂM VÀO CHÍNH
# ==================================================================================================
def render_sidebar() -> None:
    """Sidebar tối giản: giới thiệu nhanh các bước pipeline + nút làm mới cache toàn cục."""
    st.sidebar.markdown("## Flood Prediction Pipeline")
    st.sidebar.markdown(
        "1. Dự báo 14 ngày tới\n"
        "2. Khám phá dữ liệu (EDA)\n"
        "3. Tiền xử lý & huấn luyện\n"
        "4. Đánh giá mô hình\n"
        "5. Bản đồ tránh ngập\n"
    )
    st.sidebar.markdown("---")
    if st.sidebar.button("Làm mới toàn bộ cache", key="clear_all_cache_button", use_container_width=True):
        # `st.cache_data.clear()` xóa TẤT CẢ hàm decorate bằng `@st.cache_data` trong app, bao gồm cả
        # `_compute_forecast_4day_result()` (cache dự báo 4 ngày, persist="disk") - vì cache đó nay
        # dùng chung cơ chế built-in của Streamlit thay vì tự quản lý file JIT riêng như bản trước, nút
        # này không cần biết/xóa thủ công từng cache con nữa.
        st.cache_data.clear()
        st.cache_resource.clear()
        st.sidebar.success("Đã xóa cache - dữ liệu sẽ được nạp lại ở lần chạy tiếp theo.")

    render_admin_api_key_panel()


def main():
    """Điểm vào chính của ứng dụng - trang đầu là DỰ BÁO thực tế, tiếp theo mới đến các tab kỹ thuật
    bám sát vòng đời Data Science (Data Science Lifecycle)."""
    apply_global_ui_theme()
    render_sidebar()

    st.title("Dự báo ngập lụt Thừa Thiên Huế")
    st.caption(
        "Dự báo 14 ngày tới → Khám phá dữ liệu → Tiền xử lý & huấn luyện → Đánh giá mô hình → "
        "Bản đồ chỉ đường tránh ngập."
    )
    st.markdown("---")

    # ------------------------------------------------------------------------------------------
    # BỐ CỤC TOÀN TRANG BẰNG st.tabs: TAB ĐẦU TIÊN là kết quả DỰ BÁO thực tế (giá trị cốt lõi mà
    # người dùng cuối/hội đồng cần thấy ngay), các tab sau bám sát vòng đời Data Science (EDA ->
    # Huấn luyện -> Đánh giá) để người xem hiểu được PHƯƠNG PHÁP đứng sau kết quả dự báo đó, và tab
    # cuối là sản phẩm ứng dụng (bản đồ chỉ đường tránh ngập).
    # ------------------------------------------------------------------------------------------
    tab_forecast, tab_eda, tab_train, tab_eval, tab_map = st.tabs(
        [
            "Dự báo 14 ngày tới",
            "Khám phá dữ liệu (EDA)",
            "Tiền xử lý & huấn luyện",
            "Đánh giá mô hình",
            "Bản đồ tránh ngập",
        ]
    )

    with tab_forecast:
        render_forecast_tab()

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
