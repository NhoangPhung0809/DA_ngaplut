import os

# PHẢI set TRƯỚC bất kỳ import nào có thể kéo theo `protobuf` (vd tensorflow) - server đang cài
# tensorflow==2.10.1 (cần protobuf<3.20) CÙNG LÚC với streamlit bản mới (cần protobuf>=3.20), 2 yêu
# cầu xung đột trực tiếp nên không thể hạ/nâng version `protobuf` cho vừa cả 2. Ép dùng cài đặt Python
# thuần (thay vì C++ backend mặc định) là cách chính thức Google khuyến nghị cho đúng tình huống này -
# tránh lỗi "Descriptors cannot not be created directly" khi import `tensorflow.keras`. Chậm hơn 1 chút
# nhưng không cần đổi version package nào, không phá streamlit.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import glob
import gc
import json
import shutil
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

from shared_constants import FEATURE_COLS

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
except ImportError as exc:
    raise ImportError(
        "Missing dependency `imbalanced-learn`. Install it with: "
        "`pip install imbalanced-learn`"
    ) from exc

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    from ctgan import CTGAN
except ImportError:
    CTGAN = None

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    SARIMAX = None

try:
    from tensorflow.keras import Model
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Conv1D, Dense, Flatten, GRU, Input, LSTM, MaxPooling1D
    from tensorflow.keras.models import Sequential, load_model
except ImportError:
    Model = None
    K = None
    EarlyStopping = None
    Conv1D = None
    LSTM = None
    GRU = None
    Dense = None
    Flatten = None
    Input = None
    MaxPooling1D = None
    Sequential = None
    load_model = None

warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "historical"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"
LATEST_MODELS_DIR = MODELS_DIR / "latest"
CTGAN_BEFORE_PATH = BASE_DIR / "data" / "data_before_ctgan.csv"
CTGAN_AFTER_PATH = BASE_DIR / "data" / "data_after_ctgan.csv"
CTGAN_DISTRIBUTION_PATH = BASE_DIR / "data" / "ctgan_class_distribution.json"
INCREMENTAL_CACHE_DIR = BASE_DIR / "cache"
INCREMENTAL_CURSOR_PATH = INCREMENTAL_CACHE_DIR / "incremental_cursor.json"
BEST_XGBOOST_PATH = LATEST_MODELS_DIR / "best_xgboost.json"
BEST_LSTM_PATH = LATEST_MODELS_DIR / "best_lstm_model.keras"
LSTM_SEQ_SCALER_PATH = LATEST_MODELS_DIR / "lstm_seq_scaler.pkl"

TIME_COL = "Thời_gian"
DATE_COL = "Ngày"
LOCATION_COL = "Địa phương"
TARGET_COL = "Nguy_cơ_ngập"
# FEATURE_COLS import từ shared_constants.py - xem docstring file đó để biết lý do (trước đây định
# nghĩa độc lập ở 5 file, dễ lệch nhau khi đổi bộ đặc trưng).
CLASS_LABELS = [0, 1, 2]
CLASS_NAME_MAP = {
    0: "Safe",
    1: "Light Flood",
    2: "Heavy Flood",
}
ALL_MODEL_NAMES = [
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
SEQUENCE_WINDOW = 7
CTGAN_MAX_TRAIN_ROWS_PER_CLASS = 5000
CTGAN_MAX_TARGET_ROWS_PER_CLASS = 10000
CTGAN_EPOCHS = 10
CTGAN_BATCH_SIZE = 256
CTGAN_EXPORT_SAMPLE_SIZE = 1000
INCREMENTAL_LSTM_EPOCHS = 6
INCREMENTAL_XGB_N_ESTIMATORS = 60
TARGET_LEAKAGE_KEYWORDS = (
    "muc_ngap",
    "flood_class",
    "target",
    "rain_level",
    "label",
    "nguy_cơ_ngập",
    "nguy_co_ngap",
)


def ensure_base_directories() -> None:
    """Tạo các thư mục gốc phục vụ pipeline và frontend."""
    CTGAN_BEFORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    INCREMENTAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def validate_optional_dependencies() -> None:
    """Thông báo nhanh về các thư viện tùy chọn chưa cài."""
    optional_missing = []
    if LGBMClassifier is None:
        optional_missing.append("lightgbm")
    if CatBoostClassifier is None:
        optional_missing.append("catboost")
    if CTGAN is None:
        optional_missing.append("ctgan")
    if ARIMA is None or SARIMAX is None:
        optional_missing.append("statsmodels")
    if Sequential is None:
        optional_missing.append("tensorflow")

    if optional_missing:
        print(
            "Optional packages not available. Related model categories will be skipped: "
            + ", ".join(sorted(set(optional_missing)))
        )


def list_available_models() -> list[str]:
    """Trả về danh sách mô hình hiện thực sự khả dụng trong môi trường."""
    return list(build_model_registry().keys())


def normalize_location_name(file_path: str) -> str:
    """Chuẩn hóa tên địa phương từ tên file CSV."""
    stem = Path(file_path).stem
    stem = stem.replace("_10years", "").replace("_", " ").strip()
    return stem


def load_and_concatenate_csvs(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Đọc toàn bộ CSV lịch sử, ghép lại và sắp xếp theo thời gian."""
    csv_files = sorted(glob.glob(str(data_dir / "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    dataframes = []
    print(f"Found {len(csv_files)} CSV files. Reading data...")
    for file_path in csv_files:
        # `on_bad_lines="skip"`: nếu một dòng trong CSV có SỐ TRƯỜNG (số cột) không khớp với header
        # (ví dụ dòng có 8 trường trong khi header chỉ khai báo 7 cột - đúng lỗi
        # "Expected 7 fields in line 87626, saw 8" đã gặp), mặc định pandas sẽ NÉM ParserError và
        # dừng đọc toàn bộ file ngay lập tức, làm crash cả pipeline dù chỉ có vài dòng lỗi cục bộ.
        # Với `on_bad_lines="skip"`, pandas sẽ BỎ QUA riêng các dòng bị lỗi cấu trúc đó và tiếp tục
        # đọc phần còn lại của file bình thường, thay vì dừng toàn bộ. Từ pandas >= 1.3 (đang dùng
        # pandas 2.x), tham số này hoạt động tốt ngay với engine mặc định ("c"), không cần đổi sang
        # engine="python" (vốn chậm hơn nhiều lần trên file lớn ~88k dòng như ở đây).
        # Lưu ý quan trọng: đây là lớp phòng thủ (defensive coding) để pipeline không crash toàn bộ
        # vì vài dòng lỗi hiếm gặp - KHÔNG thay thế cho việc xử lý tận gốc nguyên nhân sinh ra dòng
        # lỗi (ở đề tài này, nguyên nhân gốc là do `fetch_data.py` từng ghi thêm 1 cột thừa khi append
        # dữ liệu mới, đã được sửa tận gốc + dọn lại toàn bộ file CSV lịch sử). Nếu số dòng đọc được
        # sau này thấp bất thường so với kỳ vọng, nên kiểm tra lại nguồn dữ liệu thay vì chỉ dựa vào
        # cờ skip này.
        df = pd.read_csv(file_path, on_bad_lines="skip")
        df[LOCATION_COL] = normalize_location_name(file_path)
        dataframes.append(df)
        print(f"Loaded: {Path(file_path).name} - {len(df)} rows")

    full_df = pd.concat(dataframes, ignore_index=True)
    full_df[TIME_COL] = pd.to_datetime(full_df[TIME_COL], errors="coerce")
    full_df = (
        full_df.dropna(subset=[TIME_COL])
        .sort_values([LOCATION_COL, TIME_COL])
        .reset_index(drop=True)
    )

    print(f"Total rows after concatenation: {len(full_df)}")
    return full_df


def build_daily_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp dữ liệu theo ngày để dùng feature ngày T dự báo nhãn ngày T+1."""
    daily_df = df.copy()
    daily_df[DATE_COL] = daily_df[TIME_COL].dt.floor("D")

    aggregation_map = {
        "Nhiệt_độ_C": "mean",
        "Độ_ẩm_%": "mean",
        "Lượng_mưa_mm": "sum",
        "Độ_ẩm_đất": "mean",
        "Chiều_cao_triều_m": "mean",
    }

    daily_df = (
        daily_df.groupby([LOCATION_COL, DATE_COL], as_index=False)
        .agg(aggregation_map)
        .rename(columns={DATE_COL: TIME_COL})
        .sort_values([LOCATION_COL, TIME_COL])
        .reset_index(drop=True)
    )

    print("\nConverted hourly data to daily feature dataset:")
    print(f"Daily rows after aggregation: {len(daily_df)}")
    return daily_df


def compute_train_only_medians(
    df: pd.DataFrame, feature_columns: list[str], train_ratio: float = 0.8
) -> pd.Series:
    """
    Tính median của từng cột trong `feature_columns` CHỈ TRÊN PHẦN DỮ LIỆU SẼ THUỘC TẬP TRAIN, mô
    phỏng lại ĐÚNG ranh giới thời gian mà `chronological_train_test_split()` sẽ dùng sau này (80%
    dòng ĐẦU TIÊN theo thời gian của MỖI địa phương) - dùng để điền giá trị thiếu (NaN).

    ----------------------------------------------------------------------------------------------
    TẠI SAO CẦN HÀM NÀY (ngăn rò rỉ dữ liệu tập test - data leakage)?
    ----------------------------------------------------------------------------------------------
    `create_multiclass_flood_label()` và `preprocess_features()` bắt buộc phải điền NaN TRƯỚC KHI
    `chronological_train_test_split()` chạy (vì nhãn rule-based và tập đặc trưng cần có mặt đầy đủ ở
    CẢ 2 phía trước khi tách train/test). Nếu điền bằng `df[column].median()` tính trên TOÀN BỘ dữ
    liệu (bao gồm cả những dòng SẼ thuộc tập test), các dòng thuộc tập TRAIN có thể vô tình được điền
    bằng 1 giá trị đã "biết trước" thống kê của tập test - đúng định nghĩa rò rỉ dữ liệu (data
    leakage), dù mức ảnh hưởng thường nhỏ vì chỉ tác động tới các dòng thực sự bị thiếu dữ liệu.

    Hàm này tính lại đúng ranh giới train/test theo thời gian cho TỪNG địa phương (giống hệt logic
    trong `chronological_train_test_split()`), rồi chỉ lấy median trên phần "sẽ là train" đó - đảm bảo
    giá trị dùng để điền không phụ thuộc vào bất kỳ dòng nào của tập test, dù được gọi TRƯỚC khi tách
    train/test thật.
    """
    if LOCATION_COL not in df.columns or TIME_COL not in df.columns:
        return df[feature_columns].median()

    train_rows = []
    for _, location_df in df.groupby(LOCATION_COL):
        location_df = location_df.sort_values(TIME_COL)
        split_index = max(1, int(len(location_df) * train_ratio))
        train_rows.append(location_df.iloc[:split_index])

    train_only_df = pd.concat(train_rows, ignore_index=True) if train_rows else df
    return train_only_df[feature_columns].median()


def create_multiclass_flood_label(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo nhãn 3 lớp dựa trên luật chuyên gia."""
    labeled_df = df.copy()

    # Dùng median TÍNH RIÊNG trên phần dữ liệu sẽ thuộc tập TRAIN để điền NaN - xem docstring
    # `compute_train_only_medians()` để biết lý do (ngăn rò rỉ thống kê của tập test vào tập train).
    train_only_medians = compute_train_only_medians(labeled_df, FEATURE_COLS)
    for column in FEATURE_COLS:
        if column not in labeled_df.columns:
            labeled_df[column] = 0.0
        labeled_df[column] = pd.to_numeric(labeled_df[column], errors="coerce")
        labeled_df[column] = labeled_df[column].fillna(train_only_medians[column])

    rain = labeled_df["Lượng_mưa_mm"].fillna(0)
    soil = labeled_df["Độ_ẩm_đất"].fillna(0)
    tide = labeled_df["Chiều_cao_triều_m"].fillna(0)

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

    labeled_df[TARGET_COL] = 0
    labeled_df.loc[light_flood_mask, TARGET_COL] = 1
    labeled_df.loc[heavy_flood_mask, TARGET_COL] = 2

    print("\nClass distribution after rule-based labeling:")
    print(labeled_df[TARGET_COL].value_counts().sort_index())
    return labeled_df


def apply_time_lag_target_shift(df: pd.DataFrame) -> pd.DataFrame:
    """Dùng feature ngày T để dự báo nhãn ngập của ngày T+1 theo từng địa phương."""
    lagged_df = df.copy()
    lagged_df = lagged_df.sort_values([LOCATION_COL, TIME_COL]).reset_index(drop=True)
    lagged_df[TARGET_COL] = lagged_df.groupby(LOCATION_COL)[TARGET_COL].shift(-1)

    rows_before_drop = len(lagged_df)
    lagged_df = lagged_df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    lagged_df[TARGET_COL] = lagged_df[TARGET_COL].astype(int)
    dropped_rows = rows_before_drop - len(lagged_df)

    print("\nTime-lag shift applied successfully:")
    print(f"- Grouped by        : {LOCATION_COL}")
    print(f"- Forecast horizon  : Day T -> Day T+1")
    print(f"- Dropped tail rows : {dropped_rows}")
    print("Sample rows after shift:")
    print(lagged_df[[LOCATION_COL, TIME_COL, TARGET_COL]].head())

    return lagged_df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch biến đầu vào và giữ lại các cột cần thiết."""
    processed_df = df.copy()

    if LOCATION_COL not in processed_df.columns:
        processed_df[LOCATION_COL] = "Unknown"

    processed_df[TIME_COL] = pd.to_datetime(processed_df[TIME_COL], errors="coerce")
    processed_df = processed_df.dropna(subset=[TIME_COL]).sort_values([LOCATION_COL, TIME_COL]).reset_index(drop=True)

    # Dùng median TÍNH RIÊNG trên phần dữ liệu sẽ thuộc tập TRAIN để điền NaN - xem docstring
    # `compute_train_only_medians()` để biết lý do (ngăn rò rỉ thống kê của tập test vào tập train).
    train_only_medians = compute_train_only_medians(processed_df, FEATURE_COLS)
    for column in FEATURE_COLS:
        processed_df[column] = pd.to_numeric(processed_df[column], errors="coerce")
        processed_df[column] = processed_df[column].fillna(train_only_medians[column])

    processed_df[TARGET_COL] = (
        pd.to_numeric(processed_df[TARGET_COL], errors="coerce").fillna(0).astype(int)
    )

    return processed_df[[LOCATION_COL, TIME_COL, *FEATURE_COLS, TARGET_COL]]


def detect_target_leakage_columns(df: pd.DataFrame) -> list[str]:
    """Phát hiện các cột có dấu hiệu rò rỉ nhãn để loại khỏi feature set."""
    leakage_columns = []
    for column in df.columns:
        normalized_column = str(column).strip().lower()
        if column == TARGET_COL:
            continue
        if any(keyword in normalized_column for keyword in TARGET_LEAKAGE_KEYWORDS):
            leakage_columns.append(column)
    return leakage_columns


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo ma trận feature sạch, chỉ giữ đúng các biến đầu vào hợp lệ."""
    leakage_columns = detect_target_leakage_columns(df)
    if leakage_columns:
        print(
            "Dropped potential leakage columns from feature candidates: "
            + ", ".join(sorted(leakage_columns))
        )

    missing_features = [column for column in FEATURE_COLS if column not in df.columns]
    if missing_features:
        raise ValueError(
            "Thiếu các cột feature bắt buộc: " + ", ".join(missing_features)
        )

    return df.loc[:, FEATURE_COLS].copy()


def chronological_train_test_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """
    Chia train/test theo thời gian cho từng địa phương trước, sau đó mới tạo nhãn T+1.

    Điều này giúp:
    - không dùng feature ngày T để dự đoán nhãn của chính ngày T
    - không để nhãn train "ăn sang" khoảng thời gian test
    - đảm bảo split xảy ra trước mọi bước scaling / balancing
    """
    train_parts = []
    test_parts = []

    for location_name, location_df in df.groupby(LOCATION_COL):
        location_df = location_df.sort_values(TIME_COL).reset_index(drop=True)
        if len(location_df) < 4:
            print(
                f"Skipping {location_name}: only {len(location_df)} rows, "
                "not enough for split + T+1 target shift."
            )
            continue

        train_location_df, test_location_df = train_test_split(
            location_df,
            train_size=train_ratio,
            shuffle=False,
        )

        if len(train_location_df) < 2 or len(test_location_df) < 2:
            print(
                f"Skipping {location_name}: train/test partitions are too small after split."
            )
            continue

        train_parts.append(train_location_df.copy())
        test_parts.append(test_location_df.copy())

    if not train_parts or not test_parts:
        raise ValueError("Không đủ dữ liệu để tạo train/test split theo thời gian.")

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    train_df = apply_time_lag_target_shift(train_df)
    test_df = apply_time_lag_target_shift(test_df)

    if train_df.empty or test_df.empty:
        raise ValueError("Tập train/test rỗng sau khi áp dụng nhãn trễ T+1.")

    X_train = build_feature_frame(train_df)
    y_train = train_df[TARGET_COL].copy()
    X_test = build_feature_frame(test_df)
    y_test = test_df[TARGET_COL].copy()

    print("\nChronological split completed before scaling/balancing:")
    print(f"Train rows after T+1 shift: {len(train_df)}")
    print(f"Test rows after T+1 shift : {len(test_df)}")
    print(f"Train time range          : {train_df[TIME_COL].min()} -> {train_df[TIME_COL].max()}")
    print(f"Test time range           : {test_df[TIME_COL].min()} -> {test_df[TIME_COL].max()}")

    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Chuẩn hóa dữ liệu bằng StandardScaler fit trên train."""
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=FEATURE_COLS,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=FEATURE_COLS,
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled, scaler


def apply_smote_to_training_data(X_train_scaled: pd.DataFrame, y_train: pd.Series):
    """Áp dụng SMOTE chỉ trên tập train để cân bằng 3 lớp."""
    print("\nClass distribution BEFORE SMOTE:")
    print(y_train.value_counts().sort_index())

    min_class_count = int(y_train.value_counts().min())
    if min_class_count < 2:
        raise ValueError("At least one class has fewer than 2 samples. SMOTE cannot be applied.")

    smote_k_neighbors = min(5, min_class_count - 1)
    smote = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    print("\nClass distribution AFTER SMOTE:")
    print(pd.Series(y_train_balanced).value_counts().sort_index())

    X_train_balanced = pd.DataFrame(X_train_balanced, columns=FEATURE_COLS)
    y_train_balanced = pd.Series(y_train_balanced, name=TARGET_COL)
    return X_train_balanced, y_train_balanced


def apply_random_oversampling(X_train_scaled: pd.DataFrame, y_train: pd.Series):
    """Fallback siêu nhẹ khi CTGAN hoặc SMOTE không phù hợp."""
    print("\nClass distribution BEFORE RandomOverSampler:")
    print(y_train.value_counts().sort_index())
    oversampler = RandomOverSampler(random_state=42)
    X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train_scaled, y_train)
    X_train_balanced = pd.DataFrame(X_train_balanced, columns=FEATURE_COLS)
    y_train_balanced = pd.Series(y_train_balanced, name=TARGET_COL)
    print("\nClass distribution AFTER RandomOverSampler:")
    print(y_train_balanced.value_counts().sort_index())
    return X_train_balanced, y_train_balanced


def build_training_snapshot_dataframe(X_values: pd.DataFrame, y_values: pd.Series) -> pd.DataFrame:
    """Ghép feature và target thành một DataFrame để export nhanh cho Streamlit."""
    snapshot_df = X_values.reset_index(drop=True).copy()
    snapshot_df[TARGET_COL] = pd.Series(y_values, name=TARGET_COL).reset_index(drop=True).astype(int)
    return snapshot_df


def summarize_class_distribution(y_values: pd.Series) -> dict[str, int]:
    """Chuẩn hóa value_counts thành dict JSON-friendly."""
    counts = pd.Series(y_values, name=TARGET_COL).value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def export_ctgan_snapshot_csv(snapshot_df: pd.DataFrame, output_path: Path) -> int:
    """Lưu mẫu dữ liệu trước/sau CTGAN để app có thể đọc nhanh."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_size = min(CTGAN_EXPORT_SAMPLE_SIZE, len(snapshot_df))
    if len(snapshot_df) > sample_size:
        export_df = snapshot_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        export_df = snapshot_df.reset_index(drop=True)
    export_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return len(export_df)


def export_ctgan_comparison_artifacts(
    X_before: pd.DataFrame,
    y_before: pd.Series,
    X_after: pd.DataFrame | None = None,
    y_after: pd.Series | None = None,
    method_used: str = "CTGAN",
    status: str = "completed",
    error_detail: str | None = None,
) -> None:
    """
    Export dữ liệu và phân phối lớp trước/sau augmentation cho UI Streamlit.

    `error_detail` (nếu có) là thông điệp lỗi THẬT (kiểu lỗi + message) khiến CTGAN fallback sang
    SMOTE/RandomOverSampler - trước đây lỗi này chỉ in ra console/log server nên người dùng đọc UI
    không biết vì sao luôn fallback; nay lưu kèm vào `ctgan_class_distribution.json` để hiển thị
    thẳng trên Streamlit, không cần SSH vào server đọc log mỗi lần muốn biết nguyên nhân.
    """
    before_df = build_training_snapshot_dataframe(X_before, y_before)
    before_sample_rows = export_ctgan_snapshot_csv(before_df, CTGAN_BEFORE_PATH)

    after_total_rows = None
    after_sample_rows = None
    if X_after is not None and y_after is not None:
        after_df = build_training_snapshot_dataframe(X_after, y_after)
        after_sample_rows = export_ctgan_snapshot_csv(after_df, CTGAN_AFTER_PATH)
        after_total_rows = len(after_df)

    summary_payload = {
        "method_used": method_used,
        "status": status,
        "error_detail": error_detail,
        "target_column": TARGET_COL,
        "feature_columns": FEATURE_COLS,
        "before": {
            "total_rows": len(before_df),
            "sample_rows": before_sample_rows,
            "class_distribution": summarize_class_distribution(y_before),
        },
        "after": {
            "total_rows": after_total_rows,
            "sample_rows": after_sample_rows,
            "class_distribution": summarize_class_distribution(y_after) if y_after is not None else {},
        },
    }
    with CTGAN_DISTRIBUTION_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary_payload, file, indent=2, ensure_ascii=False)


def resolve_balancing_method(balancing_method: str = "auto") -> str:
    """Chọn chiến lược cân bằng dữ liệu an toàn cho pipeline."""
    requested_method = str(balancing_method or "auto").strip().lower()
    if requested_method not in {"auto", "gan", "smote"}:
        raise ValueError("balancing_method phải là một trong: 'auto', 'gan', 'smote'.")

    if requested_method == "auto":
        return "gan" if CTGAN is not None else "smote"
    if requested_method == "gan" and CTGAN is None:
        print("Friendly warning: `ctgan` chưa được cài. Tự động fallback về SMOTE.")
        return "smote"
    return requested_method


def apply_gan_data_augmentation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    epochs: int = CTGAN_EPOCHS,
    max_train_rows_per_class: int = CTGAN_MAX_TRAIN_ROWS_PER_CLASS,
    max_target_rows_per_class: int = CTGAN_MAX_TARGET_ROWS_PER_CLASS,
    batch_size: int = CTGAN_BATCH_SIZE,
):
    """Dùng CTGAN để sinh thêm mẫu cho các lớp thiểu số 1 và 2."""
    if CTGAN is None:
        print("Friendly warning: Không tìm thấy `ctgan`. Fallback về SMOTE.")
        return apply_smote_to_training_data(X_train, y_train)

    print("\nClass distribution BEFORE CTGAN augmentation:")
    print(y_train.value_counts().sort_index())
    export_ctgan_comparison_artifacts(
        X_before=X_train,
        y_before=y_train,
        method_used="CTGAN",
        status="before_exported",
    )

    class_counts = y_train.value_counts().sort_index()
    if class_counts.empty:
        raise ValueError("Tập train rỗng, không thể thực hiện CTGAN augmentation.")

    majority_count = int(class_counts.max())
    target_count = min(majority_count, max_target_rows_per_class)
    augmented_feature_frames = [X_train.reset_index(drop=True).copy()]
    augmented_target_series = [y_train.reset_index(drop=True).copy()]
    generated_total = 0

    for class_label in [1, 2]:
        current_count = int(class_counts.get(class_label, 0))
        deficit = target_count - current_count
        if deficit <= 0:
            continue

        class_features = X_train.loc[y_train == class_label].copy()
        if len(class_features) < 10:
            print(
                f"Friendly warning: Lớp {class_label} chỉ có {len(class_features)} mẫu, "
                "không đủ ổn định cho CTGAN. Fallback về SMOTE."
            )
            X_fallback, y_fallback = apply_smote_to_training_data(X_train, y_train)
            export_ctgan_comparison_artifacts(
                X_before=X_train,
                y_before=y_train,
                X_after=X_fallback,
                y_after=y_fallback,
                method_used="SMOTE_FALLBACK",
                status="fallback_from_small_class",
                error_detail=(
                    f"Lớp {class_label} chỉ có {len(class_features)} mẫu trong tập train (< 10) - "
                    "quá ít để CTGAN học phân phối ổn định."
                ),
            )
            return X_fallback, y_fallback

        if len(class_features) > max_train_rows_per_class:
            class_features = class_features.sample(
                n=max_train_rows_per_class,
                random_state=42,
            ).reset_index(drop=True)

        print(
            f"CTGAN class {class_label}: train_rows={len(class_features)}, "
            f"current_count={current_count}, target_count={target_count}, to_generate={deficit}"
        )

        try:
            gan_model = CTGAN(
                epochs=epochs,
                batch_size=batch_size,
                verbose=False,
            )
            gan_model.fit(class_features)
            synthetic_features = gan_model.sample(deficit)
            synthetic_features = synthetic_features[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
            synthetic_features = synthetic_features.fillna(class_features.median())
            synthetic_targets = pd.Series([class_label] * len(synthetic_features), name=TARGET_COL)

            augmented_feature_frames.append(synthetic_features.reset_index(drop=True))
            augmented_target_series.append(synthetic_targets.reset_index(drop=True))
            generated_total += len(synthetic_features)
            print(
                f"CTGAN generated {len(synthetic_features)} synthetic samples for class {class_label}."
            )
        except MemoryError as exc:
            error_detail = f"MemoryError: {exc}"
            print(
                f"Friendly warning: CTGAN hit MemoryError for class {class_label} ({exc}). "
                "Fallback về RandomOverSampler."
            )
            gc.collect()
            X_fallback, y_fallback = apply_random_oversampling(X_train, y_train)
            export_ctgan_comparison_artifacts(
                X_before=X_train,
                y_before=y_train,
                X_after=X_fallback,
                y_after=y_fallback,
                method_used="RandomOverSampler_FALLBACK",
                status="fallback_from_memory_error",
                error_detail=error_detail,
            )
            return X_fallback, y_fallback
        except Exception as exc:
            error_detail = f"{type(exc).__name__}: {exc}"
            print(f"Friendly warning: CTGAN failed for class {class_label} ({error_detail}). Fallback về SMOTE.")
            print(traceback.format_exc())  # full stack trace ra console/log server để debug sâu hơn nếu cần
            gc.collect()
            X_fallback, y_fallback = apply_smote_to_training_data(X_train, y_train)
            export_ctgan_comparison_artifacts(
                X_before=X_train,
                y_before=y_train,
                X_after=X_fallback,
                y_after=y_fallback,
                method_used="SMOTE_FALLBACK",
                status="fallback_from_exception",
                error_detail=error_detail,
            )
            return X_fallback, y_fallback
        finally:
            gc.collect()

    X_train_balanced = pd.concat(augmented_feature_frames, ignore_index=True)
    y_train_balanced = pd.concat(augmented_target_series, ignore_index=True)

    shuffled_index = np.random.RandomState(42).permutation(len(X_train_balanced))
    X_train_balanced = X_train_balanced.iloc[shuffled_index].reset_index(drop=True)
    y_train_balanced = y_train_balanced.iloc[shuffled_index].reset_index(drop=True)

    print(f"\nCTGAN synthetic rows added: {generated_total}")
    print("Class distribution AFTER CTGAN augmentation:")
    print(y_train_balanced.value_counts().sort_index())
    export_ctgan_comparison_artifacts(
        X_before=X_train,
        y_before=y_train,
        X_after=X_train_balanced,
        y_after=y_train_balanced,
        method_used="CTGAN",
        status="completed",
    )
    gc.collect()
    return X_train_balanced, y_train_balanced


def balance_training_data(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    balancing_method: str = "auto",
):
    """Cân bằng dữ liệu train bằng CTGAN hoặc SMOTE tùy cấu hình."""
    selected_method = resolve_balancing_method(balancing_method)
    print(f"\nSelected balancing method: {selected_method.upper()}")

    if selected_method == "gan":
        return apply_gan_data_augmentation(X_train_scaled, y_train)
    return apply_smote_to_training_data(X_train_scaled, y_train)


def build_daily_modeling_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Gom dữ liệu theo ngày để phục vụ time-series, LSTM và mô hình hybrid."""
    daily_df = df.copy()
    daily_df[DATE_COL] = daily_df[TIME_COL].dt.floor("D")
    daily_df = (
        daily_df.groupby([LOCATION_COL, DATE_COL], as_index=False)
        .agg(
            {
                "Nhiệt_độ_C": "mean",
                "Độ_ẩm_%": "mean",
                "Lượng_mưa_mm": "sum",
                "Độ_ẩm_đất": "mean",
                "Chiều_cao_triều_m": "mean",
                TARGET_COL: "max",
            }
        )
        .rename(columns={DATE_COL: TIME_COL})
        .sort_values([LOCATION_COL, TIME_COL])
        .reset_index(drop=True)
    )
    return daily_df


def round_and_clip_predictions(predictions) -> np.ndarray:
    """Ép dự báo liên tục về nhãn lớp 0/1/2."""
    rounded = np.rint(np.asarray(predictions, dtype=float))
    return np.clip(rounded, min(CLASS_LABELS), max(CLASS_LABELS)).astype(int)


def evaluate_prediction_arrays(
    model_name: str,
    y_true,
    y_pred,
    category: str,
    deployment_compatible: bool,
    evaluation_scope: str,
) -> dict:
    """Đánh giá một mảng dự báo và trả về metrics chuẩn cho JSON."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        target_names=[CLASS_NAME_MAP[label] for label in CLASS_LABELS],
        output_dict=True,
        zero_division=0,
    )
    return {
        "model_name": model_name,
        "category": category,
        "deployment_compatible": deployment_compatible,
        "evaluation_scope": evaluation_scope,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc_ovr_macro": None,
        "classification_report": report,
    }


def safe_compute_roc_auc_ovr_macro(y_true: np.ndarray, y_proba: np.ndarray) -> float | None:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    if y_proba.ndim != 2:
        return None
    if y_proba.shape[1] != len(CLASS_LABELS):
        return None
    try:
        return float(
            roc_auc_score(
                y_true,
                y_proba,
                labels=CLASS_LABELS,
                multi_class="ovr",
                average="macro",
            )
        )
    except Exception:
        return None


def export_multiclass_roc_curve_data(
    best_model_name: str,
    model,
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> Path:
    """Xuất ROC curve data (OvR) cho 3 lớp của best model để frontend vẽ Plotly."""
    output_path = output_dir / "roc_curve_data.json"
    y_true = np.asarray(y_test).astype(int)

    if not hasattr(model, "predict_proba"):
        payload = {
            "status": "unavailable",
            "reason": "model_has_no_predict_proba",
            "model_name": best_model_name,
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        return output_path

    try:
        y_proba = model.predict_proba(X_test_scaled)
    except Exception as exc:
        payload = {
            "status": "unavailable",
            "reason": "predict_proba_failed",
            "model_name": best_model_name,
            "error": str(exc),
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        return output_path

    if y_proba is None:
        payload = {
            "status": "unavailable",
            "reason": "predict_proba_returned_none",
            "model_name": best_model_name,
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        return output_path

    y_proba = np.asarray(y_proba, dtype=float)
    if y_proba.ndim != 2 or y_proba.shape[1] != len(CLASS_LABELS):
        payload = {
            "status": "unavailable",
            "reason": "unexpected_proba_shape",
            "model_name": best_model_name,
            "proba_shape": list(y_proba.shape),
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        return output_path

    y_bin = label_binarize(y_true, classes=CLASS_LABELS)
    if y_bin.shape[1] != len(CLASS_LABELS):
        payload = {
            "status": "unavailable",
            "reason": "binarize_failed",
            "model_name": best_model_name,
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        return output_path

    per_class = {}
    for class_index, class_label in enumerate(CLASS_LABELS):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, class_index], y_proba[:, class_index])
            auc_value = float(roc_auc_score(y_bin[:, class_index], y_proba[:, class_index]))
            per_class[str(class_label)] = {
                "fpr": [float(v) for v in fpr.tolist()],
                "tpr": [float(v) for v in tpr.tolist()],
                "auc": auc_value,
            }
        except Exception:
            per_class[str(class_label)] = {
                "fpr": [],
                "tpr": [],
                "auc": None,
            }

    macro_auc = safe_compute_roc_auc_ovr_macro(y_true, y_proba)
    payload = {
        "status": "ok",
        "model_name": best_model_name,
        "auc_ovr_macro": macro_auc,
        "classes": [int(v) for v in CLASS_LABELS],
        "class_names": {str(k): v for k, v in CLASS_NAME_MAP.items()},
        "curves": per_class,
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    print(f"Saved ROC curve data to: {output_path}")
    return output_path


def export_multiclass_roc_curve_from_proba(
    best_model_name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_dir: Path,
) -> Path:
    """Xuất ROC curve data (OvR) từ y_true + y_proba (dùng cho Keras/DL models)."""
    output_path = output_dir / "roc_curve_data.json"
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_proba.ndim != 2 or y_proba.shape[1] != len(CLASS_LABELS):
        payload = {
            "status": "unavailable",
            "reason": "unexpected_proba_shape",
            "model_name": best_model_name,
            "proba_shape": list(y_proba.shape),
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        return output_path

    y_bin = label_binarize(y_true, classes=CLASS_LABELS)
    per_class = {}
    for class_index, class_label in enumerate(CLASS_LABELS):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, class_index], y_proba[:, class_index])
            auc_value = float(roc_auc_score(y_bin[:, class_index], y_proba[:, class_index]))
            per_class[str(class_label)] = {
                "fpr": [float(v) for v in fpr.tolist()],
                "tpr": [float(v) for v in tpr.tolist()],
                "auc": auc_value,
            }
        except Exception:
            per_class[str(class_label)] = {"fpr": [], "tpr": [], "auc": None}

    payload = {
        "status": "ok",
        "model_name": best_model_name,
        "auc_ovr_macro": safe_compute_roc_auc_ovr_macro(y_true, y_proba),
        "classes": [int(v) for v in CLASS_LABELS],
        "class_names": {str(k): v for k, v in CLASS_NAME_MAP.items()},
        "curves": per_class,
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    print(f"Saved ROC curve data to: {output_path}")
    return output_path


def build_sequence_datasets(
    daily_df: pd.DataFrame,
    window_size: int = SEQUENCE_WINDOW,
    train_ratio: float = 0.8,
):
    """Tạo sliding window theo từng địa phương cho LSTM và mô hình hybrid."""
    train_parts = []
    test_parts = []
    for _, location_df in daily_df.groupby(LOCATION_COL):
        location_df = location_df.sort_values(TIME_COL).reset_index(drop=True)
        if len(location_df) < window_size + 2:
            continue
        split_index = int(len(location_df) * train_ratio)
        split_index = max(window_size + 1, min(split_index, len(location_df) - 1))
        train_parts.append(location_df.iloc[:split_index].copy())
        test_parts.append(location_df.iloc[split_index:].copy())

    if not train_parts or not test_parts:
        raise ValueError("Không đủ dữ liệu chuỗi theo ngày để tạo sequence dataset.")

    train_daily = pd.concat(train_parts, ignore_index=True)

    seq_scaler = StandardScaler()
    seq_scaler.fit(train_daily[FEATURE_COLS])

    X_train_sequences = []
    y_train_sequences = []
    X_test_sequences = []
    y_test_sequences = []

    for _, location_df in daily_df.groupby(LOCATION_COL):
        location_df = location_df.sort_values(TIME_COL).reset_index(drop=True)
        if len(location_df) < window_size + 2:
            continue

        split_index = int(len(location_df) * train_ratio)
        split_index = max(window_size + 1, min(split_index, len(location_df) - 1))

        train_location_df = location_df.iloc[:split_index].copy()
        test_location_df = location_df.iloc[split_index:].copy()

        scaled_train = seq_scaler.transform(train_location_df[FEATURE_COLS])
        train_targets = train_location_df[TARGET_COL].to_numpy()
        for end_idx in range(window_size - 1, len(train_location_df)):
            start_idx = end_idx - window_size + 1
            X_train_sequences.append(scaled_train[start_idx : end_idx + 1])
            y_train_sequences.append(train_targets[end_idx])

        test_context_df = pd.concat(
            [train_location_df.tail(window_size - 1), test_location_df],
            ignore_index=True,
        )
        scaled_test = seq_scaler.transform(test_context_df[FEATURE_COLS])
        test_targets = test_context_df[TARGET_COL].to_numpy()
        for end_idx in range(window_size - 1, len(test_context_df)):
            start_idx = end_idx - window_size + 1
            X_test_sequences.append(scaled_test[start_idx : end_idx + 1])
            y_test_sequences.append(test_targets[end_idx])

    if not X_train_sequences or not X_test_sequences:
        raise ValueError("Không tạo được đủ mẫu sequence cho LSTM.")

    return (
        np.asarray(X_train_sequences, dtype=np.float32),
        np.asarray(y_train_sequences, dtype=np.int32),
        np.asarray(X_test_sequences, dtype=np.float32),
        np.asarray(y_test_sequences, dtype=np.int32),
        seq_scaler,
    )


def build_class_weight_mapping(y_values) -> dict:
    """Tạo class weight đơn giản cho dữ liệu mất cân bằng."""
    y_series = pd.Series(y_values)
    total = len(y_series)
    counts = y_series.value_counts().to_dict()
    return {
        class_label: total / (len(CLASS_LABELS) * counts[class_label])
        for class_label in CLASS_LABELS
        if class_label in counts and counts[class_label] > 0
    }


def build_lstm_classifier(input_shape):
    """Khởi tạo kiến trúc LSTM đơn giản cho phân loại đa lớp."""
    inputs = Input(shape=input_shape)
    encoded = LSTM(32, name="lstm_encoder")(inputs)
    dense_features = Dense(16, activation="relu", name="dense_features")(encoded)
    outputs = Dense(len(CLASS_LABELS), activation="softmax", name="class_output")(dense_features)
    model = Model(inputs=inputs, outputs=outputs, name="flood_lstm_classifier")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_gru_classifier(input_shape):
    inputs = Input(shape=input_shape)
    encoded = GRU(32, name="gru_encoder")(inputs)
    dense_features = Dense(16, activation="relu", name="dense_features")(encoded)
    outputs = Dense(len(CLASS_LABELS), activation="softmax", name="class_output")(dense_features)
    model = Model(inputs=inputs, outputs=outputs, name="flood_gru_classifier")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn1d_classifier(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=32, kernel_size=3, activation="relu", padding="same")(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(len(CLASS_LABELS), activation="softmax", name="class_output")(x)
    model = Model(inputs=inputs, outputs=outputs, name="flood_cnn1d_classifier")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn_lstm_classifier(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=32, kernel_size=3, activation="relu", padding="same")(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(32, name="cnn_lstm_encoder")(x)
    x = Dense(16, activation="relu")(x)
    outputs = Dense(len(CLASS_LABELS), activation="softmax", name="class_output")(x)
    model = Model(inputs=inputs, outputs=outputs, name="flood_cnn_lstm_classifier")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def clear_tensorflow_session() -> None:
    """Giải phóng resource TensorFlow/Keras để tránh giữ session quá lâu trong Streamlit."""
    if K is not None:
        K.clear_session()
    gc.collect()


def log_realtime_update(message: str) -> None:
    """Log chuẩn hóa cho incremental learning."""
    print(f"\n[REALTIME-INCREMENTAL] {message}")


def load_incremental_cursor() -> dict:
    """Đọc cursor để tránh fine-tune lặp lại cùng bản ghi theo ngày."""
    ensure_base_directories()
    if not INCREMENTAL_CURSOR_PATH.exists():
        return {}
    with INCREMENTAL_CURSOR_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_incremental_cursor(cursor: dict) -> None:
    """Lưu cursor incremental."""
    ensure_base_directories()
    with INCREMENTAL_CURSOR_PATH.open("w", encoding="utf-8") as file:
        json.dump(cursor, file, indent=2, ensure_ascii=False)


def normalize_input_new_data(new_data_df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa DataFrame đầu vào incremental để khớp schema pipeline."""
    df = new_data_df.copy()
    if LOCATION_COL not in df.columns:
        df[LOCATION_COL] = "Unknown"
    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    return df


def build_incremental_tabular_dataset(new_data_df: pd.DataFrame) -> pd.DataFrame:
    """Tạo dữ liệu tabular T->T+1 từ các bản ghi mới (hourly->daily->label->shift)."""
    normalized = normalize_input_new_data(new_data_df)
    daily_features = build_daily_feature_dataset(normalized)
    labeled_daily = create_multiclass_flood_label(daily_features)
    processed = preprocess_features(labeled_daily)
    shifted = apply_time_lag_target_shift(processed)
    return shifted


def load_latest_scaler() -> StandardScaler:
    """Nạp scaler đã fit từ lần train đầy đủ gần nhất."""
    scaler_path = LATEST_MODELS_DIR / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError("Không tìm thấy `models/latest/scaler.pkl` để incremental transform.")
    return joblib.load(scaler_path)


def load_latest_best_model():
    """Nạp mô hình tốt nhất đã lưu (`models/latest/best_model.pkl`) để suy luận/dự báo."""
    model_path = LATEST_MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Không tìm thấy `models/latest/best_model.pkl`. Hãy chạy huấn luyện trước.")
    return joblib.load(model_path)


def predict_next_3_days(future_weather_df: pd.DataFrame, model=None, scaler: StandardScaler | None = None) -> pd.DataFrame:
    """
    Dự báo nguy cơ ngập cho 3 ngày tới (Day+1, Day+2, Day+3) từ dữ liệu thời tiết dự báo.

    ----------------------------------------------------------------------------------------------
    GIẢI THÍCH LOGIC "3-DAY ROLLING PREDICTION" (dùng cho báo cáo luận văn):
    ----------------------------------------------------------------------------------------------
    - Hàm này KHÔNG tự gọi API thời tiết. Người dùng tự fetch dữ liệu forecast 3 ngày tới từ
      Open-Meteo Forecast API (`https://api.open-meteo.com/v1/forecast`, tham số `forecast_days`),
      tổng hợp về dạng THEO NGÀY (1 dòng = 1 ngày, giống `build_daily_feature_dataset()` ở trên:
      nhiệt độ/độ ẩm/độ ẩm đất/triều lấy trung bình ngày, lượng mưa lấy tổng ngày), rồi truyền
      DataFrame đó vào tham số `future_weather_df`.
    - Đây là kiểu dự báo "rolling" theo nghĩa: mỗi lần chạy lại (mỗi ngày mới), cửa sổ 3 ngày dự báo
      LUÔN TRƯỢT (roll) về phía trước theo ngày hiện tại - Day+1/Day+2/Day+3 hôm nay sẽ khác với
      Day+1/Day+2/Day+3 của hôm qua, vì luôn được tính lại dựa trên bản tin thời tiết forecast MỚI
      NHẤT tại thời điểm gọi hàm, chứ không phải một dự báo cố định một lần rồi dùng mãi.
    - Với mỗi ngày trong 3 ngày, hàm dùng CÙNG một mô hình đã huấn luyện (`best_model.pkl`) để suy
      luận ĐỘC LẬP (không đệ quy dùng kết quả Day+1 làm đầu vào cho Day+2) - vì đầu vào của mô hình
      là các biến khí tượng - thủy văn của CHÍNH ngày đó (đã có sẵn từ forecast API), không phải
      chuỗi nhãn ngập của các ngày trước, nên không cần/không nên tạo vòng lặp tự hồi quy (autoregressive
      loop) ở đây - cách này tránh hiện tượng "lỗi dồn tích" (error accumulation) thường gặp khi dự
      báo đa bước bằng cách đưa dự đoán của bước trước làm đầu vào cho bước sau.

    Tham số:
        future_weather_df: DataFrame ĐÃ tổng hợp theo ngày, sắp xếp theo thời gian tăng dần, có tối
            thiểu 3 dòng và đủ các cột trong FEATURE_COLS. Nên có thêm cột "Ngày" (datetime/date) để
            gắn nhãn ngày dự báo cho kết quả trả về; nếu thiếu, hàm tự gán Day+1/Day+2/Day+3 kể từ
            ngày chạy hiện tại.
        model: model đã huấn luyện (có `.predict()`, tốt nhất có thêm `.predict_proba()`). Nếu để
            `None`, hàm tự gọi `load_latest_best_model()` để nạp `models/latest/best_model.pkl`.
        scaler: `StandardScaler` đã fit sẵn. Nếu để `None`, hàm tự gọi `load_latest_scaler()`.

    Trả về:
        pd.DataFrame gồm các cột: `Ngày_dự_báo`, `Ngày_thứ` (Day+1/Day+2/Day+3), `Nguy_cơ_ngập`
        (0/1/2), `Nhãn_nguy_cơ` (An toàn/Ngập nhẹ/Ngập nặng), `Xác_suất_ngập_%`.
    """
    if model is None:
        model = load_latest_best_model()
    if scaler is None:
        scaler = load_latest_scaler()

    if len(future_weather_df) < 3:
        raise ValueError(
            f"Cần tối thiểu 3 ngày dữ liệu thời tiết dự báo, chỉ nhận được {len(future_weather_df)} ngày."
        )

    # Chỉ lấy đúng 3 ngày KẾ TIẾP theo thứ tự thời gian (nếu future_weather_df có nhiều hơn 3 dòng,
    # ví dụ forecast_days=7, chỉ 3 dòng đầu tiên - gần hiện tại nhất - được dùng).
    next_3_days_df = future_weather_df.copy()
    if DATE_COL in next_3_days_df.columns:
        next_3_days_df = next_3_days_df.sort_values(DATE_COL)
    next_3_days_df = next_3_days_df.head(3).reset_index(drop=True)

    # Dùng đúng thứ tự cột mà scaler mong đợi (StandardScaler ghi nhớ `feature_names_in_` khi được
    # fit bằng DataFrame có tên cột - xem `split_time_series_data()`), tránh lệch thứ tự cột âm thầm
    # gây sai kết quả dự đoán mà không báo lỗi.
    feature_columns = list(getattr(scaler, "feature_names_in_", FEATURE_COLS))
    missing_columns = [column for column in feature_columns if column not in next_3_days_df.columns]
    if missing_columns:
        raise ValueError(f"Dữ liệu thời tiết đầu vào thiếu các cột bắt buộc: {missing_columns}")

    X_future = next_3_days_df[feature_columns]
    X_future_scaled = pd.DataFrame(scaler.transform(X_future), columns=feature_columns)

    predictions = model.predict(X_future_scaled)
    if hasattr(model, "predict_proba"):
        proba_matrix = model.predict_proba(X_future_scaled)
        # Bài toán đa lớp (0=An toàn/1=Ngập nhẹ/2=Ngập nặng): "xác suất ngập" = P(lớp 1) + P(lớp 2),
        # tức xác suất KHÔNG rơi vào lớp 0 - giữ nhất quán với cách tính đang dùng trong toàn bộ đề tài
        # (xem `get_prediction_and_flood_probability` ở các phiên bản trước của `app.py`).
        if proba_matrix.shape[1] >= 3:
            flood_probabilities = proba_matrix[:, 1:].sum(axis=1)
        elif proba_matrix.shape[1] == 2:
            flood_probabilities = proba_matrix[:, 1]
        else:
            flood_probabilities = (predictions > 0).astype(float)
    else:
        flood_probabilities = (predictions > 0).astype(float)

    class_name_map = {0: "An toàn", 1: "Ngập nhẹ", 2: "Ngập nặng"}
    day_labels = ["Day+1", "Day+2", "Day+3"]
    today = pd.Timestamp.now().normalize()

    result_rows = []
    for offset in range(len(next_3_days_df)):
        prediction = int(predictions[offset])
        forecast_date = next_3_days_df.iloc[offset].get(DATE_COL) if DATE_COL in next_3_days_df.columns else None
        if pd.isna(forecast_date) or forecast_date is None:
            forecast_date = today + pd.Timedelta(days=offset + 1)

        result_rows.append(
            {
                "Ngày_dự_báo": pd.Timestamp(forecast_date).strftime("%Y-%m-%d"),
                "Ngày_thứ": day_labels[offset],
                "Nguy_cơ_ngập": prediction,
                "Nhãn_nguy_cơ": class_name_map.get(prediction, "Không xác định"),
                "Xác_suất_ngập_%": round(float(flood_probabilities[offset]) * 100, 2),
            }
        )

    return pd.DataFrame(result_rows)


def ensure_xgboost_model_available() -> XGBClassifier | None:
    """Bảo đảm có model XGBoost artifact cho incremental update."""
    if BEST_XGBOOST_PATH.exists():
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=INCREMENTAL_XGB_N_ESTIMATORS,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="mlogloss",
            n_jobs=-1,
        )
        model.load_model(str(BEST_XGBOOST_PATH))
        return model
    return None


def save_xgboost_artifact(xgb_model: XGBClassifier, run_dir: Path | None = None) -> None:
    """Lưu XGBoost model để dùng incremental."""
    ensure_base_directories()
    if run_dir is not None:
        xgb_run_path = run_dir / "xgboost_model.json"
        xgb_model.save_model(str(xgb_run_path))
    xgb_model.save_model(str(BEST_XGBOOST_PATH))


def ensure_lstm_model_available() -> tuple[object | None, StandardScaler | None]:
    """Bảo đảm có model LSTM + seq_scaler để incremental update."""
    if load_model is None:
        return None, None
    if not BEST_LSTM_PATH.exists():
        return None, None
    lstm_model = load_model(str(BEST_LSTM_PATH))
    lstm_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    if LSTM_SEQ_SCALER_PATH.exists():
        seq_scaler = joblib.load(LSTM_SEQ_SCALER_PATH)
    else:
        seq_scaler = None
    return lstm_model, seq_scaler


def save_lstm_artifact(lstm_model, seq_scaler: StandardScaler | None, run_dir: Path | None = None) -> None:
    """Lưu LSTM model và seq scaler để dùng incremental."""
    ensure_base_directories()
    if run_dir is not None:
        lstm_run_path = run_dir / "lstm_model.keras"
        lstm_model.save(str(lstm_run_path))
    lstm_model.save(str(BEST_LSTM_PATH))
    if seq_scaler is not None:
        joblib.dump(seq_scaler, LSTM_SEQ_SCALER_PATH)


def build_incremental_lstm_sequences(
    full_daily_df: pd.DataFrame,
    new_daily_df: pd.DataFrame,
    seq_scaler: StandardScaler,
    window_size: int = SEQUENCE_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Tạo sequence samples cho đúng các ngày mới, dùng context từ full history."""
    X_sequences = []
    y_sequences = []
    if new_daily_df.empty:
        return np.empty((0, window_size, len(FEATURE_COLS)), dtype=np.float32), np.empty((0,), dtype=np.int32)

    for location_name, location_new_df in new_daily_df.groupby(LOCATION_COL):
        location_all = full_daily_df.loc[full_daily_df[LOCATION_COL] == location_name].sort_values(TIME_COL).reset_index(drop=True)
        if location_all.empty:
            continue
        time_to_index = {timestamp: idx for idx, timestamp in enumerate(location_all[TIME_COL])}
        for timestamp in location_new_df[TIME_COL].tolist():
            idx = time_to_index.get(timestamp)
            if idx is None or idx < window_size - 1:
                continue
            window_df = location_all.iloc[idx - window_size + 1 : idx + 1]
            scaled_window = seq_scaler.transform(window_df[FEATURE_COLS])
            X_sequences.append(scaled_window)
            y_sequences.append(int(location_all.iloc[idx][TARGET_COL]))

    if not X_sequences:
        return np.empty((0, window_size, len(FEATURE_COLS)), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.asarray(X_sequences, dtype=np.float32), np.asarray(y_sequences, dtype=np.int32)


def incremental_train(new_data_df: pd.DataFrame) -> dict:
    """
    Fine-tune nhanh mô hình tốt nhất (XGBoost, LSTM) dựa trên dữ liệu mới.

    new_data_df có thể là hourly data (có TIME_COL) kèm các cột FEATURE_COLS và LOCATION_COL.
    """
    ensure_base_directories()
    log_realtime_update(f"Real-time Update Triggered: Fine-tuning model on {len(new_data_df)} new records.")

    if new_data_df is None or new_data_df.empty:
        return {"status": "skipped", "reason": "empty_new_data"}

    if not (LATEST_MODELS_DIR / "scaler.pkl").exists():
        log_realtime_update("Không tìm thấy scaler/model artifacts. Trigger full retrain (XGBoost) trước khi incremental update.")
        run_training_pipeline(["XGBoost"])
    scaler = load_latest_scaler()
    cursor = load_incremental_cursor()
    incremental_report = {"status": "completed", "updated_models": [], "skipped_models": []}

    tabular_shifted = build_incremental_tabular_dataset(new_data_df)
    if tabular_shifted.empty:
        incremental_report["status"] = "skipped"
        incremental_report["reason"] = "no_incremental_tabular_rows_after_shift"
    else:
        if "tabular_last_time" in cursor:
            last_time_global = pd.to_datetime(cursor["tabular_last_time"], errors="coerce")
            if pd.notna(last_time_global):
                tabular_shifted = tabular_shifted.loc[tabular_shifted[TIME_COL] > last_time_global].copy()

        if tabular_shifted.empty:
            incremental_report["status"] = "skipped"
            incremental_report["reason"] = "tabular_rows_already_processed"
        else:
            X_new = build_feature_frame(tabular_shifted)
            y_new = tabular_shifted[TARGET_COL].astype(int)
            X_new_scaled = pd.DataFrame(scaler.transform(X_new), columns=FEATURE_COLS)

            xgb_model = ensure_xgboost_model_available()
            if xgb_model is None:
                log_realtime_update("Không tìm thấy `best_xgboost.json`. Trigger full retrain (XGBoost).")
                run_training_pipeline(["XGBoost"])
                xgb_model = ensure_xgboost_model_available()
            if xgb_model is None:
                incremental_report["skipped_models"].append("XGBoost")
            else:
                xgb_model.fit(X_new_scaled, y_new, xgb_model=str(BEST_XGBOOST_PATH))
                save_xgboost_artifact(xgb_model)
                incremental_report["updated_models"].append("XGBoost")

            cursor["tabular_last_time"] = str(pd.to_datetime(tabular_shifted[TIME_COL]).max())
            save_incremental_cursor(cursor)

    lstm_model, seq_scaler = ensure_lstm_model_available()
    if lstm_model is None:
        if Sequential is not None and load_model is not None:
            log_realtime_update("Không tìm thấy `best_lstm_model.keras`. Trigger full retrain (LSTM + XGBoost).")
            available = list_available_models()
            requested = [name for name in ["XGBoost", "LSTM"] if name in available]
            if requested:
                run_training_pipeline(requested)
            lstm_model, seq_scaler = ensure_lstm_model_available()
        if lstm_model is None:
            incremental_report["skipped_models"].append("LSTM")
            return incremental_report

    try:
        full_raw_df = load_and_concatenate_csvs()
        full_daily_features = build_daily_feature_dataset(full_raw_df)
        full_labeled = create_multiclass_flood_label(full_daily_features)
        full_processed = preprocess_features(full_labeled)
        full_daily_df = build_daily_modeling_dataset(full_processed)

        new_norm = normalize_input_new_data(new_data_df)
        new_daily_features = build_daily_feature_dataset(new_norm)
        new_labeled = create_multiclass_flood_label(new_daily_features)
        new_processed = preprocess_features(new_labeled)
        new_daily_df = build_daily_modeling_dataset(new_processed)

        if seq_scaler is None:
            train_daily = full_daily_df.copy()
            seq_scaler = StandardScaler()
            seq_scaler.fit(train_daily[FEATURE_COLS])

        if "lstm_last_time" in cursor:
            last_lstm_time = pd.to_datetime(cursor["lstm_last_time"], errors="coerce")
            if pd.notna(last_lstm_time):
                new_daily_df = new_daily_df.loc[new_daily_df[TIME_COL] > last_lstm_time].copy()

        X_seq, y_seq = build_incremental_lstm_sequences(full_daily_df, new_daily_df, seq_scaler, window_size=SEQUENCE_WINDOW)
        if len(X_seq) == 0:
            incremental_report["skipped_models"].append("LSTM")
            return incremental_report

        lstm_model.fit(
            X_seq,
            y_seq,
            epochs=INCREMENTAL_LSTM_EPOCHS,
            batch_size=32,
            verbose=0,
        )
        save_lstm_artifact(lstm_model, seq_scaler)
        incremental_report["updated_models"].append("LSTM")
        cursor["lstm_last_time"] = str(pd.to_datetime(new_daily_df[TIME_COL]).max())
        save_incremental_cursor(cursor)
        return incremental_report
    finally:
        clear_tensorflow_session()


def train_arima_family_model(model_name: str, daily_df: pd.DataFrame, seasonal: bool) -> dict:
    """Huấn luyện ARIMA/SARIMA trên chuỗi nhãn theo ngày của từng địa phương."""
    y_true_all = []
    y_pred_all = []

    for _, location_df in daily_df.groupby(LOCATION_COL):
        location_df = location_df.sort_values(TIME_COL).reset_index(drop=True)
        if len(location_df) < 12:
            continue

        split_index = max(6, min(int(len(location_df) * 0.8), len(location_df) - 1))
        train_target = location_df.iloc[:split_index][TARGET_COL].astype(float)
        test_target = location_df.iloc[split_index:][TARGET_COL].astype(int)
        if test_target.empty:
            continue

        try:
            if seasonal:
                fitted_model = SARIMAX(
                    train_target,
                    order=(1, 0, 1),
                    seasonal_order=(1, 0, 0, 7),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
            else:
                fitted_model = ARIMA(train_target, order=(2, 0, 1)).fit()

            forecast = fitted_model.forecast(steps=len(test_target))
            y_true_all.extend(test_target.tolist())
            y_pred_all.extend(round_and_clip_predictions(forecast).tolist())
        except Exception:
            continue

    if not y_true_all:
        raise ValueError(f"Không đủ dữ liệu để huấn luyện {model_name}.")

    return evaluate_prediction_arrays(
        model_name=model_name,
        y_true=y_true_all,
        y_pred=y_pred_all,
        category="Time Series",
        deployment_compatible=False,
        evaluation_scope="daily_target_series",
    )


def train_lstm_sequence_model(
    model_name: str,
    daily_df: pd.DataFrame,
) -> tuple[dict, object, StandardScaler, np.ndarray, np.ndarray]:
    """Huấn luyện LSTM classifier trên sequence theo ngày."""
    X_train_seq, y_train_seq, X_test_seq, y_test_seq, seq_scaler = build_sequence_datasets(daily_df)
    lstm_model = build_lstm_classifier((X_train_seq.shape[1], X_train_seq.shape[2]))
    callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    class_weights = build_class_weight_mapping(y_train_seq)
    lstm_model.fit(
        X_train_seq,
        y_train_seq,
        validation_split=0.2,
        epochs=30,
        batch_size=64,
        callbacks=callbacks,
        verbose=0,
        class_weight=class_weights,
    )
    probabilities = lstm_model.predict(X_test_seq, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    metrics = evaluate_prediction_arrays(
        model_name=model_name,
        y_true=y_test_seq,
        y_pred=predictions,
        category="Deep Learning",
        deployment_compatible=False,
        evaluation_scope="daily_sequence",
    )
    metrics["roc_auc_ovr_macro"] = safe_compute_roc_auc_ovr_macro(y_test_seq, probabilities)
    return metrics, lstm_model, seq_scaler, np.asarray(y_test_seq, dtype=int), np.asarray(probabilities, dtype=float)


def train_sequence_deep_model(
    model_name: str,
    daily_df: pd.DataFrame,
    model_builder,
    epochs: int = 30,
    batch_size: int = 64,
) -> tuple[dict, object, StandardScaler, np.ndarray, np.ndarray]:
    """
    Huấn luyện 1 mô hình Deep Learning dạng sequence (GRU / 1D-CNN / CNN-LSTM).

    QUAN TRỌNG (sửa bug so với bản trước): trước đây hàm này chỉ trả về `(metrics, y_test_seq,
    probabilities)` - KHÔNG trả về model đã fit, nên dù model có thắng leaderboard cũng KHÔNG CÓ
    OBJECT NÀO để lưu lại triển khai (bị "mất" ngay sau khi hàm return). Giờ hàm trả về thêm chính
    `model` (Keras model đã huấn luyện) và `seq_scaler` (StandardScaler đã fit riêng cho input dạng
    sequence), để tầng gọi (`train_and_evaluate_models`) có thể giữ lại trong `trained_models` và
    dùng để export nếu model này thắng leaderboard tổng.
    """
    X_train_seq, y_train_seq, X_test_seq, y_test_seq, seq_scaler = build_sequence_datasets(daily_df)
    try:
        model = model_builder((X_train_seq.shape[1], X_train_seq.shape[2]))
        callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
        class_weights = build_class_weight_mapping(y_train_seq)
        model.fit(
            X_train_seq,
            y_train_seq,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
            class_weight=class_weights,
        )
        probabilities = model.predict(X_test_seq, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        metrics = evaluate_prediction_arrays(
            model_name=model_name,
            y_true=y_test_seq,
            y_pred=predictions,
            category="Deep Learning",
            # `deployment_compatible` vẫn được ghi lại làm THÔNG TIN MÔ TẢ (model cần loader Keras
            # riêng, không thể joblib.load() như sklearn) - KHÔNG còn dùng để LOẠI TRỪ model này khỏi
            # việc được chọn làm best model nữa (xem `select_best_model_overall` bên dưới).
            deployment_compatible=False,
            evaluation_scope="daily_sequence",
        )
        metrics["roc_auc_ovr_macro"] = safe_compute_roc_auc_ovr_macro(y_test_seq, probabilities)
        # Trả về `model` TRƯỚC khi `finally` gọi `clear_tensorflow_session()`. Lưu ý kỹ thuật: giá trị
        # trả về đã được Python đánh giá xong (model đã là 1 object cụ thể với trọng số cụ thể) trước
        # khi khối `finally` chạy, nên `K.clear_session()` (chỉ xóa graph/bộ đếm tên layer TOÀN CỤC
        # của Keras) KHÔNG làm mất trọng số hay khả năng `.save()` của riêng object `model` này.
        return metrics, model, seq_scaler, np.asarray(y_test_seq, dtype=int), np.asarray(probabilities, dtype=float)
    finally:
        clear_tensorflow_session()


def train_lstm_xgboost_hybrid_model(
    model_name: str, daily_df: pd.DataFrame
) -> tuple[dict, object, XGBClassifier, StandardScaler, np.ndarray, np.ndarray | None]:
    """
    Huấn luyện hybrid LSTM encoder + XGBoost classifier.

    QUAN TRỌNG (sửa bug so với bản trước): trước đây hàm chỉ trả về `(metrics, y_test_seq,
    hybrid_proba)` - hai thành phần đã huấn luyện là `feature_extractor` (LSTM encoder trích embedding)
    và `hybrid_classifier` (XGBoost học trên embedding đó) đều bị "mất" ngay sau khi hàm return, dù
    model hybrid có thắng leaderboard cũng không có gì để lưu. Giờ hàm trả về đầy đủ cả 2 thành phần
    + `seq_scaler` để tầng gọi có thể lưu lại triển khai khi cần.
    """
    try:
        X_train_seq, y_train_seq, X_test_seq, y_test_seq, seq_scaler = build_sequence_datasets(daily_df)
        lstm_model = build_lstm_classifier((X_train_seq.shape[1], X_train_seq.shape[2]))
        callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
        class_weights = build_class_weight_mapping(y_train_seq)
        lstm_model.fit(
            X_train_seq,
            y_train_seq,
            validation_split=0.2,
            epochs=20,
            batch_size=32,
            callbacks=callbacks,
            verbose=0,
            class_weight=class_weights,
        )

        feature_extractor = Model(
            inputs=lstm_model.input,
            outputs=lstm_model.get_layer("dense_features").output,
        )
        train_embeddings = feature_extractor.predict(X_train_seq, verbose=0)
        test_embeddings = feature_extractor.predict(X_test_seq, verbose=0)

        hybrid_classifier = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="mlogloss",
            n_jobs=-1,
        )
        hybrid_classifier.fit(train_embeddings, y_train_seq)
        hybrid_predictions = hybrid_classifier.predict(test_embeddings)
        metrics = evaluate_prediction_arrays(
            model_name=model_name,
            y_true=y_test_seq,
            y_pred=hybrid_predictions,
            category="Hybrid",
            deployment_compatible=False,
            evaluation_scope="daily_sequence",
        )
        try:
            hybrid_proba = hybrid_classifier.predict_proba(test_embeddings)
            metrics["roc_auc_ovr_macro"] = safe_compute_roc_auc_ovr_macro(y_test_seq, hybrid_proba)
        except Exception:
            hybrid_proba = None
            metrics["roc_auc_ovr_macro"] = None
        # Trả về CẢ 2 thành phần đã fit (feature_extractor + hybrid_classifier) và seq_scaler - xem
        # giải thích về `clear_tensorflow_session()` trong `finally` ở `train_sequence_deep_model()`,
        # nguyên tắc tương tự áp dụng ở đây: object đã được return trước khi session bị clear.
        return (
            metrics,
            feature_extractor,
            hybrid_classifier,
            seq_scaler,
            np.asarray(y_test_seq, dtype=int),
            hybrid_proba,
        )
    finally:
        clear_tensorflow_session()


def build_model_registry() -> dict:
    """Khai báo các mô hình đại diện theo từng nhóm phương pháp."""
    registry = {
        "Linear Regression Threshold": {
            "kind": "tabular_regressor",
            "category": "Statistical",
            "deployment_compatible": True,
            "model": LinearRegression(),
        },
        "Polynomial Regression Threshold": {
            "kind": "tabular_regressor",
            "category": "Statistical",
            "deployment_compatible": True,
            "model": Pipeline(
                [
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("regressor", LinearRegression()),
                ]
            ),
        },
        "Random Forest": {
            "kind": "tabular_classifier",
            "category": "Machine Learning",
            "deployment_compatible": True,
            "model": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        },
        "KNN": {
            "kind": "tabular_classifier",
            "category": "Machine Learning",
            "deployment_compatible": True,
            "model": KNeighborsClassifier(),
        },
        "SVC": {
            "kind": "tabular_classifier",
            "category": "Machine Learning",
            "deployment_compatible": True,
            "model": SVC(probability=True, random_state=42),
        },
        "AdaBoost": {
            "kind": "tabular_classifier",
            "category": "Machine Learning",
            "deployment_compatible": True,
            "model": AdaBoostClassifier(random_state=42),
        },
        "XGBoost": {
            "kind": "tabular_classifier",
            "category": "Machine Learning",
            "deployment_compatible": True,
            "model": XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                n_estimators=250,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                eval_metric="mlogloss",
                n_jobs=-1,
            ),
        },
    }

    if LGBMClassifier is not None:
        registry["LightGBM"] = {
            "kind": "tabular_classifier",
            "category": "Machine Learning",
            "deployment_compatible": True,
            "model": LGBMClassifier(
                objective="multiclass",
                num_class=3,
                n_estimators=250,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
        }

    if CatBoostClassifier is not None:
        registry["CatBoost"] = {
            "kind": "tabular_classifier",
            "category": "Machine Learning",
            "deployment_compatible": True,
            "model": CatBoostClassifier(
                loss_function="MultiClass",
                iterations=250,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0,
            ),
        }

    if ARIMA is not None:
        registry["ARIMA"] = {
            "kind": "time_series_arima",
            "category": "Time Series",
            "deployment_compatible": False,
        }

    if SARIMAX is not None:
        registry["SARIMA"] = {
            "kind": "time_series_sarima",
            "category": "Time Series",
            "deployment_compatible": False,
        }

    if Sequential is not None and Model is not None and Input is not None:
        registry["LSTM"] = {
            "kind": "lstm_sequence",
            "category": "Deep Learning",
            "deployment_compatible": False,
        }
        if GRU is not None:
            registry["GRU"] = {
                "kind": "gru_sequence",
                "category": "Deep Learning",
                "deployment_compatible": False,
            }
        if Conv1D is not None and MaxPooling1D is not None and Flatten is not None:
            registry["1D-CNN"] = {
                "kind": "cnn1d_sequence",
                "category": "Deep Learning",
                "deployment_compatible": False,
            }
            registry["CNN-LSTM"] = {
                "kind": "cnn_lstm_sequence",
                "category": "Deep Learning",
                "deployment_compatible": False,
            }
        registry["LSTM + XGBoost Hybrid"] = {
            "kind": "lstm_xgboost_hybrid",
            "category": "Hybrid",
            "deployment_compatible": False,
        }

    return registry


def filter_selected_models(model_registry: dict, selected_models_list: list[str]) -> dict:
    """Lọc dictionary model theo danh sách người dùng chọn từ UI."""
    if not selected_models_list:
        raise ValueError("Danh sách mô hình được chọn đang rỗng.")

    invalid_models = [name for name in selected_models_list if name not in model_registry]
    if invalid_models:
        raise ValueError(f"Các mô hình không hợp lệ: {', '.join(invalid_models)}")

    return {name: model_registry[name] for name in selected_models_list}


def train_and_evaluate_models(
    selected_models: dict,
    X_train_balanced: pd.DataFrame,
    y_train_balanced: pd.Series,
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series,
    daily_df: pd.DataFrame,
    balancing_method_used: str,
):
    """Huấn luyện và đánh giá mô hình theo từng nhóm phương pháp."""
    trained_models = {}
    evaluation_results = {}
    roc_cache: dict[str, dict[str, np.ndarray]] = {}

    print(f"\n=== TRAINING {len(selected_models)} SELECTED MODELS ACROSS MULTIPLE CATEGORIES ===")
    for index, (model_name, model_config) in enumerate(selected_models.items(), start=1):
        print(f"\n[{index:02d}/{len(selected_models):02d}] Training {model_name}...")
        model_kind = model_config["kind"]
        category = model_config["category"]
        deployment_compatible = model_config["deployment_compatible"]

        if model_kind == "tabular_classifier":
            model = model_config["model"]
            model.fit(X_train_balanced, y_train_balanced)
            trained_models[model_name] = model
            predictions = model.predict(X_test_scaled)
            metrics = evaluate_prediction_arrays(
                model_name=model_name,
                y_true=y_test,
                y_pred=predictions,
                category=category,
                deployment_compatible=deployment_compatible,
                evaluation_scope="daily_t_plus_1_tabular",
            )
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X_test_scaled)
                    metrics["roc_auc_ovr_macro"] = safe_compute_roc_auc_ovr_macro(y_test, proba)
                except Exception:
                    metrics["roc_auc_ovr_macro"] = None
        elif model_kind == "tabular_regressor":
            model = model_config["model"]
            model.fit(X_train_balanced, y_train_balanced.astype(float))
            trained_models[model_name] = model
            predictions = round_and_clip_predictions(model.predict(X_test_scaled))
            metrics = evaluate_prediction_arrays(
                model_name=model_name,
                y_true=y_test,
                y_pred=predictions,
                category=category,
                deployment_compatible=deployment_compatible,
                evaluation_scope="daily_t_plus_1_tabular",
            )
        elif model_kind == "time_series_arima":
            metrics = train_arima_family_model(model_name, daily_df, seasonal=False)
        elif model_kind == "time_series_sarima":
            metrics = train_arima_family_model(model_name, daily_df, seasonal=True)
        elif model_kind == "lstm_sequence":
            metrics, lstm_model, seq_scaler, y_true_seq, y_proba_seq = train_lstm_sequence_model(model_name, daily_df)
            trained_models[model_name] = {
                "model": lstm_model,
                "seq_scaler": seq_scaler,
            }
            roc_cache[model_name] = {"y_true": y_true_seq, "y_proba": y_proba_seq}
        elif model_kind == "gru_sequence":
            metrics, seq_model, seq_scaler, y_true_seq, y_proba_seq = train_sequence_deep_model(
                model_name=model_name,
                daily_df=daily_df,
                model_builder=build_gru_classifier,
                epochs=30,
                batch_size=64,
            )
            # Giữ lại model Keras + seq_scaler đã fit trong `trained_models`, CÙNG cấu trúc dict
            # {"model":..., "seq_scaler":...} như "LSTM" (kind=lstm_sequence) đã dùng từ trước, để
            # tầng export phía sau xử lý đồng nhất cho mọi model dạng sequence (LSTM/GRU/CNN...).
            trained_models[model_name] = {"model": seq_model, "seq_scaler": seq_scaler}
            roc_cache[model_name] = {"y_true": y_true_seq, "y_proba": y_proba_seq}
        elif model_kind == "cnn1d_sequence":
            metrics, seq_model, seq_scaler, y_true_seq, y_proba_seq = train_sequence_deep_model(
                model_name=model_name,
                daily_df=daily_df,
                model_builder=build_cnn1d_classifier,
                epochs=30,
                batch_size=64,
            )
            trained_models[model_name] = {"model": seq_model, "seq_scaler": seq_scaler}
            roc_cache[model_name] = {"y_true": y_true_seq, "y_proba": y_proba_seq}
        elif model_kind == "cnn_lstm_sequence":
            metrics, seq_model, seq_scaler, y_true_seq, y_proba_seq = train_sequence_deep_model(
                model_name=model_name,
                daily_df=daily_df,
                model_builder=build_cnn_lstm_classifier,
                epochs=30,
                batch_size=64,
            )
            trained_models[model_name] = {"model": seq_model, "seq_scaler": seq_scaler}
            roc_cache[model_name] = {"y_true": y_true_seq, "y_proba": y_proba_seq}
        elif model_kind == "lstm_xgboost_hybrid":
            (
                metrics,
                feature_extractor,
                hybrid_classifier,
                hybrid_seq_scaler,
                y_true_seq,
                y_proba_seq,
            ) = train_lstm_xgboost_hybrid_model(model_name, daily_df)
            # Hybrid có 2 thành phần (LSTM feature_extractor + XGBoost classifier) nên cấu trúc dict
            # lưu trong `trained_models` khác với sequence thuần - tầng export sẽ nhận diện qua các
            # key "feature_extractor"/"classifier" để biết đây là hybrid, cần lưu 2 file riêng biệt.
            trained_models[model_name] = {
                "feature_extractor": feature_extractor,
                "classifier": hybrid_classifier,
                "seq_scaler": hybrid_seq_scaler,
            }
            if y_proba_seq is not None:
                roc_cache[model_name] = {"y_true": y_true_seq, "y_proba": np.asarray(y_proba_seq, dtype=float)}
        else:
            raise ValueError(f"Unsupported model kind: {model_kind}")

        evaluation_results[model_name] = metrics
        evaluation_results[model_name]["balancing_method"] = balancing_method_used
        print(
            f"Category={metrics['category']} | "
            f"F1-Macro={metrics['f1_macro']:.4f} | "
            f"Accuracy={metrics['accuracy']:.4f}"
        )

    return trained_models, evaluation_results, roc_cache


def build_leaderboard_dataframe(evaluation_results: dict) -> pd.DataFrame:
    """Tạo bảng xếp hạng mô hình theo Macro F1 giảm dần."""
    rows = []
    for model_name, metrics in evaluation_results.items():
        rows.append(
            {
                "RankCandidate": model_name,
                "Model": model_name,
                "Category": metrics.get("category", "Unknown"),
                "Scope": metrics.get("evaluation_scope", "Unknown"),
                "Accuracy": metrics["accuracy"],
                "Precision(Macro)": metrics["precision_macro"],
                "Recall(Macro)": metrics["recall_macro"],
                "F1(Macro)": metrics["f1_macro"],
                "ROC_AUC_OvR_Macro": metrics.get("roc_auc_ovr_macro"),
            }
        )

    leaderboard_df = pd.DataFrame(rows).sort_values(
        by=["F1(Macro)", "Accuracy", "Precision(Macro)"],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_df.insert(0, "Rank", leaderboard_df.index + 1)
    return leaderboard_df.drop(columns=["RankCandidate"])


def save_evaluation_metrics(evaluation_results: dict, output_dir: Path) -> Path:
    """Lưu metrics của các mô hình vào thư mục chỉ định."""
    metrics_path = output_dir / "evaluation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(evaluation_results, file, indent=4, ensure_ascii=False)

    print(f"\nSaved evaluation metrics to: {metrics_path}")
    return metrics_path


def save_leaderboard_csv(leaderboard_df: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / "leaderboard.csv"
    leaderboard_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved leaderboard csv to: {output_path}")
    return output_path


def classify_model_deployment_type(model_name: str, evaluation_results: dict, trained_models: dict) -> str:
    """
    Xác định "model_type" triển khai của 1 model dựa trên CẤU TRÚC OBJECT thực tế đã lưu trong
    `trained_models`, KHÔNG dựa vào cờ `deployment_compatible` tĩnh khai báo sẵn trong registry nữa
    (đó chính là nguồn gốc bug cũ: cờ tĩnh này được set cứng `False` cho mọi Deep Learning/Hybrid,
    bất kể model thực tế mạnh hay yếu, khiến các model tốt nhất leaderboard không bao giờ được chọn).

    Trả về một trong: "sklearn_tabular" | "keras_sequence" | "hybrid_lstm_xgboost" | "unsupported".
    "unsupported" dùng cho các model không có MỘT object duy nhất để lưu triển khai (ví dụ ARIMA/
    SARIMA - được fit RIÊNG cho từng địa phương/từng fold trong vòng lặp, không tồn tại "một model"
    đại diện chung để đóng gói - đây là giới hạn KIẾN TRÚC thật sự, khác với bug lọc nhầm ở trên).
    """
    candidate = trained_models.get(model_name)
    if isinstance(candidate, dict) and "feature_extractor" in candidate and "classifier" in candidate:
        return "hybrid_lstm_xgboost"
    if isinstance(candidate, dict) and "model" in candidate and "seq_scaler" in candidate:
        return "keras_sequence"
    if candidate is not None and hasattr(candidate, "predict"):
        return "sklearn_tabular"
    return "unsupported"


def select_best_model_overall(evaluation_results: dict, trained_models: dict) -> tuple[str, str]:
    """
    Chọn model TỐT NHẤT TUYỆT ĐỐI theo leaderboard (Macro F1-score), KHÔNG loại trừ theo nhóm
    phương pháp (Machine Learning / Deep Learning / Hybrid) - đúng yêu cầu: mô hình xếp Rank #1
    trên leaderboard tổng phải luôn là mô hình được chọn triển khai, bất kể loại mô hình.

    Ngoại lệ DUY NHẤT: nếu #1 leaderboard là một model KHÔNG THỂ đóng gói thành 1 artifact triển
    khai được (ví dụ ARIMA/SARIMA - xem `classify_model_deployment_type`), hàm sẽ in cảnh báo rõ
    ràng và lùi xuống ứng viên xếp hạng kế tiếp có thể triển khai được - đây là giới hạn kỹ thuật
    khách quan (không có "một model" để lưu), khác hẳn với việc CỐ TÌNH lọc bỏ Deep Learning/Hybrid
    như logic cũ.
    """
    ranked_model_names = sorted(
        evaluation_results.keys(),
        key=lambda name: float(evaluation_results[name].get("f1_macro", 0.0)),
        reverse=True,
    )
    if not ranked_model_names:
        raise ValueError("Không có mô hình nào được huấn luyện để chọn best model.")

    for rank, model_name in enumerate(ranked_model_names, start=1):
        deployment_type = classify_model_deployment_type(model_name, evaluation_results, trained_models)
        if deployment_type != "unsupported":
            if rank > 1:
                print(
                    f"\n⚠️ Lưu ý: Rank #1 leaderboard là "
                    f"'{ranked_model_names[0]}' nhưng không thể đóng gói thành 1 artifact triển khai "
                    f"(model dạng {evaluation_results[ranked_model_names[0]].get('category')} được "
                    "fit riêng theo từng địa phương/fold, không có object đại diện chung). "
                    f"-> Chọn Rank #{rank}: '{model_name}' (F1-Macro="
                    f"{evaluation_results[model_name]['f1_macro']:.4f}) làm best model triển khai."
                )
            print(
                f"\n✅ Best model được chọn triển khai: '{model_name}' "
                f"(Rank #{rank}/{len(ranked_model_names)} | Category="
                f"{evaluation_results[model_name].get('category')} | F1-Macro="
                f"{evaluation_results[model_name]['f1_macro']:.4f} | model_type={deployment_type})"
            )
            return model_name, deployment_type

    raise ValueError(
        "Không có model nào trong leaderboard có thể đóng gói thành artifact triển khai được "
        "(toàn bộ đều thuộc nhóm không hỗ trợ export như ARIMA/SARIMA)."
    )


def export_deployment_artifacts(
    best_model_name: str,
    deployment_type: str,
    trained_models: dict,
    evaluation_results: dict,
    tabular_scaler: StandardScaler,
    run_dir: Path,
) -> dict:
    """
    CƠ CHẾ LƯU VẠN NĂNG (Universal Saving Mechanism): đóng gói best model theo ĐÚNG định dạng gốc
    của từng loại mô hình, thay vì ép mọi loại model đều phải là 1 file `best_model.pkl` bằng joblib
    (joblib/pickle CHỈ phù hợp cho object Python thuần như sklearn/XGBoost - Keras model KHÔNG nên
    pickle trực tiếp, phải dùng `model.save()` định dạng `.keras` gốc của TensorFlow để đảm bảo nạp
    lại đúng kiến trúc + trọng số + optimizer state).

    3 nhánh lưu tương ứng 3 model_type:
      - "sklearn_tabular": joblib.dump() -> best_model.pkl (như cũ, không đổi hành vi cho nhóm này).
      - "keras_sequence" (LSTM/GRU/1D-CNN/CNN-LSTM): model.save() -> best_model.keras (định dạng
        Keras v3 gốc, tự chứa kiến trúc mạng, gọi lại bằng `tensorflow.keras.models.load_model()`).
      - "hybrid_lstm_xgboost": lưu RIÊNG 2 file cho 2 thành phần: LSTM feature_extractor ->
        best_model_feature_extractor.keras, và đầu phân loại XGBoost -> best_model_xgb_head.json
        (dùng `.save_model()` gốc của XGBoost, KHÔNG joblib, để tương thích ngược version-safe hơn).

    Mỗi nhánh cũng lưu kèm 1 scaler RIÊNG khớp đúng loại input mà model đó cần (scaler tabular
    theo dòng-ngày-đơn cho sklearn, seq_scaler theo sliding-window cho sequence/hybrid) - vì 2 loại
    scaler này được fit trên 2 kiểu dữ liệu khác nhau, KHÔNG được dùng lẫn cho nhau.

    Trả về dict `artifacts` (tên file -> Path) để hàm gọi ghi tiếp vào `deployment_config.json`.
    """
    artifacts: dict[str, Path] = {}

    if deployment_type == "sklearn_tabular":
        model = trained_models[best_model_name]
        model_path = run_dir / "best_model.pkl"
        scaler_path = run_dir / "scaler.pkl"
        joblib.dump(model, model_path)
        joblib.dump(tabular_scaler, scaler_path)
        artifacts["model_path"] = model_path
        artifacts["scaler_path"] = scaler_path
        print(f"[Universal Save] sklearn_tabular -> {model_path.name} (joblib)")

    elif deployment_type == "keras_sequence":
        bundle = trained_models[best_model_name]
        model_path = run_dir / "best_model.keras"
        scaler_path = run_dir / "scaler.pkl"
        bundle["model"].save(str(model_path))
        joblib.dump(bundle["seq_scaler"], scaler_path)
        artifacts["model_path"] = model_path
        artifacts["scaler_path"] = scaler_path
        print(f"[Universal Save] keras_sequence -> {model_path.name} (Keras native .save())")

    elif deployment_type == "hybrid_lstm_xgboost":
        bundle = trained_models[best_model_name]
        feature_extractor_path = run_dir / "best_model_feature_extractor.keras"
        classifier_path = run_dir / "best_model_xgb_head.json"
        scaler_path = run_dir / "scaler.pkl"
        bundle["feature_extractor"].save(str(feature_extractor_path))
        bundle["classifier"].save_model(str(classifier_path))
        joblib.dump(bundle["seq_scaler"], scaler_path)
        artifacts["feature_extractor_path"] = feature_extractor_path
        artifacts["classifier_path"] = classifier_path
        artifacts["scaler_path"] = scaler_path
        print(
            f"[Universal Save] hybrid_lstm_xgboost -> {feature_extractor_path.name} + "
            f"{classifier_path.name} (Keras + XGBoost native save)"
        )

    else:
        raise ValueError(f"Không hỗ trợ export cho deployment_type='{deployment_type}'.")

    return artifacts


def save_deployment_config(
    best_model_name: str,
    deployment_type: str,
    evaluation_results: dict,
    artifacts: dict,
    output_dir: Path,
) -> Path:
    """
    Sinh file `deployment_config.json` - "bản đồ chỉ dẫn" để `app.py` biết CHÍNH XÁC cách nạp lại
    best model của lần huấn luyện này, KHÔNG cần đoán hay hardcode theo tên file cố định như trước
    (trước đây app.py luôn giả định `best_model.pkl` tồn tại và nạp bằng joblib - giả định này SAI
    ngay khi best model là Deep Learning/Hybrid). App.py chỉ cần đọc đúng 1 file JSON này để biết:
    (1) model tên gì, (2) thuộc model_type nào -> dùng loader tương ứng (joblib / Keras load_model /
    cả hai cho hybrid), (3) đường dẫn chính xác tới từng file artifact liên quan.
    """
    metrics = evaluation_results[best_model_name]
    config = {
        "model_name": best_model_name,
        "model_type": deployment_type,
        "category": metrics.get("category"),
        "f1_macro": metrics.get("f1_macro"),
        "accuracy": metrics.get("accuracy"),
        "window_size": SEQUENCE_WINDOW if deployment_type in {"keras_sequence", "hybrid_lstm_xgboost"} else None,
        "feature_cols": FEATURE_COLS,
        "class_labels": CLASS_LABELS,
        "class_name_map": CLASS_NAME_MAP,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        # File name tương đối trong CÙNG thư mục (models/latest/ hoặc models/run_<ts>/) - app.py chỉ
        # cần nối với thư mục đang đọc, không phụ thuộc đường dẫn tuyệt đối lúc train.
        "artifacts": {key: Path(value).name for key, value in artifacts.items()},
    }

    config_path = output_dir / "deployment_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=4, ensure_ascii=False)

    print(f"Saved deployment config to: {config_path}")
    return config_path


def save_incremental_candidate_artifacts(trained_models: dict, run_dir: Path) -> None:
    """Lưu các model ứng viên cho incremental (XGBoost, LSTM) vào latest để fine-tune nhanh."""
    if "XGBoost" in trained_models:
        try:
            save_xgboost_artifact(trained_models["XGBoost"], run_dir=run_dir)
            print(f"Saved XGBoost incremental artifact to: {BEST_XGBOOST_PATH}")
        except Exception as exc:
            print(f"Friendly warning: Không thể lưu XGBoost incremental artifact ({exc}).")

    if "LSTM" in trained_models:
        bundle = trained_models.get("LSTM")
        if isinstance(bundle, dict) and bundle.get("model") is not None:
            try:
                save_lstm_artifact(bundle["model"], bundle.get("seq_scaler"), run_dir=run_dir)
                print(f"Saved LSTM incremental artifact to: {BEST_LSTM_PATH}")
            except Exception as exc:
                print(f"Friendly warning: Không thể lưu LSTM incremental artifact ({exc}).")


def build_confusion_matrix_from_labels(y_true, y_pred, output_dir: Path) -> Path:
    """
    Vẽ confusion matrix trực tiếp từ cặp (y_true, y_pred) - dùng CHUNG được cho MỌI loại best model
    (sklearn_tabular / keras_sequence / hybrid_lstm_xgboost), vì dù kiến trúc model khác nhau, kết
    quả cuối cùng luôn quy về 1 cặp nhãn thật/nhãn dự đoán trên cùng thang 3 lớp (0/1/2). Nhờ vậy
    hàm vẽ confusion matrix KHÔNG còn phụ thuộc vào việc gọi `model.predict(X_test_scaled_dataframe)`
    (chỉ đúng cho sklearn) - tránh crash khi best model là Deep Learning/Hybrid (input của chúng là
    sequence 3 chiều, không phải DataFrame 2 chiều như X_test_scaled của nhánh tabular).
    """
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[CLASS_NAME_MAP[label] for label in CLASS_LABELS],
        yticklabels=[CLASS_NAME_MAP[label] for label in CLASS_LABELS],
    )
    plt.title("Confusion Matrix - Best Model")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    output_path = output_dir / "confusion_matrix.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to: {output_path}")
    return output_path


def plot_confusion_matrix(best_model, X_test_scaled: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> Path:
    """Vẽ confusion matrix cho best model dạng sklearn_tabular (giữ nguyên hành vi cũ)."""
    y_pred = best_model.predict(X_test_scaled)
    return build_confusion_matrix_from_labels(y_test, y_pred, output_dir)


def extract_feature_importance(best_model, X_test_scaled: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Trích xuất feature importance theo nhiều kiểu mô hình khác nhau."""
    if hasattr(best_model, "feature_importances_"):
        importance_values = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        coef_values = best_model.coef_
        if getattr(coef_values, "ndim", 1) > 1:
            importance_values = abs(coef_values).mean(axis=0)
        else:
            importance_values = abs(coef_values)
    else:
        sample_size = min(5000, len(X_test_scaled))
        X_sample = X_test_scaled.iloc[:sample_size]
        y_sample = y_test.iloc[:sample_size]
        permutation_result = permutation_importance(
            best_model,
            X_sample,
            y_sample,
            n_repeats=5,
            random_state=42,
            scoring="f1_macro",
            n_jobs=-1,
        )
        importance_values = permutation_result.importances_mean

    importance_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLS,
            "Importance": importance_values,
        }
    ).sort_values(by="Importance", ascending=False)
    return importance_df


def plot_feature_importance(best_model, X_test_scaled: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> Path:
    """Vẽ và lưu feature importance cho mô hình tốt nhất."""
    importance_df = extract_feature_importance(best_model, X_test_scaled, y_test)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature", hue="Feature", dodge=False, legend=False)
    plt.title("Feature Importance - Best Model")
    plt.tight_layout()

    output_path = output_dir / "feature_importance.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance plot to: {output_path}")
    return output_path


def print_leaderboard(leaderboard_df: pd.DataFrame) -> None:
    """In bảng xếp hạng đẹp ra terminal."""
    print("\n=== MODEL LEADERBOARD (SORTED BY MACRO F1-SCORE) ===")
    print(leaderboard_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision(Macro)": "{:.4f}".format,
            "Recall(Macro)": "{:.4f}".format,
            "F1(Macro)": "{:.4f}".format,
        },
    ))


def copy_artifact_to_latest(source_path: Path, target_name: str) -> Path:
    """Ghi đè artifact mới nhất sang `models/latest/` để frontend luôn đọc 1 nơi cố định."""
    destination = LATEST_MODELS_DIR / target_name
    shutil.copy2(source_path, destination)
    print(f"Synced latest artifact: {destination}")
    return destination


def copy_artifact_to_plots(source_path: Path, target_name: str) -> Path:
    """Đồng bộ artifact sang `plots/` để tránh tồn tại file cũ gây hiểu nhầm."""
    destination = PLOTS_DIR / target_name
    shutil.copy2(source_path, destination)
    print(f"Synced plot artifact  : {destination}")
    return destination


def run_training_pipeline(selected_models_list: list[str], balancing_method: str = "auto"):
    """
    Chạy pipeline huấn luyện theo danh sách mô hình được chọn từ UI.

    Kết quả mỗi lần chạy sẽ được:
    - lưu versioned vào `models/run_<timestamp>/`
    - đồng bộ bản mới nhất vào `models/latest/`
    """
    print("=== STARTING VERSIONED FLOOD CLASSIFICATION PIPELINE ===")
    validate_optional_dependencies()
    ensure_base_directories()

    model_registry = build_model_registry()
    selected_models = filter_selected_models(model_registry, selected_models_list)

    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = MODELS_DIR / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_and_concatenate_csvs()
    daily_feature_df = build_daily_feature_dataset(raw_df)
    labeled_daily_df = create_multiclass_flood_label(daily_feature_df)
    modeling_df = preprocess_features(labeled_daily_df)
    daily_df = build_daily_modeling_dataset(modeling_df)

    X_train, X_test, y_train, y_test = chronological_train_test_split(
        modeling_df,
        train_ratio=0.8,
    )
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    balancing_method_used = resolve_balancing_method(balancing_method)
    X_train_balanced, y_train_balanced = balance_training_data(
        X_train_scaled,
        y_train,
        balancing_method=balancing_method_used,
    )

    trained_models, evaluation_results, roc_cache = train_and_evaluate_models(
        selected_models,
        X_train_balanced,
        y_train_balanced,
        X_test_scaled,
        y_test,
        daily_df,
        balancing_method_used,
    )
    save_incremental_candidate_artifacts(trained_models, run_dir=run_dir)

    leaderboard_df = build_leaderboard_dataframe(evaluation_results)

    # ============================================================================================
    # MODEL SELECTION AND EXPORT (ĐÃ SỬA BUG "ưu tiên ngầm cho model tabular")
    # ============================================================================================
    # TRƯỚC ĐÂY: `select_best_model()` chỉ xét các model có cờ tĩnh `deployment_compatible=True`
    # (luôn là False cho MỌI Deep Learning/Hybrid trong `build_model_registry()`), nên dù GRU/Hybrid
    # đạt F1-Macro > 0.92 (hạng #1 thật sự trên leaderboard), pipeline vẫn luôn chọn model tabular
    # yếu hơn nhiều (vd. Random Forest F1=0.50) làm "best model" triển khai - đây chính là bug logic
    # đã báo cáo. BÂY GIỜ: `select_best_model_overall()` LUÔN chọn đúng Rank #1 theo leaderboard tổng,
    # bất kể model đó thuộc nhóm nào - chỉ lùi hạng khi #1 tuyệt đối KHÔNG THỂ đóng gói triển khai
    # được về mặt kỹ thuật (ARIMA/SARIMA, xem `classify_model_deployment_type`), và luôn IN RÕ lý do
    # ra log khi việc lùi hạng xảy ra, thay vì âm thầm lọc bỏ như logic cũ.
    best_model_name, deployment_type = select_best_model_overall(evaluation_results, trained_models)

    print_leaderboard(leaderboard_df)
    leaderboard_path = save_leaderboard_csv(leaderboard_df, run_dir)
    metrics_path = save_evaluation_metrics(evaluation_results, run_dir)

    # CƠ CHẾ LƯU VẠN NĂNG: dispatch theo đúng model_type thực tế (sklearn_tabular / keras_sequence /
    # hybrid_lstm_xgboost) - xem docstring chi tiết trong `export_deployment_artifacts()`.
    artifacts = export_deployment_artifacts(
        best_model_name=best_model_name,
        deployment_type=deployment_type,
        trained_models=trained_models,
        evaluation_results=evaluation_results,
        tabular_scaler=scaler,
        run_dir=run_dir,
    )
    deployment_config_path = save_deployment_config(
        best_model_name=best_model_name,
        deployment_type=deployment_type,
        evaluation_results=evaluation_results,
        artifacts=artifacts,
        output_dir=run_dir,
    )

    # ---- Confusion matrix: dùng chung 1 hàm cho mọi loại model (xem build_confusion_matrix_from_labels) ----
    if deployment_type == "sklearn_tabular":
        confusion_matrix_path = plot_confusion_matrix(trained_models[best_model_name], X_test_scaled, y_test, run_dir)
        feature_importance_path = plot_feature_importance(trained_models[best_model_name], X_test_scaled, y_test, run_dir)
    elif best_model_name in roc_cache:
        # keras_sequence / hybrid_lstm_xgboost: không có DataFrame tabular để `.predict()` trực tiếp,
        # nhưng đã có sẵn (y_true, y_proba) từ lúc đánh giá trên tập test dạng sequence -> suy ra
        # y_pred = argmax(y_proba) để vẽ confusion matrix, tái sử dụng đúng số liệu đã đánh giá,
        # không cần chạy suy luận lại lần 2.
        y_true_for_cm = roc_cache[best_model_name]["y_true"]
        y_pred_for_cm = np.argmax(roc_cache[best_model_name]["y_proba"], axis=1)
        confusion_matrix_path = build_confusion_matrix_from_labels(y_true_for_cm, y_pred_for_cm, run_dir)
        # Feature importance kiểu "tabular" (feature_importances_/coef_/permutation) không có ý nghĩa
        # trực tiếp cho input dạng sequence (mỗi feature xuất hiện lặp lại qua nhiều bước thời gian
        # trong sliding window) - bỏ qua thay vì tính sai, và ghi rõ lý do ra log/artifact.
        feature_importance_path = None
        print(
            f"[Info] Bỏ qua feature_importance.png cho model_type='{deployment_type}' - biểu đồ "
            "feature importance kiểu tabular không áp dụng trực tiếp cho input dạng sequence."
        )
    else:
        confusion_matrix_path = None
        feature_importance_path = None

    roc_curve_path = None
    if best_model_name in roc_cache:
        roc_curve_path = export_multiclass_roc_curve_from_proba(
            best_model_name,
            roc_cache[best_model_name]["y_true"],
            roc_cache[best_model_name]["y_proba"],
            run_dir,
        )
    elif deployment_type == "sklearn_tabular":
        roc_curve_path = export_multiclass_roc_curve_data(
            best_model_name,
            trained_models[best_model_name],
            X_test_scaled,
            y_test,
            run_dir,
        )

    latest_leaderboard_path = copy_artifact_to_latest(leaderboard_path, "leaderboard.csv")
    latest_metrics_path = copy_artifact_to_latest(metrics_path, "evaluation_metrics.json")
    latest_deployment_config_path = copy_artifact_to_latest(deployment_config_path, "deployment_config.json")

    # Đồng bộ TOÀN BỘ artifact model (tên file có thể khác nhau tùy model_type - xem `artifacts` trả
    # về từ `export_deployment_artifacts`) sang `models/latest/`, giữ NGUYÊN tên file gốc (không ép
    # về "best_model.pkl" cứng như trước) để khớp đúng với những gì `deployment_config.json` khai báo.
    latest_artifact_paths = {
        key: copy_artifact_to_latest(path, Path(path).name) for key, path in artifacts.items()
    }

    latest_confusion_path = (
        copy_artifact_to_latest(confusion_matrix_path, "confusion_matrix.png") if confusion_matrix_path else None
    )
    latest_feature_importance_path = (
        copy_artifact_to_latest(feature_importance_path, "feature_importance.png")
        if feature_importance_path
        else None
    )
    latest_roc_curve_path = copy_artifact_to_latest(roc_curve_path, "roc_curve_data.json") if roc_curve_path else None
    if latest_confusion_path:
        copy_artifact_to_plots(latest_confusion_path, "confusion_matrix.png")
    if latest_feature_importance_path:
        copy_artifact_to_plots(latest_feature_importance_path, "feature_importance.png")

    print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
    print(f"Best model triển khai: {best_model_name} (model_type={deployment_type})")
    print("If needed, install extra libraries with: pip install lightgbm catboost statsmodels tensorflow ctgan")
    return {
        "timestamp": ts,
        "run_dir": str(run_dir),
        "selected_models": list(selected_models.keys()),
        "best_model_name": best_model_name,
        "best_model_type": deployment_type,
        "deployment_config_path": str(latest_deployment_config_path),
        "best_model_artifacts": {key: str(path) for key, path in latest_artifact_paths.items()},
        "metrics_latest_path": str(latest_metrics_path),
        "leaderboard_latest_path": str(latest_leaderboard_path),
        "roc_curve_latest_path": str(latest_roc_curve_path) if latest_roc_curve_path else None,
        "balancing_method_used": balancing_method_used,
        "leaderboard": leaderboard_df.to_dict(orient="records"),
        "best_overall_model_name": best_model_name,
    }


def main() -> None:
    """Chạy mặc định toàn bộ 16 mô hình khi script được gọi trực tiếp."""
    run_training_pipeline(list_available_models())


if __name__ == "__main__":
    main()
