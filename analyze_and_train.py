import glob
import gc
import json
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
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
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Input
    from tensorflow.keras.models import Sequential
except ImportError:
    Model = None
    EarlyStopping = None
    LSTM = None
    Dense = None
    Input = None
    Sequential = None

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "historical"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"
LATEST_MODELS_DIR = MODELS_DIR / "latest"

TIME_COL = "Thời_gian"
DATE_COL = "Ngày"
LOCATION_COL = "Địa phương"
TARGET_COL = "Nguy_cơ_ngập"
FEATURE_COLS = [
    "Nhiệt_độ_C",
    "Độ_ẩm_%",
    "Lượng_mưa_mm",
    "Độ_ẩm_đất",
    "Chiều_cao_triều_m",
]
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
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "ARIMA",
    "SARIMA",
    "LSTM",
    "LSTM + XGBoost Hybrid",
]
SEQUENCE_WINDOW = 7
CTGAN_MAX_TRAIN_ROWS_PER_CLASS = 5000
CTGAN_MAX_TARGET_ROWS_PER_CLASS = 10000
CTGAN_EPOCHS = 10
CTGAN_BATCH_SIZE = 256


def ensure_base_directories() -> None:
    """Tạo các thư mục gốc phục vụ pipeline và frontend."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_MODELS_DIR.mkdir(parents=True, exist_ok=True)


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
        df = pd.read_csv(file_path)
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


def create_multiclass_flood_label(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo nhãn 3 lớp dựa trên luật chuyên gia."""
    labeled_df = df.copy()

    for column in FEATURE_COLS:
        if column not in labeled_df.columns:
            labeled_df[column] = 0.0
        labeled_df[column] = pd.to_numeric(labeled_df[column], errors="coerce")
        labeled_df[column] = labeled_df[column].fillna(labeled_df[column].median())

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

    for column in FEATURE_COLS:
        processed_df[column] = pd.to_numeric(processed_df[column], errors="coerce")
        processed_df[column] = processed_df[column].fillna(processed_df[column].median())

    processed_df[TARGET_COL] = (
        pd.to_numeric(processed_df[TARGET_COL], errors="coerce").fillna(0).astype(int)
    )

    return processed_df[[LOCATION_COL, TIME_COL, *FEATURE_COLS, TARGET_COL]]


def chronological_train_test_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """Chia train/test theo trục thời gian, tuyệt đối không shuffle."""
    sorted_df = df.sort_values(TIME_COL).reset_index(drop=True)
    split_index = int(len(sorted_df) * train_ratio)

    train_df = sorted_df.iloc[:split_index].copy()
    test_df = sorted_df.iloc[split_index:].copy()

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    print("\nChronological split completed:")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows : {len(test_df)}")
    print(f"Split time: {test_df[TIME_COL].min()}")

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
            return apply_smote_to_training_data(X_train, y_train)

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
            print(
                f"Friendly warning: CTGAN hit MemoryError for class {class_label} ({exc}). "
                "Fallback về RandomOverSampler."
            )
            gc.collect()
            return apply_random_oversampling(X_train, y_train)
        except Exception as exc:
            print(
                f"Friendly warning: CTGAN failed for class {class_label} ({exc}). "
                "Fallback về SMOTE."
            )
            gc.collect()
            return apply_smote_to_training_data(X_train, y_train)
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
        "classification_report": report,
    }


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


def train_lstm_sequence_model(model_name: str, daily_df: pd.DataFrame) -> tuple[dict, object]:
    """Huấn luyện LSTM classifier trên sequence theo ngày."""
    X_train_seq, y_train_seq, X_test_seq, y_test_seq, _ = build_sequence_datasets(daily_df)
    lstm_model = build_lstm_classifier((X_train_seq.shape[1], X_train_seq.shape[2]))
    callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    class_weights = build_class_weight_mapping(y_train_seq)
    lstm_model.fit(
        X_train_seq,
        y_train_seq,
        validation_split=0.2,
        epochs=25,
        batch_size=32,
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
    return metrics, lstm_model


def train_lstm_xgboost_hybrid_model(model_name: str, daily_df: pd.DataFrame) -> dict:
    """Huấn luyện hybrid LSTM encoder + XGBoost classifier."""
    X_train_seq, y_train_seq, X_test_seq, y_test_seq, _ = build_sequence_datasets(daily_df)
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
    return evaluate_prediction_arrays(
        model_name=model_name,
        y_true=y_test_seq,
        y_pred=hybrid_predictions,
        category="Hybrid",
        deployment_compatible=False,
        evaluation_scope="daily_sequence",
    )


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
                evaluation_scope="hourly_tabular",
            )
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
                evaluation_scope="hourly_tabular",
            )
        elif model_kind == "time_series_arima":
            metrics = train_arima_family_model(model_name, daily_df, seasonal=False)
        elif model_kind == "time_series_sarima":
            metrics = train_arima_family_model(model_name, daily_df, seasonal=True)
        elif model_kind == "lstm_sequence":
            metrics, _ = train_lstm_sequence_model(model_name, daily_df)
        elif model_kind == "lstm_xgboost_hybrid":
            metrics = train_lstm_xgboost_hybrid_model(model_name, daily_df)
        else:
            raise ValueError(f"Unsupported model kind: {model_kind}")

        evaluation_results[model_name] = metrics
        evaluation_results[model_name]["balancing_method"] = balancing_method_used
        print(
            f"Category={metrics['category']} | "
            f"F1-Macro={metrics['f1_macro']:.4f} | "
            f"Accuracy={metrics['accuracy']:.4f}"
        )

    return trained_models, evaluation_results


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


def select_best_model(evaluation_results: dict, trained_models: dict):
    """Chọn mô hình deploy-compatible tốt nhất theo Macro F1-score."""
    compatible_candidates = [
        model_name
        for model_name, metrics in evaluation_results.items()
        if metrics.get("deployment_compatible", False) and model_name in trained_models
    ]
    if not compatible_candidates:
        raise ValueError(
            "Không có mô hình deploy-compatible nào được huấn luyện. "
            "Vui lòng chọn ít nhất một mô hình tabular để dashboard có thể nạp."
        )

    best_model_name = max(
        compatible_candidates,
        key=lambda model_name: evaluation_results[model_name]["f1_macro"],
    )
    best_model = trained_models[best_model_name]
    print(f"\nBest deploy-compatible model selected: {best_model_name}")
    return best_model_name, best_model


def save_run_artifacts(best_model, scaler: StandardScaler, run_dir: Path) -> tuple[Path, Path]:
    """Lưu scaler và best model của một lần huấn luyện vào thư mục versioned."""
    best_model_path = run_dir / "best_model.pkl"
    scaler_path = run_dir / "scaler.pkl"

    joblib.dump(best_model, best_model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Saved best model to: {best_model_path}")
    print(f"Saved scaler to    : {scaler_path}")
    return best_model_path, scaler_path


def plot_confusion_matrix(best_model, X_test_scaled: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> Path:
    """Vẽ và lưu confusion matrix cho mô hình tốt nhất của lần chạy hiện tại."""
    y_pred = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)

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

    df = load_and_concatenate_csvs()
    df = create_multiclass_flood_label(df)
    df = preprocess_features(df)
    daily_df = build_daily_modeling_dataset(df)

    X_train, X_test, y_train, y_test = chronological_train_test_split(df, train_ratio=0.8)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    balancing_method_used = resolve_balancing_method(balancing_method)
    X_train_balanced, y_train_balanced = balance_training_data(
        X_train_scaled,
        y_train,
        balancing_method=balancing_method_used,
    )

    trained_models, evaluation_results = train_and_evaluate_models(
        selected_models,
        X_train_balanced,
        y_train_balanced,
        X_test_scaled,
        y_test,
        daily_df,
        balancing_method_used,
    )

    leaderboard_df = build_leaderboard_dataframe(evaluation_results)
    best_model_name, best_model = select_best_model(evaluation_results, trained_models)

    print_leaderboard(leaderboard_df)
    metrics_path = save_evaluation_metrics(evaluation_results, run_dir)
    best_model_path, scaler_path = save_run_artifacts(best_model, scaler, run_dir)
    confusion_matrix_path = plot_confusion_matrix(best_model, X_test_scaled, y_test, run_dir)
    feature_importance_path = plot_feature_importance(best_model, X_test_scaled, y_test, run_dir)

    latest_metrics_path = copy_artifact_to_latest(metrics_path, "evaluation_metrics.json")
    latest_model_path = copy_artifact_to_latest(best_model_path, "best_model.pkl")
    latest_scaler_path = copy_artifact_to_latest(scaler_path, "scaler.pkl")
    latest_confusion_path = copy_artifact_to_latest(confusion_matrix_path, "confusion_matrix.png")
    latest_feature_importance_path = copy_artifact_to_latest(feature_importance_path, "feature_importance.png")
    copy_artifact_to_plots(latest_confusion_path, "confusion_matrix.png")
    copy_artifact_to_plots(latest_feature_importance_path, "feature_importance.png")

    print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
    print("If needed, install extra libraries with: pip install lightgbm catboost statsmodels tensorflow ctgan")
    return {
        "timestamp": ts,
        "run_dir": str(run_dir),
        "selected_models": list(selected_models.keys()),
        "best_model_name": best_model_name,
        "best_model_run_path": str(best_model_path),
        "best_model_latest_path": str(latest_model_path),
        "scaler_latest_path": str(latest_scaler_path),
        "metrics_latest_path": str(latest_metrics_path),
        "balancing_method_used": balancing_method_used,
        "leaderboard": leaderboard_df.to_dict(orient="records"),
    }


def main() -> None:
    """Chạy mặc định toàn bộ 16 mô hình khi script được gọi trực tiếp."""
    run_training_pipeline(list_available_models())


if __name__ == "__main__":
    main()
