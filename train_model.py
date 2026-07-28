import os
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
except Exception:
    Prophet = None

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    SARIMAX = None

try:
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
except Exception:
    EarlyStopping = None
    GRU = None
    LSTM = None
    Dense = None
    Dropout = None
    Sequential = None

try:
    from nixtla import NixtlaClient
except Exception:
    NixtlaClient = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "historical"
MODELS_DIR = BASE_DIR / "models"
SOURCE_COL = "Nguồn_dữ_liệu"
DATE_COL = "Ngày"

FEATURE_COLS = [
    "Nhiệt_độ_C",
    "Độ_ẩm_%",
    "Lượng_mưa_mm",
    "Độ_ẩm_đất",
    "Chiều_cao_triều_m",
]
TARGET_COL = "Nguy_cơ_ngập"
TIME_COL = "Thời_gian"
APP_COMPATIBLE_MODELS = {
    "Linear Regression Threshold",
    "Polynomial Logistic Regression",
    "Random Forest",
    "XGBoost",
    "Stacking Ensemble",
    "Voting Ensemble",
}


def load_and_concat_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Đọc toàn bộ CSV trong thư mục dữ liệu lịch sử và gộp thành một bảng."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Thư mục {data_dir} không tồn tại.")

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Không tìm thấy file CSV nào trong {data_dir}.")

    dataframes = []
    print(f"Tìm thấy {len(csv_files)} file CSV, bắt đầu đọc...")
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df[SOURCE_COL] = csv_file.stem
        dataframes.append(df)
        print(f"Đã đọc {csv_file.name} - {len(df)} dòng")

    full_df = pd.concat(dataframes, ignore_index=True)
    print(f"\nTổng số dữ liệu sau khi gộp: {len(full_df)} dòng")
    return full_df


def create_rule_based_label(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo nhãn lại theo luật chuyên gia để đảm bảo nhất quán trước khi huấn luyện."""
    labeled = df.copy()
    if "Chiều_cao_triều_m" not in labeled.columns:
        labeled["Chiều_cao_triều_m"] = 0.0

    labeled["Chiều_cao_triều_m"] = pd.to_numeric(
        labeled["Chiều_cao_triều_m"], errors="coerce"
    ).fillna(labeled["Chiều_cao_triều_m"].median())

    labeled[TARGET_COL] = (
        (pd.to_numeric(labeled["Lượng_mưa_mm"], errors="coerce").fillna(0) > 25)
        | (
            (pd.to_numeric(labeled["Lượng_mưa_mm"], errors="coerce").fillna(0) > 15)
            & (pd.to_numeric(labeled["Độ_ẩm_đất"], errors="coerce").fillna(0) > 0.3)
        )
        | (labeled["Chiều_cao_triều_m"] > 2.5)
    ).astype(int)

    print("Đã tạo nhãn.")
    print("Phân bố nhãn:")
    print(labeled[TARGET_COL].value_counts())
    return labeled


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch missing values, ép kiểu dữ liệu và giữ lại các cột cần thiết."""
    processed = df.copy()
    processed[TIME_COL] = pd.to_datetime(processed[TIME_COL], errors="coerce")
    processed = processed.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)

    if SOURCE_COL not in processed.columns:
        processed[SOURCE_COL] = "Unknown"

    for column in FEATURE_COLS:
        if column not in processed.columns:
            processed[column] = 0.0
        processed[column] = pd.to_numeric(processed[column], errors="coerce")
        processed[column] = processed[column].fillna(processed[column].median())

    processed[TARGET_COL] = (
        pd.to_numeric(processed[TARGET_COL], errors="coerce").fillna(0).astype(int)
    )

    print(f"\nSố dòng sau khi xử lý missing values: {len(processed)}")
    return processed[[TIME_COL, SOURCE_COL, *FEATURE_COLS, TARGET_COL]]


def build_daily_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp dữ liệu theo ngày cho các mô hình chuỗi thời gian và học sâu."""
    daily_df = df.copy()
    daily_df[DATE_COL] = daily_df[TIME_COL].dt.floor("D")
    daily_df = (
        daily_df.groupby([SOURCE_COL, DATE_COL], as_index=False)
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
        .sort_values([SOURCE_COL, DATE_COL])
        .reset_index(drop=True)
    )
    return daily_df


def split_time_series_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    time_col: str = TIME_COL,
    train_ratio: float = 0.8,
):
    """Chia train/test theo thứ tự thời gian và scale dữ liệu bằng DataFrame."""
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    split_index = int(len(df_sorted) * train_ratio)

    train_df = df_sorted.iloc[:split_index].copy()
    test_df = df_sorted.iloc[split_index:].copy()

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index,
    )

    print("\nChia Train/Test thành công.")
    print(f"Train set: {len(X_train_scaled)} dòng")
    print(f"Test set: {len(X_test_scaled)} dòng")
    print(f"Mốc thời gian chia: {test_df[time_col].min()}")
    return train_df, test_df, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def split_groupwise_time_series(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    train_ratio: float = 0.8,
):
    """Chia train/test theo thời gian cho từng địa phương riêng biệt."""
    train_parts = []
    test_parts = []

    for _, group_df in df.sort_values([group_col, time_col]).groupby(group_col):
        split_index = int(len(group_df) * train_ratio)
        train_parts.append(group_df.iloc[:split_index].copy())
        test_parts.append(group_df.iloc[split_index:].copy())

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


def evaluate_predictions(
    group_name: str,
    model_name: str,
    y_true,
    y_pred,
    status: str = "trained",
    note: str = "",
):
    """Tính bộ chỉ số đánh giá cho một mô hình."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "Nhóm": group_name,
        "Model": model_name,
        "Status": status,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "Ghi chú": note,
    }


def skipped_result(group_name: str, model_name: str, note: str):
    """Tạo dòng kết quả cho mô hình bị bỏ qua."""
    return {
        "Nhóm": group_name,
        "Model": model_name,
        "Status": "skipped",
        "Accuracy": np.nan,
        "Precision": np.nan,
        "Recall": np.nan,
        "F1-Score": np.nan,
        "Ghi chú": note,
    }


def train_statistical_models(X_train, y_train, X_test, y_test):
    """Huấn luyện 2 mô hình nhóm thống kê."""
    results = []
    trained_models = {}

    models = {
        "Linear Regression Threshold": LinearRegression(),
        "Polynomial Logistic Regression": Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
    }

    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        model.fit(X_train, y_train)
        if model_name == "Linear Regression Threshold":
            raw_pred = np.clip(model.predict(X_test), 0, 1)
            y_pred = (raw_pred >= 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
        results.append(evaluate_predictions("Thống kê", model_name, y_test, y_pred))
        trained_models[model_name] = model

    return results, trained_models


def train_machine_learning_models(X_train, y_train, X_test, y_test):
    """Huấn luyện 2 mô hình nhóm học máy."""
    results = []
    trained_models = {}

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
        ),
    }

    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append(evaluate_predictions("Học máy", model_name, y_test, y_pred))
        trained_models[model_name] = model

    return results, trained_models


def train_hybrid_models(X_train, y_train, X_test, y_test):
    """Huấn luyện 2 mô hình lai từ các mô hình tabular."""
    results = []
    trained_models = {}

    rf = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
    )
    lgbm = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )

    models = {
        "Stacking Ensemble": StackingClassifier(
            estimators=[("rf", rf), ("xgb", xgb_model)],
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            passthrough=False,
        ),
        "Voting Ensemble": VotingClassifier(
            estimators=[("rf", rf), ("xgb", xgb_model), ("lgbm", lgbm)],
            voting="soft",
        ),
    }

    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append(evaluate_predictions("Lai", model_name, y_test, y_pred))
        trained_models[model_name] = model

    return results, trained_models


def train_time_series_models(train_daily, test_daily):
    """Huấn luyện 2 mô hình chuỗi thời gian trên dữ liệu tổng hợp theo ngày."""
    results = []

    if ARIMA is None:
        results.append(skipped_result("Chuỗi thời gian", "ARIMA", "Thiếu package statsmodels"))
    else:
        y_true_all = []
        y_pred_all = []
        for _, train_group in train_daily.groupby(SOURCE_COL):
            source_name = train_group[SOURCE_COL].iloc[0]
            test_group = test_daily[test_daily[SOURCE_COL] == source_name]
            if len(train_group) < 30 or len(test_group) == 0:
                continue
            try:
                model = ARIMA(train_group[TARGET_COL].astype(float), order=(2, 0, 1)).fit()
                forecast = model.forecast(steps=len(test_group))
                predictions = (np.asarray(forecast) >= 0.5).astype(int)
                y_true_all.extend(test_group[TARGET_COL].tolist())
                y_pred_all.extend(predictions.tolist())
            except Exception:
                continue
        if y_true_all:
            results.append(evaluate_predictions("Chuỗi thời gian", "ARIMA", y_true_all, y_pred_all))
        else:
            results.append(skipped_result("Chuỗi thời gian", "ARIMA", "Không đủ dữ liệu để fit ổn định"))

    if SARIMAX is None:
        results.append(skipped_result("Chuỗi thời gian", "SARIMAX", "Thiếu package statsmodels"))
    else:
        y_true_all = []
        y_pred_all = []
        for _, train_group in train_daily.groupby(SOURCE_COL):
            source_name = train_group[SOURCE_COL].iloc[0]
            test_group = test_daily[test_daily[SOURCE_COL] == source_name]
            if len(train_group) < 30 or len(test_group) == 0:
                continue
            try:
                model = SARIMAX(
                    train_group[TARGET_COL].astype(float),
                    exog=train_group[FEATURE_COLS],
                    order=(1, 0, 1),
                    seasonal_order=(1, 0, 1, 7),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                forecast = model.forecast(steps=len(test_group), exog=test_group[FEATURE_COLS])
                predictions = (np.asarray(forecast) >= 0.5).astype(int)
                y_true_all.extend(test_group[TARGET_COL].tolist())
                y_pred_all.extend(predictions.tolist())
            except Exception:
                continue
        if y_true_all:
            results.append(evaluate_predictions("Chuỗi thời gian", "SARIMAX", y_true_all, y_pred_all))
        else:
            results.append(skipped_result("Chuỗi thời gian", "SARIMAX", "Không đủ dữ liệu để fit ổn định"))

    return results


def create_group_sequences(df, feature_cols, target_col, group_col, window_size=7):
    """Tạo sliding window theo từng địa phương cho mô hình LSTM/GRU."""
    sequences = []
    labels = []

    for _, group_df in df.sort_values([group_col, DATE_COL]).groupby(group_col):
        feature_values = group_df[feature_cols].to_numpy()
        target_values = group_df[target_col].to_numpy()
        if len(group_df) <= window_size:
            continue
        for idx in range(window_size, len(group_df)):
            sequences.append(feature_values[idx - window_size : idx])
            labels.append(target_values[idx])

    if not sequences:
        return np.empty((0, window_size, len(feature_cols))), np.array([])
    return np.asarray(sequences), np.asarray(labels)


def build_sequence_model(model_type: str, input_shape):
    """Khởi tạo mô hình LSTM hoặc GRU đơn giản."""
    model = Sequential()
    if model_type == "LSTM":
        model.add(LSTM(32, input_shape=input_shape))
    else:
        model.add(GRU(32, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_deep_learning_models(train_daily, test_daily):
    """Huấn luyện 2 mô hình học sâu trên dữ liệu chuỗi theo ngày."""
    if Sequential is None:
        return [
            skipped_result("Học sâu", "LSTM", "Thiếu tensorflow/keras"),
            skipped_result("Học sâu", "GRU", "Thiếu tensorflow/keras"),
        ]

    feature_scaler = StandardScaler()
    scaled_train = train_daily.copy()
    scaled_test = test_daily.copy()
    scaled_train[FEATURE_COLS] = feature_scaler.fit_transform(train_daily[FEATURE_COLS])
    scaled_test[FEATURE_COLS] = feature_scaler.transform(test_daily[FEATURE_COLS])

    X_train_seq, y_train_seq = create_group_sequences(
        scaled_train, FEATURE_COLS, TARGET_COL, SOURCE_COL, window_size=7
    )
    X_test_seq, y_test_seq = create_group_sequences(
        scaled_test, FEATURE_COLS, TARGET_COL, SOURCE_COL, window_size=7
    )

    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        return [
            skipped_result("Học sâu", "LSTM", "Không đủ sequence để huấn luyện"),
            skipped_result("Học sâu", "GRU", "Không đủ sequence để huấn luyện"),
        ]

    results = []
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    for model_name in ["LSTM", "GRU"]:
        print(f"\n--- {model_name} ---")
        model = build_sequence_model(model_name, (X_train_seq.shape[1], X_train_seq.shape[2]))
        model.fit(
            X_train_seq,
            y_train_seq,
            validation_split=0.2,
            epochs=12,
            batch_size=32,
            verbose=0,
            callbacks=[early_stopping],
        )
        y_pred_prob = model.predict(X_test_seq, verbose=0).reshape(-1)
        y_pred = (y_pred_prob >= 0.5).astype(int)
        results.append(evaluate_predictions("Học sâu", model_name, y_test_seq, y_pred))

    return results


def train_other_models(train_daily, test_daily):
    """Huấn luyện 2 mô hình thuộc nhóm khác: Prophet và TimeGPT."""
    results = []

    if Prophet is None:
        results.append(skipped_result("Khác", "Prophet", "Thiếu package prophet"))
    else:
        y_true_all = []
        y_pred_all = []
        for _, train_group in train_daily.groupby(SOURCE_COL):
            source_name = train_group[SOURCE_COL].iloc[0]
            test_group = test_daily[test_daily[SOURCE_COL] == source_name]
            if len(test_group) == 0:
                continue
            try:
                prophet_train = train_group.rename(
                    columns={DATE_COL: "ds", TARGET_COL: "y"}
                )[["ds", "y", *FEATURE_COLS]].copy()
                prophet_test = test_group.rename(
                    columns={DATE_COL: "ds", TARGET_COL: "y"}
                )[["ds", "y", *FEATURE_COLS]].copy()
                model = Prophet(daily_seasonality=True, weekly_seasonality=True)
                for feature in FEATURE_COLS:
                    model.add_regressor(feature)
                model.fit(prophet_train)
                forecast = model.predict(prophet_test[["ds", *FEATURE_COLS]])
                predictions = (forecast["yhat"].to_numpy() >= 0.5).astype(int)
                y_true_all.extend(prophet_test["y"].tolist())
                y_pred_all.extend(predictions.tolist())
            except Exception:
                continue
        if y_true_all:
            results.append(evaluate_predictions("Khác", "Prophet", y_true_all, y_pred_all))
        else:
            results.append(skipped_result("Khác", "Prophet", "Không fit được trên dữ liệu hiện tại"))

    timegpt_api_key = os.getenv("NIXTLA_API_KEY", "").strip()
    if NixtlaClient is None or not timegpt_api_key:
        results.append(skipped_result("Khác", "TimeGPT", "Thiếu package nixtla hoặc NIXTLA_API_KEY"))
    else:
        try:
            client = NixtlaClient(api_key=timegpt_api_key)
            y_true_all = []
            y_pred_all = []
            for _, train_group in train_daily.groupby(SOURCE_COL):
                source_name = train_group[SOURCE_COL].iloc[0]
                test_group = test_daily[test_daily[SOURCE_COL] == source_name]
                if len(test_group) == 0:
                    continue
                timegpt_train = train_group[[DATE_COL, TARGET_COL]].rename(
                    columns={DATE_COL: "ds", TARGET_COL: "y"}
                )
                forecast = client.forecast(df=timegpt_train, h=len(test_group), freq="D")
                predictions = (forecast["TimeGPT"].to_numpy() >= 0.5).astype(int)
                y_true_all.extend(test_group[TARGET_COL].tolist())
                y_pred_all.extend(predictions.tolist())
            if y_true_all:
                results.append(evaluate_predictions("Khác", "TimeGPT", y_true_all, y_pred_all))
            else:
                results.append(skipped_result("Khác", "TimeGPT", "Không lấy được forecast từ API"))
        except Exception as exc:
            results.append(skipped_result("Khác", "TimeGPT", f"Lỗi khi gọi API: {exc}"))

    return results


def plot_feature_importance(model, feature_names, model_name, save_path=None):
    """Lưu biểu đồ feature importance, không gọi `plt.show()` để tránh treo script."""
    if not hasattr(model, "feature_importances_"):
        print(f"\nMô hình {model_name} không hỗ trợ feature importance.")
        return

    feature_importance_df = pd.DataFrame(
        {"Đặc trưng": feature_names, "Độ quan trọng": model.feature_importances_}
    ).sort_values(by="Độ quan trọng", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Độ quan trọng", y="Đặc trưng", data=feature_importance_df)
    plt.title(f"Feature Importance - {model_name}")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nĐã lưu biểu đồ Feature Importance tại {save_path}")

    plt.close()


def save_results(results_df: pd.DataFrame, save_dir: Path = MODELS_DIR):
    """Lưu bảng kết quả so sánh mô hình."""
    save_dir.mkdir(parents=True, exist_ok=True)
    results_path = save_dir / "all_model_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Đã lưu bảng kết quả mô hình tại {results_path}")


def select_best_app_model(results_df: pd.DataFrame, trained_models: dict):
    """Chọn mô hình tốt nhất còn tương thích với app realtime hiện tại."""
    filtered = results_df[
        results_df["Model"].isin(APP_COMPATIBLE_MODELS)
        & (results_df["Status"] == "trained")
    ].copy()
    if filtered.empty:
        raise RuntimeError("Không có mô hình nào tương thích với app để lưu lại.")

    filtered = filtered.sort_values(by=["F1-Score", "Accuracy"], ascending=False)
    best_model_name = filtered.iloc[0]["Model"]
    return best_model_name, trained_models[best_model_name]


def save_best_model_and_scaler(model, model_name, scaler, save_dir: Path = MODELS_DIR):
    """Lưu mô hình tốt nhất và scaler để app sử dụng lại."""
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "best_flood_model.pkl"
    scaler_path = save_dir / "scaler.pkl"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Đã lưu mô hình tốt nhất cho app ({model_name}) tại {model_path}")
    print(f"Đã lưu scaler tại {scaler_path}")


def main():
    """Chạy toàn bộ pipeline huấn luyện mô hình mở rộng."""
    print("=== BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN MÔ HÌNH ===\n")
    print("--- PHẦN 1: ĐỌC DỮ LIỆU & TẠO NHÃN ---\n")

    try:
        df = load_and_concat_data()
    except FileNotFoundError as exc:
        print(exc)
        return

    df = create_rule_based_label(df)
    df_processed = preprocess_data(df)
    daily_df = build_daily_dataset(df_processed)

    _, _, X_train, X_test, y_train, y_test, scaler = split_time_series_data(
        df_processed,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        time_col=TIME_COL,
        train_ratio=0.8,
    )
    train_daily, test_daily = split_groupwise_time_series(
        daily_df,
        group_col=SOURCE_COL,
        time_col=DATE_COL,
        train_ratio=0.8,
    )

    print("\n--- PHẦN 2: HUẤN LUYỆN CÁC NHÓM MÔ HÌNH ---")
    all_results = []
    trained_models = {}

    statistical_results, statistical_models = train_statistical_models(X_train, y_train, X_test, y_test)
    ml_results, ml_models = train_machine_learning_models(X_train, y_train, X_test, y_test)
    hybrid_results, hybrid_models = train_hybrid_models(X_train, y_train, X_test, y_test)
    ts_results = train_time_series_models(train_daily, test_daily)
    dl_results = train_deep_learning_models(train_daily, test_daily)
    other_results = train_other_models(train_daily, test_daily)

    all_results.extend(statistical_results)
    all_results.extend(ml_results)
    all_results.extend(hybrid_results)
    all_results.extend(ts_results)
    all_results.extend(dl_results)
    all_results.extend(other_results)

    trained_models.update(statistical_models)
    trained_models.update(ml_models)
    trained_models.update(hybrid_models)

    results_df = pd.DataFrame(all_results).sort_values(
        by=["Status", "F1-Score", "Accuracy"],
        ascending=[True, False, False],
        na_position="last",
    ).reset_index(drop=True)

    print("\n--- PHẦN 3: ĐÁNH GIÁ & XUẤT KẾT QUẢ ---\n")
    print("BẢNG SO SÁNH CÁC MÔ HÌNH:")
    print(results_df.to_string(index=False))

    print("\nVẼ BIỂU ĐỒ FEATURE IMPORTANCE")
    if "XGBoost" in trained_models:
        plot_feature_importance(
            trained_models["XGBoost"],
            FEATURE_COLS,
            "XGBoost",
            save_path=MODELS_DIR / "xgboost_feature_importance.png",
        )
    if "Random Forest" in trained_models:
        plot_feature_importance(
            trained_models["Random Forest"],
            FEATURE_COLS,
            "Random Forest",
            save_path=MODELS_DIR / "rf_feature_importance.png",
        )

    save_results(results_df)
    best_model_name, best_model = select_best_app_model(results_df, trained_models)
    save_best_model_and_scaler(best_model, best_model_name, scaler)

    print("\n=== HOÀN THÀNH TOÀN BỘ QUÁ TRÌNH! ===")


if __name__ == "__main__":
    main()
