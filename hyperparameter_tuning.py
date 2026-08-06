"""
Module: hyperparameter_tuning.py
=================================
Mục đích: Cung cấp các chiến lược tinh chỉnh siêu tham số (hyperparameter tuning) và chiến lược
huấn luyện lại mô hình (retraining) cho bài toán DỰ BÁO NGUY CƠ NGẬP LỤT (3 lớp: 0-An toàn,
1-Ngập nhẹ, 2-Ngập nặng), đồng bộ về schema cột với các file `fetch_data.py` / `analyze_and_train.py`
đang dùng trong đề tài.

File này CHẠY ĐỘC LẬP (standalone): phần `if __name__ == "__main__":` ở cuối file có sẵn dữ liệu
giả lập (mock data) mô phỏng đúng cấu trúc dữ liệu thật, để có thể demo toàn bộ 3 chiến lược tuning
+ pipeline retraining N+1 mà không cần file CSV thật.

Bố cục:
    1. Mock data generator                     -> generate_mock_flood_dataset()
    2. Tuning Random Forest bằng GridSearchCV   -> tune_random_forest_gridsearch()
    3. Tuning XGBoost bằng Optuna               -> tune_xgboost_optuna()
    4. Tuning LSTM (PyTorch) bằng Optuna        -> LSTMFloodClassifier, tune_lstm_optuna()
    5. Pipeline Batch Retraining N+1            -> retrain_pipeline_N_plus_1()
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
# Optuna mặc định in log rất chi tiết cho từng trial (INFO level), gây rối màn hình console
# khi chạy hàng chục/hàng trăm trial liên tiếp -> hạ xuống WARNING để chỉ hiện log quan trọng.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------------------------
# Schema dữ liệu dùng CHUNG với toàn bộ pipeline của đề tài (giống FEATURE_COLS/TARGET_COL trong
# train_model.py, analyze_and_train.py, eda_analysis.py) để code trong file này có thể "cắm thẳng"
# vào dữ liệu thật của đề tài mà không cần đổi tên cột.
# ---------------------------------------------------------------------------------------------
FEATURE_COLS = [
    "Nhiệt_độ_C",
    "Độ_ẩm_%",
    "Lượng_mưa_mm",
    "Độ_ẩm_đất",
    "Chiều_cao_triều_m",
]
TARGET_COL = "Nguy_cơ_ngập"
TIME_COL = "Thời_gian"
NUM_CLASSES = 3  # 0: An toàn, 1: Ngập nhẹ, 2: Ngập nặng

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =================================================================================================
# 0. MOCK DATA GENERATOR
# =================================================================================================
def generate_mock_flood_dataset(
    n_rows: int = 3000,
    start_date: str = "2015-01-01",
    freq: str = "h",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sinh dữ liệu giả lập (mock) mô phỏng ĐÚNG cấu trúc dữ liệu khí tượng - thủy văn thật của đề tài
    (5 biến đầu vào + 1 nhãn 3 lớp), để các hàm tuning/retraining bên dưới có thể chạy độc lập,
    không phụ thuộc vào việc đã có sẵn file CSV thu thập từ Open-Meteo hay chưa.

    Nhãn được tạo lại theo ĐÚNG luật rule-based 3 lớp đang dùng trong `analyze_and_train.py`
    (mưa/độ ẩm đất/triều cường), để dữ liệu giả lập có phân phối lớp mất cân bằng THỰC TẾ
    (giống hệt tình huống dữ liệu ngập lụt thật: lớp "Ngập nặng" luôn là thiểu số).
    """
    rng = np.random.default_rng(random_state)

    timestamps = pd.date_range(start=start_date, periods=n_rows, freq=freq)

    # Lượng mưa: phân phối Gamma (lệch phải) - đúng đặc trưng vật lý của lượng mưa (nhiều giờ
    # không mưa, một số ít thời điểm mưa rất lớn).
    rain = np.clip(rng.gamma(shape=1.1, scale=6.0, size=n_rows), 0, None)
    soil_moisture = np.clip(rng.normal(0.30, 0.10, n_rows), 0.0, 1.0)
    tide_height = np.clip(rng.normal(1.2, 0.7, n_rows), 0.0, 5.0)
    temperature = rng.normal(27.0, 3.0, n_rows)
    humidity = np.clip(rng.normal(80.0, 8.0, n_rows), 0.0, 100.0)

    # Tái tạo nhãn 3 lớp theo đúng rule-based đang dùng trong pipeline huấn luyện thật của đề tài.
    target = np.zeros(n_rows, dtype=int)
    light_mask = (rain > 25) | ((rain > 15) & (soil_moisture > 0.30)) | ((rain > 10) & (tide_height > 1.20))
    heavy_mask = (
        (rain > 50)
        | ((rain > 30) & (soil_moisture > 0.45))
        | ((rain > 20) & (soil_moisture > 0.40) & (tide_height > 1.50))
        | (tide_height > 2.50)
    )
    target[light_mask] = 1
    target[heavy_mask] = 2

    return pd.DataFrame(
        {
            TIME_COL: timestamps,
            "Nhiệt_độ_C": temperature,
            "Độ_ẩm_%": humidity,
            "Lượng_mưa_mm": rain,
            "Độ_ẩm_đất": soil_moisture,
            "Chiều_cao_triều_m": tide_height,
            TARGET_COL: target,
        }
    )


# =================================================================================================
# 1. TUNING RANDOM FOREST BẰNG GridSearchCV
# =================================================================================================
def tune_random_forest_gridsearch(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 3,
    n_jobs: int = -1,
) -> dict:
    """
    Tinh chỉnh Random Forest bằng GridSearchCV (tìm kiếm vét cạn theo lưới tham số).

    ----------------------------------------------------------------------------------------------
    TẠI SAO DÙNG GridSearchCV CHO RANDOM FOREST (mà không dùng Optuna ở bước này)?
    ----------------------------------------------------------------------------------------------
    1) Không gian tham số NHỎ và RỜI RẠC:
       Ở đây chỉ tinh chỉnh 3 siêu tham số quan trọng nhất của Random Forest, mỗi tham số chỉ có
       3-4 giá trị rời rạc được lựa chọn dựa trên kinh nghiệm thực nghiệm (n_estimators, max_depth,
       min_samples_split). Tổng số tổ hợp = 4 x 4 x 3 = 48 tổ hợp -> hoàn toàn khả thi để duyệt
       VÉT CẠN (exhaustive search) trong thời gian hợp lý, đặc biệt khi kết hợp chạy song song
       (n_jobs=-1) trên nhiều lõi CPU.

    2) GridSearchCV đảm bảo tìm được tổ hợp TỐI ƯU TUYỆT ĐỐI trong phạm vi lưới đã khai báo:
       Không có yếu tố ngẫu nhiên/xác suất trong cách duyệt (khác với Bayesian Optimization của
       Optuna), nên với cùng một dữ liệu đầu vào, kết quả tuning luôn TÁI LẬP được 100% giữa các
       lần chạy khác nhau. Đây là điểm rất quan trọng khi trình bày và bảo vệ luận văn, vì hội đồng
       có thể yêu cầu chạy lại để kiểm chứng kết quả.

    3) Dễ giải thích, minh bạch cho báo cáo:
       Kết quả `cv_results_` của GridSearchCV liệt kê đầy đủ điểm số của TỪNG tổ hợp tham số đã thử,
       rất thuận tiện để đưa vào bảng so sánh trong luận văn (ví dụ: "So sánh hiệu năng theo
       n_estimators và max_depth").

    4) Ngược lại, GridSearchCV KHÔNG PHÙ HỢP khi không gian tham số lớn/liên tục (như ở XGBoost hay
       kiến trúc LSTM bên dưới): nếu áp dụng lưới đủ mịn để không bỏ sót vùng tối ưu, số tổ hợp sẽ
       tăng theo cấp số nhân (combinatorial explosion / curse of dimensionality), khiến thời gian
       tuning trở nên bất khả thi. Đó là lý do ở phần 2 và phần 3 bên dưới, đề tài chuyển sang dùng
       Optuna (Bayesian Optimization) thay vì tiếp tục dùng GridSearchCV.
    """
    # Lưới tham số được giới hạn hợp lý: max_depth=None nghĩa là cây phát triển đến khi các lá
    # thuần khiết (pure) hoặc chạm min_samples_split - luôn cần so sánh với các độ sâu bị giới hạn
    # để tránh overfitting trên tập huấn luyện.
    param_grid = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [None, 8, 12, 16],
        "min_samples_split": [2, 5, 10],
    }

    base_model = RandomForestClassifier(
        random_state=42,
        n_jobs=n_jobs,
        # Dữ liệu ngập lụt luôn mất cân bằng lớp nặng (lớp 2 hiếm hơn nhiều so với lớp 0);
        # class_weight="balanced_subsample" giúp mỗi cây trong rừng tự cân bằng lại trọng số lớp
        # dựa trên bootstrap sample riêng của nó, thay vì học thiên lệch về lớp đa số.
        class_weight="balanced_subsample",
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        # F1-macro: tính F1 riêng cho từng lớp rồi lấy trung bình KHÔNG trọng số theo số lượng mẫu.
        # Phù hợp cho bài toán đa lớp mất cân bằng vì nó buộc mô hình phải dự đoán tốt CẢ lớp thiểu
        # số (Ngập nặng), thay vì chỉ tối ưu Accuracy (vốn dễ bị "đánh lừa" bởi lớp đa số).
        scoring="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
        refit=True,  # Sau khi tìm ra bộ tham số tốt nhất, tự động huấn luyện lại trên TOÀN BỘ X_train
    )

    grid_search.fit(X_train, y_train)

    return {
        "best_estimator": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "best_cv_f1_macro": grid_search.best_score_,
        "cv_results": pd.DataFrame(grid_search.cv_results_),
    }


# =================================================================================================
# 2. TUNING XGBOOST BẰNG OPTUNA
# =================================================================================================
def tune_xgboost_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    n_trials: int = 30,
    timeout: int | None = None,
) -> dict:
    """
    Tinh chỉnh XGBoost bằng Optuna (Bayesian Optimization dựa trên thuật toán TPE).

    ----------------------------------------------------------------------------------------------
    TẠI SAO DÙNG OPTUNA CHO XGBOOST (thay vì GridSearchCV)?
    ----------------------------------------------------------------------------------------------
    1) Không gian tham số RỘNG và LIÊN TỤC:
       Khác với Random Forest, các siêu tham số quan trọng nhất của XGBoost (learning_rate,
       subsample, colsample_bytree, reg_lambda...) là các biến LIÊN TỤC trên khoảng giá trị rộng.
       Nếu dùng GridSearchCV với bước nhảy đủ mịn để không bỏ sót vùng tối ưu (ví dụ learning_rate
       thử 20 giá trị, kết hợp với 5-6 tham số khác), tổng số tổ hợp sẽ lên tới hàng chục nghìn,
       hoàn toàn không khả thi về mặt thời gian huấn luyện.

    2) Optuna dùng thuật toán TPE (Tree-structured Parzen Estimator) - một dạng Bayesian
       Optimization: thay vì duyệt toàn bộ lưới một cách "mù quáng" như GridSearchCV, Optuna XÂY
       DỰNG MÔ HÌNH XÁC SUẤT về mối quan hệ giữa bộ tham số và điểm số mục tiêu dựa trên các trial
       ĐÃ CHẠY TRƯỚC ĐÓ, từ đó đề xuất bộ tham số có khả năng cho kết quả tốt hơn ở trial tiếp theo.
       => Với CÙNG một ngân sách thời gian/số lần thử, Optuna hội tụ về vùng tham số tối ưu nhanh
       hơn đáng kể so với Grid Search hoặc Random Search.

    3) Hỗ trợ CẮT TỈA SỚM (Pruning):
       `optuna.pruners.MedianPruner` cho phép dừng sớm các trial có xu hướng cho kết quả kém hơn
       trung vị của các trial trước đó, tránh lãng phí thời gian huấn luyện cho những tổ hợp tham
       số rõ ràng không triển vọng.

    4) Optimization metric = F1-Score (Macro):
       Theo đúng yêu cầu đề tài, việc chọn F1-macro làm hàm mục tiêu (thay vì Accuracy hay LogLoss)
       đảm bảo Optuna tìm ra bộ tham số giúp mô hình cân bằng tốt giữa 3 lớp, đặc biệt là không bỏ
       sót các trường hợp "Ngập nặng" hiếm gặp nhưng có hậu quả nghiêm trọng nếu dự đoán sai.
    """

    def objective(trial: optuna.trial.Trial) -> float:
        # Không gian tìm kiếm (search space) được khai báo TRỰC TIẾP trong hàm objective (Define-by-Run
        # API của Optuna) - đây là điểm khác biệt cốt lõi so với GridSearchCV (vốn yêu cầu khai báo
        # lưới tĩnh từ trước bằng param_grid).
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            # learning_rate dùng thang log (log=True) vì khoảng cách "có ý nghĩa" giữa 0.001 và 0.01
            # tương đương về mặt ảnh hưởng với khoảng cách giữa 0.01 và 0.1 - lấy mẫu đều trên thang
            # log giúp Optuna khám phá không gian tham số hiệu quả hơn lấy mẫu đều tuyến tính.
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        model = xgb.XGBClassifier(
            **params,
            random_state=42,
            eval_metric="mlogloss",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)

        # Hàm mục tiêu (objective) mà Optuna sẽ MAXIMIZE qua từng trial.
        return f1_score(y_valid, y_pred, average="macro", zero_division=0)

    # seed=42 để đảm bảo thứ tự các trial được đề xuất có thể tái lập giữa các lần chạy khác nhau,
    # phục vụ việc so sánh công bằng khi viết báo cáo luận văn.
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_params = study.best_params
    best_model = xgb.XGBClassifier(**best_params, random_state=42, eval_metric="mlogloss", n_jobs=-1)
    best_model.fit(X_train, y_train)

    return {
        "best_estimator": best_model,
        "best_params": best_params,
        "best_f1_macro": study.best_value,
        "study": study,  # trả về cả study để có thể vẽ optuna.visualization (optimization_history, ...)
    }


# =================================================================================================
# 3. TUNING LSTM (PyTorch) BẰNG OPTUNA
# =================================================================================================
def create_sequences(features: np.ndarray, target: np.ndarray, window_size: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """
    Tạo sliding window (chuỗi thời gian trượt) từ dữ liệu dạng bảng, phục vụ đầu vào cho LSTM.
    Mỗi mẫu huấn luyện là `window_size` bước thời gian liên tiếp, nhãn là trạng thái ngập tại
    bước thời gian NGAY SAU cửa sổ đó (dự báo dựa trên diễn biến gần nhất).
    """
    sequences, labels = [], []
    for idx in range(window_size, len(features)):
        sequences.append(features[idx - window_size : idx])
        labels.append(target[idx])
    return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.int64)


class LSTMFloodClassifier(nn.Module):
    """
    Kiến trúc LSTM ĐỘNG (dynamic architecture): số lớp LSTM và số unit của MỖI lớp được truyền vào
    linh hoạt qua `hidden_sizes`, cho phép Optuna không chỉ tìm siêu tham số huấn luyện mà còn tìm
    KIẾN TRÚC MẠNG (architecture search) - điều mà GridSearchCV rất khó biểu diễn vì số lượng tham
    số kiến trúc thay đổi tùy theo số lớp được chọn (không gian tìm kiếm có điều kiện - conditional
    search space).
    """

    def __init__(self, input_size: int, hidden_sizes: list[int], dropout: float, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.lstm_layers = nn.ModuleList()
        current_input_size = input_size
        for hidden_size in hidden_sizes:
            self.lstm_layers.append(
                nn.LSTM(input_size=current_input_size, hidden_size=hidden_size, batch_first=True)
            )
            current_input_size = hidden_size

        # Dropout đặt sau cụm LSTM (trước lớp phân loại cuối) để giảm overfitting - tương đương
        # layer Dropout đặt sau LSTM trong kiến trúc Keras ở `train_model.py`.
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(current_input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for lstm_layer in self.lstm_layers:
            out, _ = lstm_layer(out)
        # Chỉ lấy hidden state ở bước thời gian CUỐI CÙNG của chuỗi (many-to-one), vì bài toán ở đây
        # là dự báo MỘT nhãn ngập duy nhất cho thời điểm kế tiếp, không phải dự báo cả chuỗi đầu ra.
        last_step = out[:, -1, :]
        return self.classifier(self.dropout(last_step))


def _train_lstm_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    learning_rate: float,
    max_epochs: int = 30,
    patience: int = 4,
    device: torch.device = DEVICE,
) -> tuple[float, nn.Module]:
    """
    Vòng lặp huấn luyện LSTM có tích hợp EARLY STOPPING (tương đương callback `EarlyStopping` của
    TensorFlow/Keras đang dùng trong `train_model.py`), để tránh overfitting trong quá trình chạy
    thử của TỪNG trial Optuna.

    TẠI SAO EARLY STOPPING LÀ BẮT BUỘC TRONG VÒNG LẶP OPTUNA?
    - Mỗi trial Optuna là MỘT LẦN huấn luyện độc lập từ đầu. Nếu để mô hình chạy đủ `max_epochs`
      cho TẤT CẢ các trial (có thể hàng chục trial), tổng thời gian tuning sẽ rất lớn, kể cả với
      những bộ tham số rõ ràng không tốt (val_loss đã tăng trở lại từ sớm).
    - Early Stopping giúp mỗi trial TỰ DỪNG ngay khi validation loss không còn cải thiện sau
      `patience` epoch liên tiếp, vừa tiết kiệm thời gian tuning tổng thể, vừa đóng vai trò như một
      hình thức regularization (chọn điểm dừng tại lúc mô hình tổng quát hóa tốt nhất, trước khi
      bắt đầu học thuộc lòng nhiễu của tập huấn luyện).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

    for _epoch in range(max_epochs):
        model.train()
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

        # ----- Đánh giá trên tập validation sau mỗi epoch để theo dõi Early Stopping -----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_features, batch_targets in valid_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                logits = model(batch_features)
                val_losses.append(criterion(logits, batch_targets).item())
        current_val_loss = float(np.mean(val_losses))

        if current_val_loss < best_val_loss - 1e-4:
            # Val loss cải thiện rõ rệt -> lưu lại trạng thái mô hình tốt nhất tính đến thời điểm này.
            best_val_loss = current_val_loss
            best_state_dict = {key: value.clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                # EARLY STOPPING: val loss không cải thiện sau `patience` epoch liên tiếp -> dừng sớm.
                break

    if best_state_dict is not None:
        # Khôi phục lại trọng số tại epoch có val_loss tốt nhất (restore_best_weights=True, giống
        # tham số cùng tên của Keras EarlyStopping), thay vì giữ trọng số của epoch cuối cùng
        # (vốn có thể đã overfitting).
        model.load_state_dict(best_state_dict)

    return best_val_loss, model


def tune_lstm_optuna(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    X_valid_seq: np.ndarray,
    y_valid_seq: np.ndarray,
    n_trials: int = 15,
    max_epochs_per_trial: int = 25,
    early_stopping_patience: int = 4,
    device: torch.device = DEVICE,
) -> dict:
    """
    Tinh chỉnh kiến trúc + siêu tham số huấn luyện của LSTM bằng Optuna.

    ----------------------------------------------------------------------------------------------
    TẠI SAO DÙNG OPTUNA CHO LSTM (thay vì GridSearchCV)?
    ----------------------------------------------------------------------------------------------
    1) Không gian tìm kiếm là KHÔNG GIAN HỖN HỢP VÀ CÓ ĐIỀU KIỆN (mixed & conditional search space):
       - Tham số KIẾN TRÚC: số lớp LSTM (1-3 lớp, biến rời rạc) và số unit của MỖI lớp (32-256).
         Lưu ý quan trọng: số lượng tham số "số unit" phụ thuộc vào số lớp được chọn - nếu số lớp = 1
         thì chỉ có 1 tham số "units_layer_0", nhưng nếu số lớp = 3 thì có 3 tham số riêng biệt.
         Đây là dạng không gian tìm kiếm CÓ ĐIỀU KIỆN mà GridSearchCV KHÔNG THỂ biểu diễn được, vì
         GridSearchCV yêu cầu khai báo một lưới THAM SỐ CỐ ĐỊNH (static grid) ngay từ đầu.
       - Tham số HUẤN LUYỆN: dropout (0.1-0.5, liên tục) và learning_rate (liên tục, thang log).
       Optuna sử dụng "Define-by-Run API": không gian tìm kiếm được định nghĩa TRỰC TIẾP bằng code
       Python thông thường (vòng lặp for, if/else) ngay bên trong hàm objective, nên có thể dễ dàng
       biểu diễn các phụ thuộc kiểu "số tham số con phụ thuộc giá trị tham số cha" như trên.

    2) Chi phí huấn luyện của MỖI trial rất cao (phải chạy qua nhiều epoch, có backpropagation):
       Nếu áp dụng GridSearchCV/duyệt vét cạn cho không gian này (giả sử mỗi tham số chỉ chia làm
       5 mức: 3 (số lớp) x 5 x 5 x 5 (units mỗi lớp, worst-case 3 lớp) x 5 (dropout) x 5
       (learning_rate) = hàng nghìn tổ hợp), tổng thời gian huấn luyện sẽ VƯỢT XA khả năng thực hiện
       trong khuôn khổ luận văn. Với Optuna (TPE - Bayesian Optimization), chỉ cần vài chục trial là
       đã có thể hội tụ về vùng kiến trúc/tham số tốt, vì mỗi trial tiếp theo được "gợi ý thông minh"
       dựa trên kết quả các trial trước, thay vì duyệt ngẫu nhiên/vét cạn.

    3) Kết hợp Early Stopping bên trong từng trial (xem `_train_lstm_with_early_stopping`) để mỗi
       lần thử không lãng phí thời gian huấn luyện hết `max_epochs_per_trial` epoch nếu mô hình đã
       ngừng cải thiện từ sớm - tối ưu kép cả về tham số (Optuna) lẫn số epoch cần thiết (Early
       Stopping) cho mỗi tham số đó.
    """
    input_size = X_train_seq.shape[-1]

    def objective(trial: optuna.trial.Trial) -> float:
        # ---- Không gian kiến trúc (architecture search space) ----
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_sizes = [
            trial.suggest_int(f"units_layer_{layer_idx}", 32, 256, step=32) for layer_idx in range(n_layers)
        ]

        # ---- Không gian siêu tham số huấn luyện (training hyperparameters) ----
        dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)
        # learning_rate lấy mẫu theo thang LOG (log=True): đây là khuyến nghị chuẩn cho learning_rate
        # trong mọi bài toán deep learning, vì ảnh hưởng của learning_rate lên quá trình hội tụ mang
        # tính chất NHÂN (multiplicative) chứ không phải CỘNG (additive).
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train_seq), torch.from_numpy(y_train_seq)),
            batch_size=batch_size,
            shuffle=True,
        )
        valid_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_valid_seq), torch.from_numpy(y_valid_seq)),
            batch_size=batch_size,
            shuffle=False,
        )

        model = LSTMFloodClassifier(input_size=input_size, hidden_sizes=hidden_sizes, dropout=dropout_rate)

        best_val_loss, trained_model = _train_lstm_with_early_stopping(
            model,
            train_loader,
            valid_loader,
            learning_rate=learning_rate,
            max_epochs=max_epochs_per_trial,
            patience=early_stopping_patience,
            device=device,
        )

        # Ngoài val_loss (dùng làm mục tiêu tối ưu chính vì mượt/ít nhiễu hơn giữa các epoch), vẫn
        # tính thêm F1-macro trên tập validation và lưu vào `user_attrs` để tiện đối chiếu/báo cáo,
        # không dùng trực tiếp F1 làm mục tiêu tối ưu ở đây vì F1 tính trên batch rời rạc dao động
        # mạnh hơn loss liên tục, có thể khiến Optuna hội tụ kém ổn định hơn.
        trained_model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch_features, batch_targets in valid_loader:
                logits = trained_model(batch_features.to(device))
                predictions = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(predictions.tolist())
                all_targets.extend(batch_targets.numpy().tolist())
        trial.set_user_attr("f1_macro", f1_score(all_targets, all_preds, average="macro", zero_division=0))

        return best_val_loss  # Optuna sẽ MINIMIZE giá trị này

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    return {
        "best_params": best_params,
        "best_val_loss": study.best_value,
        "best_f1_macro": study.best_trial.user_attrs.get("f1_macro"),
        "study": study,
    }


# =================================================================================================
# 4. PIPELINE "N+1" BATCH RETRAINING
# =================================================================================================
def retrain_pipeline_N_plus_1(
    old_data_path: str | Path,
    new_data_path: str | Path,
    model_factory: Callable[[], object] | None = None,
    feature_cols: list[str] = FEATURE_COLS,
    target_col: str = TARGET_COL,
    time_col: str = TIME_COL,
    output_model_path: str | Path | None = None,
) -> dict:
    """
    Pipeline mô phỏng chiến lược BATCH RETRAINING trên N+1 (dữ liệu cũ + dữ liệu mới), thay vì học
    tăng cường (incremental learning) chỉ trên riêng phần dữ liệu mới (+1).

    ================================================================================================
    GIẢI TRÌNH CHO CÂU HỎI CỦA GIẢNG VIÊN HƯỚNG DẪN (dùng cho phần bảo vệ luận văn):
    "Khi nhận dữ liệu mới (+1), mô hình nên học tăng cường (incremental) hay huấn luyện lại trên
    toàn bộ dữ liệu cũ + mới (N+1)?"
    ================================================================================================
    Đề tài lựa chọn chiến lược BATCH RETRAINING (N+1), vì các lý do sau:

    1) TRÁNH "CATASTROPHIC FORGETTING" (LÃNG QUÊN THẢM KHỐC):
       Nếu chỉ gọi `model.fit()` thêm trên riêng batch dữ liệu mới (incremental/online learning),
       các mô hình dựa trên gradient (Neural Network cập nhật trọng số theo batch mới, hay XGBoost
       tiếp tục "boost" thêm cây dựa trên batch mới) có xu hướng điều chỉnh mạnh theo PHÂN PHỐI của
       batch mới nhất, và có thể dần "quên" các quy luật đã học được từ dữ liệu cũ. Điều này đặc
       biệt nguy hiểm với bài toán ngập lụt: lớp "Ngập nặng" vốn đã RẤT HIẾM trong toàn bộ tập dữ
       liệu; nếu một batch dữ liệu mới (+1, ví dụ chỉ vài ngày) tình cờ KHÔNG chứa case ngập nặng
       nào, mô hình học tăng cường có nguy cơ dần mất khả năng nhận diện đúng lớp này.

    2) DỮ LIỆU KHÍ TƯỢNG - THỦY VĂN MANG TÍNH MÙA VỤ (SEASONALITY) VÀ KHÔNG DỪNG (NON-STATIONARY):
       Một batch dữ liệu mới thường chỉ đại diện cho MỘT giai đoạn thời gian ngắn (vài ngày/tuần),
       không phản ánh đầy đủ chu kỳ mùa mưa - mùa khô trong năm (theo EDA của đề tài, tháng 10-12
       là cao điểm mưa/ngập, các tháng giữa năm lại rất thấp). Nếu cập nhật mô hình liên tục theo
       từng batch ngắn hạn, mô hình dễ bị "trôi" (concept drift) lệch theo đặc điểm thời tiết cục bộ
       gần nhất, thay vì giữ được bức tranh quy luật dài hạn trên toàn bộ 10 năm dữ liệu.

    3) CẦN TÁI ÁP DỤNG BƯỚC CÂN BẰNG DỮ LIỆU (SMOTE/CTGAN) TRÊN TOÀN BỘ PHÂN PHỐI LỚP MỚI:
       Pipeline huấn luyện chính của đề tài (`analyze_and_train.py`) có bước cân bằng lớp thiểu số
       (SMOTE/CTGAN) TRƯỚC khi huấn luyện. Bước cân bằng này CẦN được tính toán lại trên phân phối
       lớp của TOÀN BỘ tập N+1 (vì tỷ lệ mất cân bằng có thể thay đổi khi có thêm dữ liệu mới) -
       điều mà học tăng cường (chỉ xử lý riêng batch +1, vốn thường quá nhỏ để áp dụng SMOTE/CTGAN
       một cách có ý nghĩa thống kê) không thể thực hiện đúng.

    4) TÍNH TÁI LẬP VÀ KHẢ NĂNG KIỂM CHỨNG (REPRODUCIBILITY & AUDITABILITY):
       Với batch retraining, MỖI phiên bản mô hình được huấn luyện lại từ đầu trên MỘT tập dữ liệu
       cụ thể, rõ ràng, có thể gắn nhãn phiên bản (versioning - giống cơ chế `models/run_<timestamp>/`
       đang dùng trong `analyze_and_train.py`) và tái tạo lại kết quả đánh giá bất kỳ lúc nào.
       Ngược lại, với incremental learning, trạng thái cuối cùng của mô hình phụ thuộc vào TOÀN BỘ
       LỊCH SỬ các lần cập nhật trước đó (path-dependent) - rất khó tái lập hoặc debug khi mô hình
       đột nhiên suy giảm chất lượng sau nhiều vòng cập nhật.

    ĐÁNH ĐỔI (TRADE-OFF) CẦN NÊU RÕ TRONG LUẬN VĂN:
       - Chi phí tính toán của N+1 cao hơn incremental learning, vì phải huấn luyện lại từ đầu trên
         TOÀN BỘ dữ liệu (bao gồm cả phần dữ liệu cũ đã học trước đó), thay vì chỉ cập nhật trên
         phần dữ liệu mới.
       - Tuy nhiên, với quy mô dữ liệu của đề tài (dữ liệu theo giờ/ngày của 5 địa phương trong
         khoảng 10 năm - vẫn ở mức vài trăm nghìn dòng), chi phí huấn luyện lại một mô hình dạng
         cây (Random Forest/XGBoost) chỉ mất vài phút, nằm trong ngưỡng hoàn toàn chấp nhận được so
         với lợi ích về độ tin cậy và độ chính xác của mô hình - đặc biệt là với lớp "Ngập nặng"
         hiếm gặp nhưng có hậu quả nghiêm trọng nếu bị dự đoán sai.

    Tham số:
        old_data_path: đường dẫn CSV dữ liệu lịch sử (N) - ví dụ file trong `data/historical/`.
        new_data_path: đường dẫn CSV dữ liệu mới phát sinh (+1) - ví dụ
                        `cache/new_historical_rows.csv` do `fetch_data.py` xuất ra.
        model_factory: hàm khởi tạo mô hình (không tham số) để huấn luyện lại; nếu không truyền,
                        mặc định dùng RandomForestClassifier để mô phỏng nhanh. Trong thực tế đề tài
                        có thể truyền vào một hàm khởi tạo mô hình với bộ tham số tốt nhất đã tìm
                        được từ `tune_xgboost_optuna` / `tune_random_forest_gridsearch` ở trên.
        output_model_path: nếu có, lưu mô hình + scaler mới ra file .pkl (dùng joblib), mô phỏng
                        đúng cơ chế `save_best_model_and_scaler` đang dùng trong `train_model.py`.
    """
    old_data_path = Path(old_data_path)
    new_data_path = Path(new_data_path)

    # ------------------------------------------------------------------------------------------
    # BƯỚC 1: NẠP DỮ LIỆU LỊCH SỬ (N) VÀ DỮ LIỆU MỚI PHÁT SINH (+1)
    # ------------------------------------------------------------------------------------------
    old_df = pd.read_csv(old_data_path)
    new_df = pd.read_csv(new_data_path)
    print(f"[N+1] Dữ liệu cũ (N)  : {len(old_df):>8,} dòng  <- {old_data_path.name}")
    print(f"[N+1] Dữ liệu mới (+1): {len(new_df):>8,} dòng  <- {new_data_path.name}")

    # ------------------------------------------------------------------------------------------
    # BƯỚC 2: GỘP (CONCATENATE) THÀNH TẬP DỮ LIỆU N+1 HOÀN CHỈNH
    # ------------------------------------------------------------------------------------------
    combined_df = pd.concat([old_df, new_df], ignore_index=True)

    if time_col in combined_df.columns:
        # Khử trùng lặp theo mốc thời gian (nếu +1 vô tình chứa lại vài dòng cuối của N do lệch múi
        # giờ/độ trễ API) - giữ lại bản ghi MỚI NHẤT (keep="last") tương ứng mỗi timestamp, tương tự
        # logic khử trùng đang áp dụng khi append dữ liệu trong `fetch_data.py`.
        combined_df[time_col] = pd.to_datetime(combined_df[time_col], errors="coerce")
        combined_df = (
            combined_df.dropna(subset=[time_col])
            .drop_duplicates(subset=[time_col], keep="last")
            .sort_values(time_col)
            .reset_index(drop=True)
        )

    print(f"[N+1] Tổng dữ liệu sau khi gộp N+1: {len(combined_df):>8,} dòng")

    missing_columns = [column for column in [*feature_cols, target_col] if column not in combined_df.columns]
    if missing_columns:
        raise ValueError(f"Dữ liệu N+1 thiếu các cột bắt buộc: {missing_columns}")

    # ------------------------------------------------------------------------------------------
    # BƯỚC 3: KÍCH HOẠT LẠI TOÀN BỘ QUY TRÌNH HUẤN LUYỆN TRÊN TOÀN BỘ TẬP N+1
    # (không phải chỉ fit thêm trên riêng phần dữ liệu +1 - đây chính là điểm khác biệt cốt lõi
    # giữa Batch Retraining và Incremental Learning).
    # ------------------------------------------------------------------------------------------
    features = combined_df[feature_cols]
    labels = combined_df[target_col].astype(int)

    # shuffle=False vì đây là dữ liệu chuỗi thời gian: tập validation phải luôn nằm ở giai đoạn
    # SAU tập huấn luyện để mô phỏng đúng bối cảnh dự báo tương lai từ quá khứ, tránh rò rỉ dữ liệu
    # (data leakage) từ tương lai về quá khứ như khi chia ngẫu nhiên.
    X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    if model_factory is None:
        # Mặc định dùng RandomForest để mô phỏng nhanh quá trình retraining; trong pipeline thật của
        # đề tài, `model_factory` nên được truyền vào từ bộ tham số tốt nhất đã tìm được ở
        # `tune_xgboost_optuna(...)` hoặc `tune_random_forest_gridsearch(...)` phía trên.
        def model_factory() -> RandomForestClassifier:
            return RandomForestClassifier(
                n_estimators=200, random_state=42, class_weight="balanced_subsample", n_jobs=-1
            )

    model = model_factory()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_valid_scaled)

    f1_macro = f1_score(y_valid, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_valid, y_pred)
    print(f"[N+1] Retraining hoàn tất | Accuracy={accuracy:.4f} | F1-macro={f1_macro:.4f}")

    result = {
        "model": model,
        "scaler": scaler,
        "combined_dataset_size": len(combined_df),
        "accuracy": accuracy,
        "f1_macro": f1_macro,
    }

    if output_model_path is not None:
        import joblib

        output_model_path = Path(output_model_path)
        output_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "scaler": scaler}, output_model_path)
        result["saved_path"] = str(output_model_path)
        print(f"[N+1] Đã lưu mô hình mới tại: {output_model_path}")

    return result


# =================================================================================================
# DEMO CHẠY ĐỘC LẬP (STANDALONE) - dùng dữ liệu giả lập để minh họa toàn bộ 4 chiến lược ở trên.
# Số lượng trial/epoch được set NHỎ để demo chạy nhanh; khi áp dụng cho luận văn thật, nên tăng
# n_trials (XGBoost: 50-100, LSTM: 30-50) để có kết quả tuning đáng tin cậy hơn.
# =================================================================================================
if __name__ == "__main__":
    SCRATCH_DIR = Path(__file__).resolve().parent / "cache" / "tuning_demo"
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("BƯỚC 0: SINH DỮ LIỆU GIẢ LẬP (MOCK DATA)")
    print("=" * 90)
    mock_df = generate_mock_flood_dataset(n_rows=2500, random_state=42)
    print(f"Tổng số dòng dữ liệu giả lập: {len(mock_df)}")
    print("Phân bố nhãn (0-An toàn / 1-Ngập nhẹ / 2-Ngập nặng):")
    print(mock_df[TARGET_COL].value_counts().sort_index())

    # Chia train/valid theo thời gian (không shuffle) để mô phỏng đúng bài toán dự báo chuỗi thời gian.
    split_index = int(len(mock_df) * 0.8)
    train_df, valid_df = mock_df.iloc[:split_index], mock_df.iloc[split_index:]
    X_train_tab, y_train_tab = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_valid_tab, y_valid_tab = valid_df[FEATURE_COLS], valid_df[TARGET_COL]

    print("\n" + "=" * 90)
    print("BƯỚC 1: TUNING RANDOM FOREST BẰNG GridSearchCV")
    print("=" * 90)
    rf_result = tune_random_forest_gridsearch(X_train_tab, y_train_tab, cv=3)
    print("Best params (Random Forest):", rf_result["best_params"])
    print(f"Best CV F1-macro: {rf_result['best_cv_f1_macro']:.4f}")

    print("\n" + "=" * 90)
    print("BƯỚC 2: TUNING XGBOOST BẰNG OPTUNA")
    print("=" * 90)
    xgb_result = tune_xgboost_optuna(X_train_tab, y_train_tab, X_valid_tab, y_valid_tab, n_trials=8)
    print("Best params (XGBoost):", xgb_result["best_params"])
    print(f"Best Validation F1-macro: {xgb_result['best_f1_macro']:.4f}")

    print("\n" + "=" * 90)
    print("BƯỚC 3: TUNING LSTM (PyTorch) BẰNG OPTUNA")
    print("=" * 90)
    tabular_scaler = StandardScaler()
    train_features_scaled = tabular_scaler.fit_transform(train_df[FEATURE_COLS])
    valid_features_scaled = tabular_scaler.transform(valid_df[FEATURE_COLS])

    X_train_seq, y_train_seq = create_sequences(train_features_scaled, train_df[TARGET_COL].to_numpy(), window_size=7)
    X_valid_seq, y_valid_seq = create_sequences(valid_features_scaled, valid_df[TARGET_COL].to_numpy(), window_size=7)

    lstm_result = tune_lstm_optuna(
        X_train_seq, y_train_seq, X_valid_seq, y_valid_seq,
        n_trials=5, max_epochs_per_trial=8, early_stopping_patience=2,
    )
    print("Best params (LSTM):", lstm_result["best_params"])
    print(f"Best Validation Loss: {lstm_result['best_val_loss']:.4f} | F1-macro: {lstm_result['best_f1_macro']:.4f}")

    print("\n" + "=" * 90)
    print("BƯỚC 4: PIPELINE BATCH RETRAINING N+1")
    print("=" * 90)
    old_data_demo_path = SCRATCH_DIR / "mock_old_data_N.csv"
    new_data_demo_path = SCRATCH_DIR / "mock_new_data_plus1.csv"

    # Mô phỏng: 90% đầu là dữ liệu lịch sử (N), 10% cuối là dữ liệu mới vừa thu thập thêm (+1).
    split_n = int(len(mock_df) * 0.9)
    mock_df.iloc[:split_n].to_csv(old_data_demo_path, index=False)
    mock_df.iloc[split_n:].to_csv(new_data_demo_path, index=False)

    retrain_result = retrain_pipeline_N_plus_1(
        old_data_path=old_data_demo_path,
        new_data_path=new_data_demo_path,
        output_model_path=SCRATCH_DIR / "retrained_model_N_plus_1.pkl",
    )
    print(f"Kích thước tập N+1 sau khi gộp: {retrain_result['combined_dataset_size']}")
    print(f"Accuracy sau retraining: {retrain_result['accuracy']:.4f} | F1-macro: {retrain_result['f1_macro']:.4f}")

    print("\n" + "=" * 90)
    print("HOÀN TẤT DEMO. Toàn bộ artifact demo được lưu tại:", SCRATCH_DIR)
    print("=" * 90)
