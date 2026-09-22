"""
Hàm dùng CHUNG cho `ablation_study.py` và `calibration_study.py` - nạp ĐÚNG model tốt nhất đang triển
khai thật (`models/latest/deployment_config.json`), để 2 nghiên cứu này phản ánh đúng hành vi của
model THẬT SỰ đang phục vụ người dùng, thay vì 1 model cố định tuỳ ý (Random Forest) không liên quan
gì tới model thắng leaderboard thật của lần train gần nhất.

CHỈ hỗ trợ model DẠNG BẢNG (`kind == "tabular_classifier"` trong `build_model_registry()` -
analyze_and_train.py) - các model dạng chuỗi/hybrid (LSTM/GRU/1D-CNN/Hybrid) cần pipeline huấn luyện
HOÀN TOÀN khác (cửa sổ chuỗi ngày, không phải ma trận X/y theo hàng đơn giản) nên KHÔNG tương thích
trực tiếp với khung "cân bằng dữ liệu rồi model.fit(X, y)" mà 2 script này dùng. Khi model thắng
leaderboard thật sự là dạng chuỗi, tự động rơi về model DẠNG BẢNG tốt nhất (theo F1-Macro trong
`evaluation_metrics.json`), có in cảnh báo rõ ràng lý do - không âm thầm đổi mà không nói gì.
"""

import json
from pathlib import Path

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

from analyze_and_train import build_model_registry

BASE_DIR = Path(__file__).resolve().parent
DEPLOYMENT_CONFIG_PATH = BASE_DIR / "models" / "latest" / "deployment_config.json"
EVALUATION_METRICS_PATH = BASE_DIR / "models" / "latest" / "evaluation_metrics.json"

# Fallback CUỐI CÙNG khi không đọc được file thật nào (vd chưa train lần nào) - dùng tham số đã
# regularize sẵn giống hệt `build_model_registry()`, KHÔNG phải bịa số ngẫu nhiên.
FALLBACK_MODEL_NAME = "Random Forest"


def load_best_tabular_model_spec() -> tuple[str, object]:
    """
    Trả về `(tên_model, instance_sklearn_CHƯA_fit)` của model tốt nhất đang triển khai thật - thứ tự
    ưu tiên:
      1. Đúng model thắng leaderboard thật (`deployment_config.json`), NẾU là dạng bảng.
      2. Model DẠNG BẢNG tốt nhất theo F1-Macro (`evaluation_metrics.json`), nếu #1 là dạng chuỗi/hybrid.
      3. Random Forest tham số cố định, nếu không đọc được file thật nào (vd chưa train lần nào).

    Instance trả về CHƯA fit (`sklearn.base.clone`) - gọi lại hàm này 1 lần rồi tự `clone()` thêm cho
    MỖI arm cần huấn luyện riêng (không dùng chung 1 instance đã fit giữa các arm).
    """
    registry = build_model_registry()

    deployed_name = None
    if DEPLOYMENT_CONFIG_PATH.exists():
        try:
            with DEPLOYMENT_CONFIG_PATH.open("r", encoding="utf-8") as f:
                deployment_config = json.load(f)
            deployed_name = deployment_config.get("model_name")
        except Exception as exc:
            print(f"Friendly warning: không đọc được deployment_config.json ({exc}).")

    if deployed_name and registry.get(deployed_name, {}).get("kind") == "tabular_classifier":
        print(f"Dùng ĐÚNG model đang triển khai thật: {deployed_name}")
        return deployed_name, clone(registry[deployed_name]["model"])

    if deployed_name:
        print(
            f"Friendly warning: model đang triển khai thật ('{deployed_name}') là dạng chuỗi/hybrid, "
            "KHÔNG tương thích trực tiếp với khung nghiên cứu này (cần pipeline windowed sequence "
            "riêng) - tự động rơi về model DẠNG BẢNG tốt nhất theo F1-Macro."
        )

    if EVALUATION_METRICS_PATH.exists():
        try:
            with EVALUATION_METRICS_PATH.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            tabular_candidates = [
                (name, result.get("f1_macro", -1.0))
                for name, result in metrics.items()
                if registry.get(name, {}).get("kind") == "tabular_classifier"
            ]
            if tabular_candidates:
                best_name, best_f1 = max(tabular_candidates, key=lambda item: item[1])
                print(f"Dùng model DẠNG BẢNG tốt nhất theo F1-Macro ({best_f1:.4f}): {best_name}")
                return best_name, clone(registry[best_name]["model"])
        except Exception as exc:
            print(f"Friendly warning: không đọc được evaluation_metrics.json ({exc}).")

    print(f"Friendly warning: không tìm được model thật nào từ file - fallback về {FALLBACK_MODEL_NAME}.")
    return FALLBACK_MODEL_NAME, clone(registry[FALLBACK_MODEL_NAME]["model"])
