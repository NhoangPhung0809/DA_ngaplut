import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from analyze_and_train import incremental_train, run_training_pipeline


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
TRAINING_STATUS_PATH = CACHE_DIR / "training_status.json"


def write_training_state(state: dict) -> None:
    """Ghi trạng thái huấn luyện để Streamlit đọc lại."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with TRAINING_STATUS_PATH.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, ensure_ascii=False)


def read_training_state() -> dict:
    """Đọc trạng thái hiện tại nếu đã có trước đó."""
    if not TRAINING_STATUS_PATH.exists():
        return {}
    with TRAINING_STATUS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    """Nhận danh sách model và cấu hình balancing từ command line."""
    parser = argparse.ArgumentParser(description="Background training worker for Streamlit.")
    parser.add_argument("--mode", default="full", choices=["full", "incremental"])
    parser.add_argument("--models-json", required=True, help="JSON list of selected model names.")
    parser.add_argument("--balancing-method", default="auto", help="Balancing method: auto | gan | smote.")
    parser.add_argument("--new-data-csv", default="", help="CSV path chứa dữ liệu mới cho incremental training.")
    return parser.parse_args()


def main() -> int:
    """Chạy pipeline train trong process riêng và cập nhật trạng thái cho frontend."""
    args = parse_args()
    selected_models = json.loads(args.models_json)
    current_state = read_training_state()
    running_state = {
        "status": "running",
        "pid": os.getpid(),
        "selected_models": selected_models,
        "balancing_method": args.balancing_method,
        "mode": args.mode,
        "started_at": current_state.get("started_at") or datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "best_model_name": None,
        "run_dir": None,
        "error": None,
    }
    write_training_state(running_state)

    try:
        if args.mode == "incremental":
            if not args.new_data_csv:
                raise ValueError("--new-data-csv là bắt buộc khi chạy mode=incremental.")
            new_data_path = Path(args.new_data_csv)
            if not new_data_path.exists():
                raise FileNotFoundError(f"Không tìm thấy new data CSV: {new_data_path}")
            import pandas as pd

            new_df = pd.read_csv(new_data_path)
            update_result = incremental_train(new_df)
            completed_state = {
                **running_state,
                "status": "completed",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "incremental_result": update_result,
            }
        else:
            result = run_training_pipeline(selected_models, balancing_method=args.balancing_method)
            completed_state = {
                **running_state,
                "status": "completed",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "best_model_name": result.get("best_model_name"),
                "run_dir": result.get("run_dir"),
                "metrics_latest_path": result.get("metrics_latest_path"),
                "best_model_latest_path": result.get("best_model_latest_path"),
                "scaler_latest_path": result.get("scaler_latest_path"),
            }
        write_training_state(completed_state)
        return 0
    except Exception as exc:
        traceback.print_exc()
        failed_state = {
            **running_state,
            "status": "failed",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "error": str(exc),
        }
        write_training_state(failed_state)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
