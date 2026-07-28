import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "historical"
MODELS_DIR = BASE_DIR / "models"
REQUIRED_DATA_COLUMNS = {
    "Thời_gian",
    "Nhiệt_độ_C",
    "Độ_ẩm_%",
    "Lượng_mưa_mm",
    "Độ_ẩm_đất",
    "Chiều_cao_triều_m",
    "Nguy_cơ_ngập",
}
REQUIRED_MODEL_FILES = {"best_flood_model.pkl", "scaler.pkl"}


def run_python_script(script_name: str):
    """Chạy một script Python bằng cùng interpreter hiện tại."""
    subprocess.run(
        [sys.executable, str(BASE_DIR / script_name)],
        check=True,
        cwd=BASE_DIR,
    )


def data_files_are_valid() -> bool:
    """Kiểm tra toàn bộ dữ liệu lịch sử có tồn tại và đủ cột bắt buộc."""
    if not DATA_DIR.exists():
        return False

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        return False

    for csv_file in csv_files:
        try:
            sample = pd.read_csv(csv_file, nrows=5)
            if not REQUIRED_DATA_COLUMNS.issubset(sample.columns):
                print(f"⚠️ File thiếu cột bắt buộc: {csv_file.name}")
                return False
        except Exception as exc:
            print(f"⚠️ Lỗi đọc dữ liệu {csv_file.name}: {exc}")
            return False

    return True


def model_files_exist() -> bool:
    """Kiểm tra mô hình và scaler đã tồn tại."""
    if not MODELS_DIR.exists():
        return False
    existing_files = {path.name for path in MODELS_DIR.iterdir() if path.is_file()}
    return REQUIRED_MODEL_FILES.issubset(existing_files)


def launch_streamlit():
    """Mở ứng dụng Streamlit bằng cùng môi trường Python."""
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(BASE_DIR / "app.py")],
        check=True,
        cwd=BASE_DIR,
    )


def main():
    print("🚀 Bắt đầu chạy hệ thống...")

    data_ready = data_files_are_valid()
    model_ready = model_files_exist()

    if not data_ready:
        print("\n📥 Dữ liệu chưa sẵn sàng, bắt đầu tải lại dữ liệu lịch sử...")
        run_python_script("fetch_data.py")
        print("✅ Tải dữ liệu hoàn tất!")
        data_ready = True

    if data_ready and not model_ready:
        print("\n🤖 Mô hình chưa sẵn sàng, bắt đầu huấn luyện...")
        run_python_script("train_model.py")
        print("✅ Huấn luyện hoàn tất!")
    elif data_ready and model_ready:
        print("✅ Dữ liệu và mô hình đã sẵn sàng.")

    print("\n🌐 Mở Dashboard Streamlit...")
    launch_streamlit()


if __name__ == "__main__":
    main()
