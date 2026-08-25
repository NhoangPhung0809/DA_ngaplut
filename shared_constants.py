"""
Hằng số dùng CHUNG cho toàn bộ pipeline (app.py, analyze_and_train.py, train_model.py,
eda_analysis.py, hyperparameter_tuning.py).

TẠI SAO FILE NÀY TỒN TẠI: trước đây `FEATURE_COLS` được định nghĩa ĐỘC LẬP ở 5 file khác nhau (cùng
1 giá trị copy-paste) - nếu bộ đặc trưng của model thay đổi (thêm/bớt/đổi thứ tự cột), phải nhớ sửa
đúng cả 5 nơi, dễ sửa sót 1-2 chỗ khiến scaler/model suy luận sai lệch mà không có lỗi rõ ràng nào
báo trước (dữ liệu vẫn "chạy được", chỉ là sai). Nay chỉ còn DUY NHẤT 1 định nghĩa ở đây, các file
khác `from shared_constants import FEATURE_COLS`.

File này KHÔNG import bất kỳ thư viện nặng nào (pandas, numpy, sklearn...) - chỉ chứa hằng số Python
thuần, để import được an toàn ở MỌI nơi (kể cả app.py, vốn cố tình tránh import trực tiếp
`analyze_and_train.py` ở cấp module để không kéo theo TensorFlow/XGBoost/CTGAN chỉ để đọc 1 hằng số).
"""

# Danh sách đặc trưng đầu vào của model, ĐÚNG THỨ TỰ dùng khi huấn luyện (StandardScaler ghi nhớ thứ
# tự cột này qua `feature_names_in_`) - đổi thứ tự ở đây tương đương đổi thứ tự cột đưa vào model.
FEATURE_COLS = [
    "Nhiệt_độ_C",
    "Độ_ẩm_%",
    "Lượng_mưa_mm",
    "Độ_ẩm_đất",
    "Chiều_cao_triều_m",
]
