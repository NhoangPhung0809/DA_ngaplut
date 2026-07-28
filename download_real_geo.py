
import os
import requests
import geopandas as gpd

def main():
    # URL GeoJSON tỉnh Thừa Thiên Huế (mã 46)
    url = "https://raw.githubusercontent.com/TungWorm/Vietnam-GeoJSON/master/geojson/provinces/46.geojson"
    output_path = "data/geo/thuathienhue_districts.geojson"
    
    os.makedirs("data/geo", exist_ok=True)
    
    print("🔄 Đang tải dữ liệu ranh giới Thừa Thiên Huế...")
    try:
        r = requests.get(url)
        r.raise_for_status()
        
        print("🔍 Đang xử lý...")
        # Lưu file
        with open(output_path, 'wb') as f:
            f.write(r.content)
        
        # Đọc lại để kiểm tra
        gdf = gpd.read_file(output_path)
        print(f"\n✅ Thành công! Đã lưu {len(gdf)} huyện vào: {output_path}")
        print("📋 Các huyện có trong file:")
        print(gdf['name'].tolist())
        print("\n🎯 Bây giờ chạy lại app để xem!")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        print("\n💡 Để tải thủ công:")
        print("1. Truy cập https://github.com/TungWorm/Vietnam-GeoJSON")
        print("2. Tải file 46.geojson")
        print("3. Đổi tên thành thuathienhue_districts.geojson")
        print("4. Đặt vào thư mục data/geo/")

if __name__ == "__main__":
    main()
