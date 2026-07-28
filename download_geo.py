
import os
import osmnx as ox
import pandas as pd

def main():
    # Cấu hình
    ox.settings.use_cache = True
    ox.settings.log_console = False
    
    # Tên các huyện với admin_level=6
    places = [
        {"name": "TP Huế", "query": "Thành phố Huế, Thừa Thiên Huế, Việt Nam"},
        {"name": "Hương Thủy", "query": "Huyện Hương Thủy, Thừa Thiên Huế, Việt Nam"},
        {"name": "Phú Vang", "query": "Huyện Phú Vang, Thừa Thiên Huế, Việt Nam"},
        {"name": "Hương Trà", "query": "Huyện Hương Trà, Thừa Thiên Huế, Việt Nam"},
        {"name": "Quảng Điền", "query": "Huyện Quảng Điền, Thừa Thiên Huế, Việt Nam"},
    ]
    
    print("🔄 Bắt đầu tải ranh giới hành chính (admin_level=6)...")
    
    gdfs = []
    for place in places:
        try:
            print(f"📥 Đang tải: {place['name']}")
            # Tải features với admin_level=6
            gdf = ox.features_from_place(
                place["query"],
                tags={"admin_level": "6"}
            )
            # Lọc chỉ lấy polygon
            gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            if len(gdf) > 0:
                gdf = gdf.head(1)  # Lấy đầu tiên
                gdf['name'] = place['name']
                gdfs.append(gdf)
        except Exception as e:
            print(f"⚠️ Lỗi với {place['name']}: {e}")
            # Thử lại với geocode_to_gdf
            try:
                gdf = ox.geocode_to_gdf(place["query"])
                gdf['name'] = place['name']
                gdfs.append(gdf)
            except Exception as e2:
                print(f"   Thử lại cũng lỗi: {e2}")
    
    if gdfs:
        merged = pd.concat(gdfs, ignore_index=True)
        os.makedirs("data/geo", exist_ok=True)
        output_path = "data/geo/thuathienhue_districts.geojson"
        merged.to_file(output_path, driver='GeoJSON')
        print(f"\n✅ Thành công! Đã lưu {len(merged)} huyện vào: {output_path}")
        print("🎯 Chạy lại app để xem kết quả!")
    else:
        print("\n❌ Không tải được dữ liệu nào!")

if __name__ == "__main__":
    main()
