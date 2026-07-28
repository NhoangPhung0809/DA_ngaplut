
import geopandas as gpd

gdf = gpd.read_file("data/geo/thuathienhue_districts.geojson")

print("Các cột trong GeoJSON:")
print(gdf.columns.tolist())
print("\nThông tin từng huyện:")
for idx, row in gdf.iterrows():
    print(f"\nHuyện {idx+1}:")
    print(f"  Tên (name): {row.get('name', 'N/A')}")
    print(f"  Display name: {row.get('display_name', 'N/A')[:100]}...")
