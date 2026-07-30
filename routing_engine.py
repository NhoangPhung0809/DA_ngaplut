from __future__ import annotations

from pathlib import Path
from typing import Any

import folium
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box, shape


BASE_DIR = Path(__file__).resolve().parent
GRAPH_CACHE_PATH = BASE_DIR / "data" / "geo" / "hue_drive.graphml"
DEFAULT_PLACE_NAME = "Thừa Thiên Huế, Vietnam"
DEFAULT_CENTER = (16.4637, 107.5909)
DEFAULT_PENALTY_FACTOR = 10_000

SAFE_STATUSES = {"safe", "no flood", "dry"}
LIGHT_FLOOD_STATUSES = {"light flood", "light_flood", "minor flood"}
HEAVY_FLOOD_STATUSES = {"heavy flood", "heavy_flood", "severe flood"}

EDGE_COLOR_BY_STATUS = {
    "Safe": "green",
    "Light Flood": "red",
    "Heavy Flood": "red",
}

REGION_FILL_BY_STATUS = {
    "Safe": "#16a34a",
    "Light Flood": "#f59e0b",
    "Heavy Flood": "#dc2626",
}

RISK_PRIORITY = {
    "Safe": 0,
    "Light Flood": 1,
    "Heavy Flood": 2,
}


def _load_graphml(cache_path: Path) -> nx.MultiDiGraph:
    if hasattr(ox, "load_graphml"):
        return ox.load_graphml(cache_path)
    return ox.io.load_graphml(cache_path)


def _save_graphml(graph: nx.MultiDiGraph, cache_path: Path) -> None:
    if hasattr(ox, "save_graphml"):
        ox.save_graphml(graph, cache_path)
        return
    ox.io.save_graphml(graph, cache_path)


def download_or_load_graph(
    cache_path: str | Path = GRAPH_CACHE_PATH,
    place_name: str = DEFAULT_PLACE_NAME,
    force_download: bool = False,
) -> nx.MultiDiGraph:
    """
    Download the drivable OSM graph for Hue and cache it locally as GraphML.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    ox.settings.use_cache = True
    ox.settings.log_console = False

    if cache_path.exists() and not force_download:
        return _load_graphml(cache_path)

    graph = ox.graph_from_place(place_name, network_type="drive")
    _save_graphml(graph, cache_path)
    return graph


def _normalize_risk_status(raw_status: Any) -> str:
    value = str(raw_status or "Safe").strip().lower()
    if value in HEAVY_FLOOD_STATUSES:
        return "Heavy Flood"
    if value in LIGHT_FLOOD_STATUSES:
        return "Light Flood"
    if value in SAFE_STATUSES:
        return "Safe"
    if "heavy" in value:
        return "Heavy Flood"
    if "light" in value:
        return "Light Flood"
    if "flood" in value:
        return "Heavy Flood"
    return "Safe"


def _normalize_coordinate_pair(lat_or_lon: float, lon_or_lat: float) -> tuple[float, float]:
    """
    Return a (lon, lat) pair.

    If the first value looks like latitude and the second like longitude,
    swap them so shapely receives the correct order.
    """
    first = float(lat_or_lon)
    second = float(lon_or_lat)

    if abs(first) <= 90 and abs(second) <= 180 and abs(second) > 90:
        return second, first
    return first, second


def _build_polygon_from_pairs(pairs: list[tuple[float, float]]) -> Polygon:
    normalized = [_normalize_coordinate_pair(first, second) for first, second in pairs]
    if normalized[0] != normalized[-1]:
        normalized.append(normalized[0])
    return Polygon(normalized)


def _coerce_region_geometry(region: dict[str, Any]):
    geometry = region.get("geometry")
    if geometry is not None:
        if hasattr(geometry, "geom_type"):
            return geometry
        return shape(geometry)

    polygon_points = region.get("polygon") or region.get("coordinates")
    if polygon_points:
        return _build_polygon_from_pairs(list(polygon_points))

    bounds = region.get("bounds")
    if bounds and len(bounds) == 4:
        a, b, c, d = [float(value) for value in bounds]
        if abs(a) <= 90 and abs(b) > 90 and abs(c) <= 90 and abs(d) > 90:
            min_lat, min_lon, max_lat, max_lon = a, b, c, d
            return box(min_lon, min_lat, max_lon, max_lat)
        return box(a, b, c, d)

    min_lat = region.get("min_lat")
    min_lon = region.get("min_lon")
    max_lat = region.get("max_lat")
    max_lon = region.get("max_lon")
    if None not in (min_lat, min_lon, max_lat, max_lon):
        return box(float(min_lon), float(min_lat), float(max_lon), float(max_lat))

    raise ValueError(f"Cannot build flood geometry from region: {region}")


def normalize_flooded_regions(flooded_regions: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """
    Normalize user-provided flood regions into a consistent geometry/risk schema.

    Accepted region examples:
    - {"name": "Zone A", "risk": "Heavy Flood", "geometry": shapely_polygon}
    - {"risk": "Light Flood", "polygon": [(16.47, 107.58), (16.48, 107.60), ...]}
    - {"risk": "Heavy Flood", "bounds": [16.45, 107.55, 16.50, 107.62]}
    """
    normalized_regions: list[dict[str, Any]] = []

    for index, region in enumerate(flooded_regions or [], start=1):
        risk_status = _normalize_risk_status(region.get("risk") or region.get("status"))
        geometry = _coerce_region_geometry(region)
        normalized_regions.append(
            {
                "name": region.get("name", f"Region {index}"),
                "risk": risk_status,
                "geometry": geometry,
            }
        )

    return normalized_regions


def _edge_geometry(graph: nx.MultiDiGraph, u: int, v: int, data: dict[str, Any]) -> LineString:
    geometry = data.get("geometry")
    if geometry is not None:
        return geometry

    start = (graph.nodes[u]["x"], graph.nodes[u]["y"])
    end = (graph.nodes[v]["x"], graph.nodes[v]["y"])
    return LineString([start, end])


def _edge_risk_status(edge_geometry: LineString, regions: list[dict[str, Any]]) -> str:
    selected_status = "Safe"
    for region in regions:
        if edge_geometry.intersects(region["geometry"]):
            region_status = region["risk"]
            if RISK_PRIORITY[region_status] > RISK_PRIORITY[selected_status]:
                selected_status = region_status
    return selected_status


def annotate_flood_risk_on_graph(
    graph: nx.MultiDiGraph,
    flooded_regions: list[dict[str, Any]] | None,
    penalty_factor: float = DEFAULT_PENALTY_FACTOR,
) -> nx.MultiDiGraph:
    """
    Copy the graph and assign flood status, display color, and routing weight per edge.
    """
    normalized_regions = normalize_flooded_regions(flooded_regions)
    graph_with_risk = graph.copy()

    for u, v, key, data in graph_with_risk.edges(keys=True, data=True):
        base_length = float(data.get("length", 1.0))
        edge_geom = _edge_geometry(graph_with_risk, u, v, data)
        risk_status = _edge_risk_status(edge_geom, normalized_regions)
        is_flooded = risk_status in {"Light Flood", "Heavy Flood"}

        data["geometry"] = edge_geom
        data["base_weight"] = base_length
        data["flood_risk"] = risk_status
        data["is_flooded"] = is_flooded
        data["visual_color"] = EDGE_COLOR_BY_STATUS[risk_status]
        data["weight"] = base_length * penalty_factor if is_flooded else base_length

    return graph_with_risk


def _geometry_to_latlon_sequences(geometry) -> list[list[tuple[float, float]]]:
    if isinstance(geometry, LineString):
        return [[(lat, lon) for lon, lat in geometry.coords]]

    if isinstance(geometry, MultiLineString):
        return [[(lat, lon) for lon, lat in line.coords] for line in geometry.geoms]

    raise TypeError(f"Unsupported line geometry: {type(geometry)}")


def _polygon_to_latlon_rings(geometry) -> list[list[tuple[float, float]]]:
    polygons: list[Polygon] = []
    if isinstance(geometry, Polygon):
        polygons = [geometry]
    elif isinstance(geometry, MultiPolygon):
        polygons = list(geometry.geoms)
    else:
        return []

    rings: list[list[tuple[float, float]]] = []
    for polygon in polygons:
        rings.append([(lat, lon) for lon, lat in polygon.exterior.coords])
    return rings


def _add_flood_regions_to_map(base_map: folium.Map, regions: list[dict[str, Any]]) -> None:
    for region in regions:
        color = REGION_FILL_BY_STATUS[region["risk"]]
        for ring in _polygon_to_latlon_rings(region["geometry"]):
            folium.Polygon(
                locations=ring,
                color=color,
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=0.18,
                tooltip=f"{region['name']} ({region['risk']})",
            ).add_to(base_map)


def build_visual_line_map(
    graph: nx.MultiDiGraph,
    flooded_regions: list[dict[str, Any]] | None,
    map_center: tuple[float, float] = DEFAULT_CENTER,
    zoom_start: int = 12,
) -> folium.Map:
    """
    Render a Visual Line map:
    - green street segments for Safe
    - red street segments for Light Flood / Heavy Flood
    """
    normalized_regions = normalize_flooded_regions(flooded_regions)
    graph_with_risk = annotate_flood_risk_on_graph(graph, normalized_regions)
    edge_gdf = ox.graph_to_gdfs(graph_with_risk, nodes=False, fill_edge_geometry=True)

    visual_map = folium.Map(location=map_center, zoom_start=zoom_start, tiles="OpenStreetMap")
    _add_flood_regions_to_map(visual_map, normalized_regions)

    for row in edge_gdf.itertuples():
        sequences = _geometry_to_latlon_sequences(row.geometry)
        for sequence in sequences:
            folium.PolyLine(
                locations=sequence,
                color=row.visual_color,
                weight=3,
                opacity=0.9,
                tooltip=f"Road risk: {row.flood_risk}",
            ).add_to(visual_map)

    return visual_map


def _best_edge_data(graph: nx.MultiDiGraph, u: int, v: int) -> dict[str, Any]:
    edge_bundle = graph.get_edge_data(u, v)
    if not edge_bundle:
        raise ValueError(f"No edge found between nodes {u} and {v}.")
    return min(edge_bundle.values(), key=lambda item: float(item.get("weight", item.get("length", 1.0))))


def _route_to_coordinates(graph: nx.MultiDiGraph, route: list[int]) -> list[tuple[float, float]]:
    coordinates: list[tuple[float, float]] = []

    for index in range(len(route) - 1):
        u = route[index]
        v = route[index + 1]
        edge_data = _best_edge_data(graph, u, v)
        edge_geometry = _edge_geometry(graph, u, v, edge_data)
        segment_sequences = _geometry_to_latlon_sequences(edge_geometry)

        for sequence in segment_sequences:
            if coordinates and sequence and coordinates[-1] == sequence[0]:
                coordinates.extend(sequence[1:])
            else:
                coordinates.extend(sequence)

    return coordinates


def get_safe_route(
    G: nx.MultiDiGraph,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    flooded_regions: list[dict[str, Any]] | None,
    penalty_factor: float = DEFAULT_PENALTY_FACTOR,
) -> tuple[list[int], folium.Map, nx.MultiDiGraph]:
    """
    Calculate a safe route that strongly penalizes flooded roads.

    Steps:
    - nearest start/end nodes are found from the provided coordinates
    - edges in Light Flood or Heavy Flood zones get length * penalty_factor
    - Dijkstra uses the modified weight to avoid flood-prone roads
    - the resulting route is rendered as a bold blue polyline on a Folium map
    """
    normalized_regions = normalize_flooded_regions(flooded_regions)
    graph_with_penalties = annotate_flood_risk_on_graph(
        G,
        normalized_regions,
        penalty_factor=penalty_factor,
    )

    start_node = ox.distance.nearest_nodes(graph_with_penalties, X=start_lon, Y=start_lat)
    end_node = ox.distance.nearest_nodes(graph_with_penalties, X=end_lon, Y=end_lat)

    route = nx.dijkstra_path(graph_with_penalties, start_node, end_node, weight="weight")
    route_coords = _route_to_coordinates(graph_with_penalties, route)

    route_map = build_visual_line_map(graph_with_penalties, normalized_regions)
    folium.Marker(
        location=[start_lat, start_lon],
        tooltip="Start",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(route_map)
    folium.Marker(
        location=[end_lat, end_lon],
        tooltip="End",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(route_map)
    folium.PolyLine(
        locations=route_coords,
        color="blue",
        weight=7,
        opacity=0.95,
        tooltip="Dynamic safe route",
    ).add_to(route_map)

    if route_coords:
        route_map.fit_bounds(route_coords)

    return route, route_map, graph_with_penalties


def save_map_html(map_object: folium.Map, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_object.save(str(output_path))
    return output_path


def visual_line_and_route_demo(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    flooded_regions: list[dict[str, Any]] | None,
    graph_cache_path: str | Path = GRAPH_CACHE_PATH,
) -> dict[str, Any]:
    """
    Convenience wrapper to:
    - download/load the Hue driving graph
    - build a Visual Line map
    - compute a dynamic safe route
    """
    graph = download_or_load_graph(cache_path=graph_cache_path)
    visual_map = build_visual_line_map(graph, flooded_regions)
    route, route_map, graph_with_penalties = get_safe_route(
        graph,
        start_lat=start_lat,
        start_lon=start_lon,
        end_lat=end_lat,
        end_lon=end_lon,
        flooded_regions=flooded_regions,
    )

    return {
        "graph": graph_with_penalties,
        "visual_line_map": visual_map,
        "route": route,
        "route_map": route_map,
    }


__all__ = [
    "DEFAULT_PENALTY_FACTOR",
    "GRAPH_CACHE_PATH",
    "annotate_flood_risk_on_graph",
    "build_visual_line_map",
    "download_or_load_graph",
    "get_safe_route",
    "normalize_flooded_regions",
    "save_map_html",
    "visual_line_and_route_demo",
]
