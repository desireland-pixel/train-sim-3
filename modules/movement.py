import pandas as pd
import math

def load_points(data_dir):
    return pd.read_csv(data_dir / "points.csv")

def get_zone_for_warehouse(warehouse_id):
    if warehouse_id in ["W1", "W2"]:
        return "Zone1"
    elif warehouse_id in ["W3", "W4"]:
        return "Zone2"
    elif warehouse_id in ["W5", "W6"]:
        return "Zone3"
    return None

def build_route(warehouse_id, platform_id, points_df, warehouses_df):
    """
    Build a list of (from_name, to_name, (x_start, y_start), (x_end, y_end))
    dynamically using zone logic.
    """
    zone = get_zone_for_warehouse(warehouse_id)
    zone_entry = points_df.loc[points_df.name == f"{zone}Entry"].iloc[0]
    waiting_area = points_df.loc[points_df.name == "WaitingArea"].iloc[0]
    station_entry = points_df.loc[points_df.name == "StationEntry"].iloc[0]
    wh = warehouses_df.loc[warehouses_df.warehouse_id == warehouse_id].iloc[0]

    # Platform coordinates can come from fixed list or df
    platform_map = {
        1: (200, 150),
        2: (200, 100),
        3: (200, 50),
        4: (200, 0),
        5: (200, -50)
    }
    x_platform, y_platform = platform_map[int(platform_id)]

    route = [
        ("WaitingArea", "ZoneEntry", (waiting_area.x, waiting_area.y), (zone_entry.x, zone_entry.y)),
        ("ZoneEntry", warehouse_id, (zone_entry.x, zone_entry.y), (wh.x, wh.y)),
        (warehouse_id, "StationEntry", (wh.x, wh.y), (station_entry.x, station_entry.y)),
        ("StationEntry", f"P{platform_id}", (station_entry.x, station_entry.y), (x_platform, y_platform))
    ]
    return route

def interpolate_position(route, current_time, walk_speed):
    """
    Compute current (x, y) based on time across route segments.
    - walk_speed: units/min
    - route: list of segments (with start and end coords)
    """
    total_distance = sum(math.dist(seg[2], seg[3]) for seg in route)
    total_time = total_distance / walk_speed
    t = min(current_time, total_time)

    # Walk along segments sequentially
    elapsed = 0
    for _, _, (x1, y1), (x2, y2) in route:
        seg_dist = math.dist((x1, y1), (x2, y2))
        seg_time = seg_dist / walk_speed
        if elapsed + seg_time >= t:
            frac = (t - elapsed) / seg_time
            x = x1 + frac * (x2 - x1)
            y = y1 + frac * (y2 - y1)
            return x, y
        elapsed += seg_time
    return route[-1][3]
