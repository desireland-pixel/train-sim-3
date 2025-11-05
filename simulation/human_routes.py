# simulation/human_routes.py
import math

def get_zone_for_warehouse(warehouse_id):
    if warehouse_id in ["W1", "W2"]:
        return "Zone1"
    if warehouse_id in ["W3", "W4"]:
        return "Zone2"
    if warehouse_id in ["W5", "W6"]:
        return "Zone3"
    return None

def build_route(warehouse_id, platform_id, points_df, warehouses_df):
    zone = get_zone_for_warehouse(warehouse_id)
    zone_entry = points_df.loc[points_df.name == f"{zone}Entry"].iloc[0]
    waiting_area = points_df.loc[points_df.name == "WaitingArea"].iloc[0]
    station_entry = points_df.loc[points_df.name == "StationEntry"].iloc[0]
    wh = warehouses_df.loc[warehouses_df.warehouse_id == warehouse_id].iloc[0]

    platform_map = {
        1: (200, 150),
        2: (200, 100),
        3: (200, 50),
        4: (200, 0),
        5: (200, -50)
    }
    x_platform, y_platform = platform_map[int(platform_id)]

    route = [
        ("WaitingArea", "ZoneEntry", (float(waiting_area.x), float(waiting_area.y)), (float(zone_entry.x), float(zone_entry.y))),
        ("ZoneEntry", warehouse_id, (float(zone_entry.x), float(zone_entry.y)), (float(wh.x), float(wh.y))),
        (warehouse_id, "StationEntry", (float(wh.x), float(wh.y)), (float(station_entry.x), float(station_entry.y))),
        ("StationEntry", f"P{platform_id}", (float(station_entry.x), float(station_entry.y)), (x_platform, y_platform))
    ]
    return route

def interpolate_position(route, time_since_start, walk_speed):
    total_distance = sum(math.dist(seg[2], seg[3]) for seg in route)
    if walk_speed <= 0:
        return route[0][2]
    total_time = total_distance / walk_speed
    t = min(max(0, time_since_start), total_time)

    elapsed = 0.0
    for _, _, (x1, y1), (x2, y2) in route:
        seg_dist = math.dist((x1, y1), (x2, y2))
        seg_time = seg_dist / walk_speed if walk_speed > 0 else float('inf')
        if elapsed + seg_time >= t:
            if seg_time == 0:
                return (x2, y2)
            frac = (t - elapsed) / seg_time
            x = x1 + frac * (x2 - x1)
            y = y1 + frac * (y2 - y1)
            return (x, y)
        elapsed += seg_time
    return route[-1][3]
