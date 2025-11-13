# simulation/human_movement.py
import pandas as pd
import numpy as np

def compute_human_movements(
    selected_train,
    collector_summary,
    warehouses_df,
    trains_df,
    points_df,
    job_start_allowance=1,
    waiting_at_warehouse=1,
    waiting_at_platform=1
):
    """
    Compute movement timeline for all human collectors assigned to a specific train.
    """

    # ---------------------------
    # Prepare key data
    # ---------------------------
    if collector_summary is None or collector_summary.get("df") is None:
        return pd.DataFrame()

    df = collector_summary["df"]
    train_info = trains_df.loc[trains_df["train_id"] == selected_train].iloc[0]
    arrive_time = int(train_info["arrive_time"])
    platform = str(train_info["platform"])

    # timing
    earliest_start = arrive_time - 10
    appearance_time = earliest_start - job_start_allowance
    latest_finish = arrive_time

    # static points
    waiting_area = points_df.loc[points_df["name"] == "WaitingArea"].iloc[0]
    station_entry = points_df.loc[points_df["name"] == "StationEntry"].iloc[0]

    # find platform point (x, y) dynamically
    if {"x_platform", "y_platform"}.issubset(trains_df.columns):
        platform_x = float(train_info["x_platform"])
        platform_y = float(train_info["y_platform"])
    else:
        # fallback in case the columns are missing
        platform_x, platform_y = 200, 0

    # warehouses data
    warehouse_pos = warehouses_df.set_index("warehouse_id")[["x", "y", "walk_time_to_platform"]].to_dict(orient="index")

    # result container
    movements = []

    # ---------------------------
    # Iterate over each human collector
    # ---------------------------
    for _, row in df.iterrows():
        person_id = row["Person"]
        warehouses_list = [w.strip() for w in row["Warehouse(s)"].split(",") if w.strip()]
        warehouses_list = warehouses_list[::-1]  # reverse to match desired order if needed
        # We'll sort warehouses based on walk_time_to_platform descending (highest first)
        warehouses_list = sorted(
            warehouses_list,
            key=lambda w: warehouse_pos[w]["walk_time_to_platform"] if w in warehouse_pos else 0,
            reverse=True
        )

        current_time = appearance_time
        current_x = waiting_area.x
        current_y = waiting_area.y

        # Initial appearance
        movements.append({
            "time": current_time,
            "person_id": person_id,
            "train_id": selected_train,
            "x": current_x,
            "y": current_y,
            "status": "Appear (WaitingArea)"
        })

        # Walk to each warehouse
        for w in warehouses_list:
            if w not in warehouse_pos:
                continue

            wx, wy = warehouse_pos[w]["x"], warehouse_pos[w]["y"]
            walk_time = warehouse_pos[w]["walk_time_to_platform"]  # we can approximate for now

            # Move to warehouse
            current_time += walk_time
            movements.append({
                "time": current_time,
                "person_id": person_id,
                "train_id": selected_train,
                "x": wx,
                "y": wy,
                "status": f"At {w}"
            })

            # Wait at warehouse
            current_time += waiting_at_warehouse

        # After last warehouse, move to platform (via StationEntry)
        current_time += 1  # small step to StationEntry
        movements.append({
            "time": current_time,
            "person_id": person_id,
            "train_id": selected_train,
            "x": station_entry.x,
            "y": station_entry.y,
            "status": "At StationEntry"
        })

        current_time += 1
        movements.append({
            "time": current_time,
            "person_id": person_id,
            "train_id": selected_train,
            "x": platform_x,
            "y": platform_y,
            "status": f"At Platform {platform}"
        })

        # Wait at platform
        current_time += waiting_at_platform

        # Return to waiting area
        current_time += 1
        movements.append({
            "time": current_time,
            "person_id": person_id,
            "train_id": selected_train,
            "x": waiting_area.x,
            "y": waiting_area.y,
            "status": "Return (WaitingArea)"
        })

    return pd.DataFrame(movements)
