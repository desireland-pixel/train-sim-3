# simulation/human_movement.py
import pandas as pd
import numpy as np

def compute_human_movements(
    selected_train,
    collector_summary,
    warehouses_df,
    trains_df,
    points_df,
    human_registry=None,
    next_human_id=1,
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
        person_id = row["Person"]  # e.g. "Hc1_RE1"

        # Try to find existing permanent label for this temp collector
        if person_id in human_registry:
            permanent_label = human_registry[person_id]
        else:
            # Assign a new permanent label
            permanent_label = f"H{next_human_id}"
            human_registry[person_id] = permanent_label
            next_human_id += 1

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

        # ---------------------------
        # Initial appearance
        # ---------------------------
        status = "Appear (WaitingArea)"
        active = False
        temp_label = ""
        movements.append({
            "time": current_time,
            "permanent_label": permanent_label,
            "train_id": selected_train,
            "x": current_x,
            "y": current_y,
            "status": status,
            "temp_label": person_id,
            "active": active
        })

        # ---------------------------
        # Walk to each warehouse
        # ---------------------------
        for w in warehouses_list:
            if w not in warehouse_pos:
                continue

            wx, wy = warehouse_pos[w]["x"], warehouse_pos[w]["y"]
            walk_time = warehouse_pos[w]["walk_time_to_platform"]  # we can approximate for now

            # Move to warehouse
            current_time += walk_time
            status = f"At {w}"
            active = True
            temp_label = f"Hc{person_id}_{selected_train}" if active else ""
            movements.append({
                "time": current_time,
                "permanent_label": permanent_label,
                "train_id": selected_train,
                "x": wx,
                "y": wy,
                "status": status,
                "temp_label": person_id,
                "active": active
            })

            # Wait at warehouse
            current_time += waiting_at_warehouse

        # ---------------------------
        # Move to StationEntry
        # ---------------------------
        current_time += 1
        status = "At StationEntry"
        active = True
        temp_label = f"Hc{person_id}_{selected_train}" if active else ""
        movements.append({
            "time": current_time,
            "permanent_label": permanent_label,
            "train_id": selected_train,
            "x": station_entry.x,
            "y": station_entry.y,
            "status": status,
            "temp_label": person_id,
            "active": active
        })

        # ---------------------------
        # Move to Platform
        # ---------------------------
        current_time += 1
        status = f"At Platform {platform}"
        active = True
        temp_label = f"Hc{person_id}_{selected_train}" if active else ""
        movements.append({
            "time": current_time,
            "permanent_label": permanent_label,
            "train_id": selected_train,
            "x": platform_x,
            "y": platform_y,
            "status": status,
            "temp_label": person_id,
            "active": active
        })

        # ---------------------------
        # Wait at Platform
        # ---------------------------
        current_time += waiting_at_platform

        # ---------------------------
        # Return to Waiting Area
        # ---------------------------
        current_time += 1
        status = "Return (WaitingArea)"
        active = False
        temp_label = ""
        movements.append({
            "time": current_time,
            "permanent_label": permanent_label,
            "train_id": selected_train,
            "x": waiting_area.x,
            "y": waiting_area.y,
            "status": status,
            "temp_label": person_id,
            "active": active
        })

    # ---------------------------
    # Return DataFrame
    # ---------------------------
    return pd.DataFrame(movements), human_registry, next_human_id
