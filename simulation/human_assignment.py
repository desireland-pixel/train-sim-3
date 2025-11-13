# simulation/human_assignment.py

import math
import pandas as pd
from collections import defaultdict


def infer_train_id_from_pkg(pkg_id, trains_df):
    """Infer train_id from package_id prefix."""
    try:
        prefix = str(pkg_id)[:2]
        idx = int(prefix)
        if 1 <= idx <= len(trains_df):
            return trains_df.iloc[idx - 1]['train_id']
    except Exception:
        pass
    return "UNKNOWN"


def assign_packages(packages_df, trains_df, warehouses_df, capacity):
    """
    Assign packages to human collectors based on warehouse capacity, zone, and cluster.

    Decision Tree (updated):
    For each train_id:
        1. Directly assign warehouses with >= capacity packages.
        2. For leftovers:
            - Compute LB = ceil(total_leftover / capacity)
            - Compute Z_cost = zone grouping cost
            - If Z_cost ≤ LB → choose Zone
            - Else compute C_cost
            - If C_cost < Z_cost → choose Cluster
            - Else compute Comb_cost (zone+cluster)
            - If Comb_cost < Z_cost → choose Zone+Cluster
            - Else → fallback to Zone
    """

    packages = packages_df.copy().reset_index(drop=True)
    trains = trains_df.copy().reset_index(drop=True)
    warehouses = warehouses_df.copy().reset_index(drop=True)

    packages['package_id'] = packages['package_id'].astype(str)
    packages['warehouse_id'] = packages['warehouse_id'].astype(str)
    packages['train_id'] = packages['package_id'].apply(lambda pid: infer_train_id_from_pkg(pid, trains))

    all_assignments = []
    train_groups = packages.groupby('train_id')

    # Mapping: warehouse_id -> zone, cluster
    wh_zone_map = warehouses.set_index('warehouse_id')['zone'].to_dict()
    wh_cluster_map = warehouses.set_index('warehouse_id')['cluster'].to_dict()

    for tid, grp in train_groups:
        total_packages_train = len(grp)
        if total_packages_train == 0:
            continue

        assignments_train = []
        person_idx = 0

        # Step 0: Group packages per warehouse
        wh_to_pkgs_train = defaultdict(list)
        for _, row in grp.iterrows():
            wh_to_pkgs_train[row['warehouse_id']].append(row['package_id'])

        remaining_wh = {}
        # Step 1: Directly assign warehouses with full capacity
        for wh, pkgs in wh_to_pkgs_train.items():
            n = len(pkgs)
            if n >= capacity:
                full_batches = n // capacity
                for f in range(full_batches):
                    person = f"Hc{person_idx + 1}_{tid}"
                    batch_pkgs = pkgs[f * capacity:(f + 1) * capacity]
                    for p in batch_pkgs:
                        assignments_train.append({
                            'package_id': p,
                            'warehouse_id': wh,
                            'train_id': tid,
                            'person': person
                        })
                    person_idx += 1

                leftover = n % capacity
                if leftover > 0:
                    remaining_wh[wh] = pkgs[-leftover:]
            else:
                remaining_wh[wh] = pkgs

        # If nothing remains, move to next train
        if not remaining_wh:
            all_assignments.extend(assignments_train)
            continue

        # Step 2: Compute lower bound (LB)
        total_leftover = sum(len(pkgs) for pkgs in remaining_wh.values())
        LB = math.ceil(total_leftover / capacity)

        # Step 3: Compute zone grouping cost
        zone_wh_map = defaultdict(list)
        for wh in remaining_wh.keys():
            zone_wh_map[wh_zone_map[wh]].append(wh)

        zone_need = {}
        for zone, whs in zone_wh_map.items():
            cnt = sum(len(remaining_wh[wh]) for wh in whs)
            zone_need[zone] = math.ceil(cnt / capacity)
        Z_cost = sum(zone_need.values())

        # --- Decision: Zone vs Cluster vs Zone+Cluster ---
        allocation_plan = []

        if Z_cost <= LB:
            # Zone allocation
            for zone, whs in zone_wh_map.items():
                allocation_plan.append(('zone', zone, whs, zone_need[zone]))

        else:
            # Compute cluster costs
            cluster_wh_map = defaultdict(list)
            for wh in remaining_wh.keys():
                cluster_wh_map[wh_cluster_map[wh]].append(wh)

            cluster_need = {}
            for cl, whs in cluster_wh_map.items():
                cnt = sum(len(remaining_wh[wh]) for wh in whs)
                cluster_need[cl] = math.ceil(cnt / capacity)
            C_cost = sum(cluster_need.values())

            # Compute combined Zone+Cluster costs
            comb_wh_map = defaultdict(list)
            for wh in remaining_wh.keys():
                z = wh_zone_map[wh]
                c = wh_cluster_map[wh]
                comb_wh_map[f"{z}_{c}"].append(wh)

            comb_need = {}
            for comb_key, whs in comb_wh_map.items():
                cnt = sum(len(remaining_wh[wh]) for wh in whs)
                comb_need[comb_key] = math.ceil(cnt / capacity)
            Comb_cost = sum(comb_need.values())

            # --- Updated decision logic (corrected) ---
            if C_cost < Z_cost:
                # choose cluster allocation
                for cl, whs in cluster_wh_map.items():
                    allocation_plan.append(('cluster', cl, whs, cluster_need[cl]))

            elif Comb_cost < Z_cost:
                # choose zone+cluster combined grouping
                for comb_key, whs in comb_wh_map.items():
                    allocation_plan.append(('zone+cluster', comb_key, whs, comb_need[comb_key]))

            else:
                # fallback to zone allocation
                for zone, whs in zone_wh_map.items():
                    allocation_plan.append(('zone', zone, whs, zone_need[zone]))

        # Step 4: Allocate persons based on allocation plan (enhanced inside-zone logic)
        for alloc_type, target, whs, num_persons in allocation_plan:
            wh_pkg_map = {wh: remaining_wh.get(wh, []) for wh in whs if remaining_wh.get(wh)}
            if not wh_pkg_map:
                continue

            nonzero_wh_count = len(wh_pkg_map)

            # ── Inside-zone/cluster allocation decision tree ──
            if num_persons == 1:
                # Single person handles all
                all_pkgs = []
                for pkgs in wh_pkg_map.values():
                    all_pkgs += pkgs
                person = f"Hc{person_idx + 1}_{tid}"
                for wh, pkgs in wh_pkg_map.items():
                    for p in pkgs:
                        assignments_train.append({
                            'package_id': p,
                            'warehouse_id': wh,
                            'train_id': tid,
                            'person': person
                        })
                person_idx += 1

            elif nonzero_wh_count == num_persons:
                # One person per nonzero warehouse
                for wh, pkgs in wh_pkg_map.items():
                    person = f"Hc{person_idx + 1}_{tid}"
                    for p in pkgs:
                        assignments_train.append({
                            'package_id': p,
                            'warehouse_id': wh,
                            'train_id': tid,
                            'person': person
                        })
                    person_idx += 1

            else:
                # Greedy merge/split allocation
                whs_sorted = sorted(wh_pkg_map.keys())
                current_pkgs = []
                current_whs = []
                current_count = 0
                for wh in whs_sorted:
                    pkgs = wh_pkg_map[wh][:]
                    while pkgs:
                        available = capacity - current_count
                        if len(pkgs) <= available:
                            current_pkgs += pkgs
                            current_whs.append(wh)
                            current_count += len(pkgs)
                            pkgs = []
                        else:
                            current_pkgs += pkgs[:available]
                            current_whs.append(wh)
                            pkgs = pkgs[available:]
                            current_count += available

                        if current_count >= capacity:
                            person = f"Hc{person_idx + 1}_{tid}"
                            for wh2 in set(current_whs):
                                for p in [p for p in current_pkgs if p in wh_pkg_map[wh2]]:
                                    assignments_train.append({
                                        'package_id': p,
                                        'warehouse_id': wh2,
                                        'train_id': tid,
                                        'person': person
                                    })
                            person_idx += 1
                            current_pkgs = []
                            current_whs = []
                            current_count = 0

                # Handle remaining partial person
                if current_pkgs:
                    person = f"Hc{person_idx + 1}_{tid}"
                    for wh2 in set(current_whs):
                        for p in [p for p in current_pkgs if p in wh_pkg_map[wh2]]:
                            assignments_train.append({
                                'package_id': p,
                                'warehouse_id': wh2,
                                'train_id': tid,
                                'person': person
                            })
                    person_idx += 1

        all_assignments.extend(assignments_train)

    # --- Create output summaries ---
    assignments_df = pd.DataFrame(all_assignments)

    if assignments_df.empty:
        summary_df = pd.DataFrame()
        per_train_detail = {}
        metadata = {'total_packages': 0, 'capacity': capacity, 'total_persons': 0}
        return assignments_df, summary_df, per_train_detail, metadata

    # Summary per train and warehouse
    summary_df = assignments_df.groupby(["train_id", "warehouse_id"]).size().unstack(fill_value=0)
    all_warehouses = list(warehouses_df["warehouse_id"])
    summary_df = summary_df.reindex(columns=all_warehouses, fill_value=0)
    summary_df = summary_df / capacity
    summary_df = summary_df.reset_index()

    warehouse_cols = [c for c in summary_df.columns if str(c).startswith("W")]
    summary_df["Total Persons"] = summary_df[warehouse_cols].sum(axis=1).apply(math.ceil)
    summary_df[warehouse_cols] = summary_df[warehouse_cols].round(2)

    # Detailed per-train breakdown
    per_train_detail = {}
    for tid, grp in assignments_df.groupby('train_id'):
        detail_rows = []
        for (wh, person), g in grp.groupby(['warehouse_id', 'person']):
            pkgs = list(g['package_id'])
            detail_rows.append({
                'warehouse': wh,
                'person': person,
                'packages': pkgs,
                'count': len(pkgs)
            })
        per_train_detail[tid] = pd.DataFrame(detail_rows).sort_values(['warehouse', 'person']).reset_index(drop=True)

    metadata = {
        'total_packages': len(packages),
        'capacity': capacity,
        'total_persons': assignments_df['person'].nunique()
    }

    return assignments_df, summary_df, per_train_detail, metadata


def build_collector_summary(selected_train, per_train_detail, warehouses_df, trains_df,
                            job_start_allowance=1, waiting_at_warehouse=1, waiting_at_platform=1):
    """
    Build a collector summary for the selected train.
    Flexible for timing configuration (job start, waiting at warehouse/platform).
    """

    # Extract train info
    train_info = trains_df[trains_df["train_id"] == selected_train].iloc[0]
    arrive_time = train_info["arrive_time"]

    # Flexible timing parameters
    earliest_start = arrive_time - 10
    appearance_time = earliest_start - job_start_allowance  # Configurable
    # (You can later factor waiting_at_warehouse/platform into travel durations)

    # Convert to HH:MM
    time_fmt = lambda t: f"{9 + t//60:02d}:{t%60:02d}"
    earliest_start_str = time_fmt(earliest_start)
    latest_finish_str = time_fmt(arrive_time)

    # Build base DataFrame for this train
    if selected_train not in per_train_detail:
        return {"df": pd.DataFrame(), "earliest_start_str": earliest_start_str,
                "latest_finish_str": latest_finish_str, "appearance_time": appearance_time}

    train_df = per_train_detail[selected_train]

    # --- Correct column names ---
    # Expected columns: warehouse, person, packages, count
    grouped = (
        train_df.groupby("person")
        .agg({"warehouse": list, "packages": lambda x: sum(x, []) if isinstance(x.iloc[0], list) else list(x)})
        .reset_index()
    )

    # Sort warehouses by walk_time_to_platform descending
    walk_map = dict(zip(warehouses_df["warehouse_id"], warehouses_df["walk_time_to_platform"]))

    def sort_warehouses(ws):
        return sorted(ws, key=lambda w: walk_map.get(w, 0), reverse=True)

    grouped["Warehouse(s)"] = grouped["warehouse"].apply(
        lambda ws: ", ".join(sort_warehouses(ws))
    )

    def sorted_package_ids(person_row):
        ws_sorted = sort_warehouses(person_row["warehouse"])
        pkgs = []
        for w in ws_sorted:
            subset = train_df[(train_df["person"] == person_row["person"]) & (train_df["warehouse"] == w)]
            pkgs += sum(subset["packages"], [])
        return ", ".join(pkgs)

    grouped["Package IDs"] = grouped.apply(sorted_package_ids, axis=1)

    grouped = grouped[["person", "Warehouse(s)", "Package IDs"]]
    grouped = grouped.rename(columns={"person": "Person"})

    summary = {
        "df": grouped,
        "earliest_start_str": earliest_start_str,
        "latest_finish_str": latest_finish_str,
        "appearance_time": appearance_time,
        "waiting_at_warehouse": waiting_at_warehouse,
        "waiting_at_platform": waiting_at_platform
    }

    return summary
