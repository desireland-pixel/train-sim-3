# simulation/human_assignment.py

import math
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations

def infer_train_id_from_pkg(pkg_id, trains_df):
    try:
        prefix = str(pkg_id)[:2]
        idx = int(prefix)
        if 1 <= idx <= len(trains_df):
            return trains_df.iloc[idx - 1]['train_id']
    except Exception:
        pass
    return "UNKNOWN"

def assign_packages(packages_df, trains_df, warehouses_df, capacity):
    packages = packages_df.copy().reset_index(drop=True)
    trains = trains_df.copy().reset_index(drop=True)
    warehouses = warehouses_df.copy().reset_index(drop=True)

    packages['package_id'] = packages['package_id'].astype(str)
    packages['warehouse_id'] = packages['warehouse_id'].astype(str)
    packages['train_id'] = packages['package_id'].apply(lambda pid: infer_train_id_from_pkg(pid, trains))

    all_assignments = []
    train_groups = packages.groupby('train_id')

    for tid, grp in train_groups:
        if len(grp) == 0:
            continue

        # --- Step 0: Workshop direct allocation ---
        wh_to_pkgs = defaultdict(list)
        for _, row in grp.iterrows():
            wh_to_pkgs[row['warehouse_id']].append(row['package_id'])

        workshop_assignments = []
        remaining_wh = {}
        person_counter = 1

        for wh in warehouses['warehouse_id']:
            pkgs = wh_to_pkgs.get(wh, [])
            full_batches = len(pkgs) // capacity
            leftover = len(pkgs) % capacity
            # assign full batches
            for b in range(full_batches):
                batch_pkgs = pkgs[b*capacity:(b+1)*capacity]
                person = f"Hc{person_counter}_{tid}"
                for p in batch_pkgs:
                    workshop_assignments.append({
                        'package_id': p, 'warehouse_id': wh, 'train_id': tid, 'person': person
                    })
                person_counter += 1
            # leftover packages
            if leftover > 0:
                remaining_wh[wh] = pkgs[-leftover:]

        # If no remaining packages, Step 0 solved all
        if not remaining_wh:
            all_assignments.extend(workshop_assignments)
            continue

        # --- Step 1: Compute LB ---
        total_remaining = sum(len(pkgs) for pkgs in remaining_wh.values())
        LB = math.ceil(total_remaining / capacity)

        # --- Step 2: Zone mapping ---
        zone_map = defaultdict(list)
        cluster_map = defaultdict(list)
        for wh in remaining_wh:
            wh_info = warehouses[warehouses['warehouse_id']==wh].iloc[0]
            zone_map[wh_info.zone].append((wh, remaining_wh[wh]))
            cluster_map[wh_info.cluster].append((wh, remaining_wh[wh]))

        # --- Step 3: Compute Z_cost ---
        zone_needs = {z: math.ceil(sum(len(p) for _,p in lst)/capacity) for z,lst in zone_map.items()}
        Z_cost = sum(zone_needs.values())

        # --- Step 4: Assign persons per zone if possible ---
        final_assignments = workshop_assignments.copy()
        assigned_wh = set()

        if Z_cost <= LB:
            # simple zone allocation
            for z, wh_list in zone_map.items():
                for wh, pkgs in wh_list:
                    if not pkgs:
                        continue
                    person = f"Hc{person_counter}_{tid}"
                    for p in pkgs:
                        final_assignments.append({'package_id': p, 'warehouse_id': wh, 'train_id': tid, 'person': person})
                    person_counter += 1
                    assigned_wh.add(wh)
        else:
            # Step 4a: Try Zone+Cluster combinations
            best_combo = None
            best_total_persons = Z_cost
            zone_items = list(zone_map.items())
            cluster_items = list(cluster_map.items())

            for z_num in range(1, len(zone_items)+1):
                for c_num in range(1, len(cluster_items)+1):
                    for z_combo in combinations(zone_items, z_num):
                        for c_combo in combinations(cluster_items, c_num):
                            # skip if overlapping warehouses
                            z_whs = set()
                            for _, wl in z_combo:
                                z_whs.update([wh for wh,_ in wl])
                            c_whs = set()
                            for _, wl in c_combo:
                                c_whs.update([wh for wh,_ in wl])
                            if z_whs & c_whs:
                                continue
                            # compute total persons
                            total_persons_combo = 0
                            for _, wl in list(z_combo)+list(c_combo):
                                total_persons_combo += sum(math.ceil(len(p)/capacity) for _,p in wl)
                            if total_persons_combo < best_total_persons:
                                best_total_persons = total_persons_combo
                                best_combo = (z_combo, c_combo)

            if best_combo:
                z_combo, c_combo = best_combo
                # assign persons
                for _, wl in list(z_combo)+list(c_combo):
                    for wh, pkgs in wl:
                        if not pkgs:
                            continue
                        person = f"Hc{person_counter}_{tid}"
                        for p in pkgs:
                            final_assignments.append({'package_id': p, 'warehouse_id': wh, 'train_id': tid, 'person': person})
                        person_counter += 1
                        assigned_wh.add(wh)
            else:
                # fallback: assign per zone in CSV order
                for z, wh_list in zone_map.items():
                    for wh, pkgs in wh_list:
                        if not pkgs:
                            continue
                        person = f"Hc{person_counter}_{tid}"
                        for p in pkgs:
                            final_assignments.append({'package_id': p, 'warehouse_id': wh, 'train_id': tid, 'person': person})
                        person_counter += 1
                        assigned_wh.add(wh)

        # Step 5: Any leftover warehouses not assigned? (Cluster-only allocation)
        for wh, pkgs in remaining_wh.items():
            if wh in assigned_wh:
                continue
            if not pkgs:
                continue
            person = f"Hc{person_counter}_{tid}"
            for p in pkgs:
                final_assignments.append({'package_id': p, 'warehouse_id': wh, 'train_id': tid, 'person': person})
            person_counter += 1

        all_assignments.extend(final_assignments)

    # --- Prepare output DataFrames ---
    assignments_df = pd.DataFrame(all_assignments)
    if assignments_df.empty:
        summary_df = pd.DataFrame()
        per_train_detail = {}
        metadata = {'total_packages': 0, 'capacity': capacity, 'total_persons': 0}
        return assignments_df, summary_df, per_train_detail, metadata

    summary_df = assignments_df.groupby(["train_id", "warehouse_id"]).size().unstack(fill_value=0)
    all_warehouses = list(warehouses_df["warehouse_id"])
    summary_df = summary_df.reindex(columns=all_warehouses, fill_value=0)
    summary_df = summary_df / capacity
    summary_df = summary_df.reset_index()

    warehouse_cols = [c for c in summary_df.columns if str(c).startswith("W")]
    summary_df["Total Persons"] = np.ceil(summary_df[warehouse_cols].sum(axis=1)).astype(int)
    summary_df[warehouse_cols] = summary_df[warehouse_cols].round(2)

    per_train_detail = {}
    for tid, grp in assignments_df.groupby('train_id'):
        detail_rows = []
        for (wh, person), g in grp.groupby(['warehouse_id', 'person']):
            pkgs = list(g['package_id'])
            detail_rows.append({'warehouse': wh, 'person': person, 'packages': pkgs, 'count': len(pkgs)})
        per_train_detail[tid] = pd.DataFrame(detail_rows).sort_values(['warehouse', 'person']).reset_index(drop=True)

    metadata = {
        'total_packages': len(packages),
        'capacity': capacity,
        'total_persons': assignments_df['person'].nunique()
    }

    return assignments_df, summary_df, per_train_detail, metadata
