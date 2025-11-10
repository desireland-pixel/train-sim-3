# simulation/human_assignment.py

import math
import numpy as np
import pandas as pd
from collections import defaultdict

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
        total_packages_train = len(grp)
        if total_packages_train == 0:
            continue

        # Map packages per warehouse
        wh_to_pkgs = defaultdict(list)
        for _, row in grp.iterrows():
            wh_to_pkgs[row['warehouse_id']].append(row['package_id'])

        wh_counts = {wh: len(pkgs) for wh, pkgs in wh_to_pkgs.items()}

        persons_train = []
        person_counter = 1
        assignments_train = []

        # Step 0: Workshop direct allocation
        leftovers_per_wh = {}
        for wh, count in wh_counts.items():
            if count >= capacity:
                num_persons = count // capacity
                for _ in range(num_persons):
                    person = f"Hc{person_counter}_{tid}"
                    person_counter += 1
                    pkgs = wh_to_pkgs[wh][:capacity]
                    wh_to_pkgs[wh] = wh_to_pkgs[wh][capacity:]
                    for pkg_id in pkgs:
                        assignments_train.append({
                            'package_id': pkg_id,
                            'warehouse_id': wh,
                            'train_id': tid,
                            'person': person
                        })
                if wh_to_pkgs[wh]:
                    leftovers_per_wh[wh] = wh_to_pkgs[wh]
            else:
                if wh_to_pkgs[wh]:
                    leftovers_per_wh[wh] = wh_to_pkgs[wh]

        # If no leftovers, done
        if not leftovers_per_wh:
            all_assignments.extend(assignments_train)
            continue

        # Step 1: Lower Bound
        total_leftover = sum(len(pkgs) for pkgs in leftovers_per_wh.values())
        LB = math.ceil(total_leftover / capacity)

        # Step 2: Zone calculation
        zone_map = warehouses.set_index('warehouse_id')['zone'].to_dict()
        zone_to_wh = defaultdict(list)
        for wh in leftovers_per_wh.keys():
            zone_to_wh[zone_map[wh]].append(wh)

        zone_need = {}
        for z, wh_list in zone_to_wh.items():
            total = sum(len(leftovers_per_wh[wh]) for wh in wh_list)
            zone_need[z] = math.ceil(total / capacity)
        Z_cost = sum(zone_need.values())

        # Step 3 & 4: Cluster evaluation
        cluster_map = warehouses.set_index('warehouse_id')['cluster'].to_dict()
        cluster_to_wh = defaultdict(list)
        for wh in leftovers_per_wh.keys():
            cluster_to_wh[cluster_map[wh]].append(wh)

        cluster_need = {}
        for c, wh_list in cluster_to_wh.items():
            total = sum(len(leftovers_per_wh[wh]) for wh in wh_list)
            cluster_need[c] = math.ceil(total / capacity)
        total_cluster_cost = sum(cluster_need.values())

        # Decide allocation strategy
        allocation_order = []

        # If Z_cost <= LB, assign zones
        if Z_cost <= LB:
            # Assign by zones
            for z, wh_list in zone_to_wh.items():
                for wh in wh_list:
                    pkgs = leftovers_per_wh[wh]
                    for i in range(0, len(pkgs), capacity):
                        person = f"Hc{person_counter}_{tid}"
                        person_counter += 1
                        for pkg_id in pkgs[i:i+capacity]:
                            assignments_train.append({
                                'package_id': pkg_id,
                                'warehouse_id': wh,
                                'train_id': tid,
                                'person': person
                            })
        else:
            # Assign clusters first
            for c, wh_list in cluster_to_wh.items():
                cluster_pkgs = []
                for wh in wh_list:
                    cluster_pkgs.extend(leftovers_per_wh[wh])
                for i in range(0, len(cluster_pkgs), capacity):
                    person = f"Hc{person_counter}_{tid}"
                    person_counter += 1
                    for pkg_id in cluster_pkgs[i:i+capacity]:
                        wh_for_pkg = [wh for wh in wh_list if pkg_id in leftovers_per_wh[wh]][0]
                        assignments_train.append({
                            'package_id': pkg_id,
                            'warehouse_id': wh_for_pkg,
                            'train_id': tid,
                            'person': person
                        })

        all_assignments.extend(assignments_train)

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
