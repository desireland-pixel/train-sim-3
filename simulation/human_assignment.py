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

    # Preserve warehouse order from CSV
    warehouse_order = list(warehouses['warehouse_id'])

    for tid, grp in train_groups:
        if len(grp) == 0:
            continue

        # Count packages per warehouse
        wh_counts = {wh: 0 for wh in warehouse_order}
        wh_to_pkgs = defaultdict(list)
        for _, row in grp.iterrows():
            wh_counts[row['warehouse_id']] += 1
            wh_to_pkgs[row['warehouse_id']].append(row['package_id'])

        assignments_train = []
        persons_train = []
        person_idx = 0

        # ----------------------------
        # STEP 0: Workshop direct allocation
        # ----------------------------
        leftovers_per_wh = {}
        for wh in warehouse_order:
            n = wh_counts.get(wh, 0)
            if n >= capacity:
                num_persons = n // capacity
                for _ in range(num_persons):
                    person_id = f"Hc{len(persons_train)+1}_{tid}"
                    persons_train.append(person_id)
                    for pkg_id in wh_to_pkgs[wh][:capacity]:
                        assignments_train.append({
                            'package_id': pkg_id,
                            'warehouse_id': wh,
                            'train_id': tid,
                            'person': person_id
                        })
                    wh_to_pkgs[wh] = wh_to_pkgs[wh][capacity:]
            if wh_to_pkgs[wh]:
                leftovers_per_wh[wh] = wh_to_pkgs[wh]

        # ----------------------------
        # If all leftovers done, skip further steps
        # ----------------------------
        if not leftovers_per_wh:
            all_assignments.extend(assignments_train)
            continue

        # ----------------------------
        # STEP 1: Lower Bound
        # ----------------------------
        total_leftover = sum(len(pkgs) for pkgs in leftovers_per_wh.values())
        LB = math.ceil(total_leftover / capacity)

        # ----------------------------
        # STEP 2: Zones
        # ----------------------------
        zone_map = warehouses.set_index('warehouse_id')['zone'].to_dict()
        cluster_map = warehouses.set_index('warehouse_id')['cluster'].to_dict()

        # zone -> list of warehouses
        zones = defaultdict(list)
        for wh, zone in zone_map.items():
            if wh in leftovers_per_wh:
                zones[zone].append(wh)

        zone_need = {}
        for zone, whs in zones.items():
            cnt = sum(len(leftovers_per_wh[wh]) for wh in whs)
            zone_need[zone] = math.ceil(cnt / capacity)
        Z_cost = sum(zone_need.values())

        # ----------------------------
        # STEP 3 & 4: Cluster evaluation
        # ----------------------------
        clusters = defaultdict(list)
        for wh, cl in cluster_map.items():
            if wh in leftovers_per_wh:
                clusters[cl].append(wh)
        cluster_need = {}
        for cl, whs in clusters.items():
            cnt = sum(len(leftovers_per_wh[wh]) for wh in whs)
            cluster_need[cl] = math.ceil(cnt / capacity)
        C_cost = sum(cluster_need.values())

        # Decide which strategy to use
        # Step 0 always done. Now:
        # priority: Step0 → Zones → Zone+Cluster combination (if better) → Cluster-only

        # Prepare bins for assignment
        remaining_persons_needed = max(LB - len(persons_train), 0)
        bins = []

        # ----------------------------
        # Assign Zones if Z_cost <= LB or Cluster combo doesn't reduce
        # ----------------------------
        if Z_cost <= LB or C_cost >= LB:
            # Assign zones in CSV order of warehouses
            for zone in sorted(zones.keys()):
                for wh in zones[zone]:
                    pkgs = leftovers_per_wh[wh]
                    while pkgs:
                        person_id = f"Hc{len(persons_train)+1}_{tid}"
                        persons_train.append(person_id)
                        take = pkgs[:capacity]
                        pkgs = pkgs[capacity:]
                        leftovers_per_wh[wh] = pkgs
                        for pkg_id in take:
                            assignments_train.append({
                                'package_id': pkg_id,
                                'warehouse_id': wh,
                                'train_id': tid,
                                'person': person_id
                            })
            all_assignments.extend(assignments_train)
            continue

        # ----------------------------
        # Zone+Cluster combination to get closer to LB
        # ----------------------------
        # Simple greedy: iterate clusters in CSV order
        for cl in sorted(clusters.keys()):
            whs = clusters[cl]
            cluster_pkgs = []
            for wh in whs:
                cluster_pkgs.extend(leftovers_per_wh[wh])
            if cluster_pkgs:
                person_id = f"Hc{len(persons_train)+1}_{tid}"
                persons_train.append(person_id)
                assigned = cluster_pkgs[:capacity]
                remaining = cluster_pkgs[capacity:]
                idx = 0
                # assign to warehouses in order
                for wh in whs:
                    to_assign = []
                    for pkg_id in leftovers_per_wh[wh]:
                        if pkg_id in assigned:
                            to_assign.append(pkg_id)
                    for pkg_id in to_assign:
                        assignments_train.append({
                            'package_id': pkg_id,
                            'warehouse_id': wh,
                            'train_id': tid,
                            'person': person_id
                        })
                        leftovers_per_wh[wh].remove(pkg_id)
        # ----------------------------
        # Any remaining leftovers, assign cluster-only
        # ----------------------------
        for wh in warehouse_order:
            if wh in leftovers_per_wh:
                pkgs = leftovers_per_wh[wh]
                while pkgs:
                    person_id = f"Hc{len(persons_train)+1}_{tid}"
                    persons_train.append(person_id)
                    take = pkgs[:capacity]
                    pkgs = pkgs[capacity:]
                    leftovers_per_wh[wh] = pkgs
                    for pkg_id in take:
                        assignments_train.append({
                            'package_id': pkg_id,
                            'warehouse_id': wh,
                            'train_id': tid,
                            'person': person_id
                        })

        all_assignments.extend(assignments_train)

    # ----------------------------
    # Build summary dataframe
    # ----------------------------
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
        'total_persons': len(persons_train)
    }

    return assignments_df, summary_df, per_train_detail, metadata
