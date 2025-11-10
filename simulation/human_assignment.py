# simulation/human_assignment.py

import math
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

    # Precompute zone and cluster maps
    zone_map = warehouses.set_index('warehouse_id')['zone'].to_dict()
    cluster_map = warehouses.set_index('warehouse_id')['cluster'].to_dict()

    for tid, grp in train_groups:
        if len(grp) == 0:
            continue

        # Map warehouse → list of packages
        wh_to_pkgs = defaultdict(list)
        for _, row in grp.iterrows():
            wh_to_pkgs[row['warehouse_id']].append(row['package_id'])

        wh_counts = {wh: len(pkgs) for wh, pkgs in wh_to_pkgs.items()}

        assignments_train = []
        person_counter = 1

        # --------------------
        # STEP 0: Workshop direct allocation
        # --------------------
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

        if not leftovers_per_wh:
            all_assignments.extend(assignments_train)
            continue

        # --------------------
        # Compute LB
        # --------------------
        total_leftover = sum(len(pkgs) for pkgs in leftovers_per_wh.values())
        LB = math.ceil(total_leftover / capacity)

        # --------------------
        # STEP 1: Zone-only allocation
        # --------------------
        zone_to_wh = defaultdict(list)
        for wh in leftovers_per_wh.keys():
            zone_to_wh[zone_map[wh]].append(wh)

        zone_need = {}
        for z, wh_list in zone_to_wh.items():
            total = sum(len(leftovers_per_wh[wh]) for wh in wh_list)
            zone_need[z] = math.ceil(total / capacity)

        Z_cost = sum(zone_need.values())

        # If zone-only allocation meets LB or hybrid cannot improve, use Zones
        if Z_cost <= LB:
            # Assign persons by zones only
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
            all_assignments.extend(assignments_train)
            continue

        # --------------------
        # STEP 2: Evaluate Cluster-only allocation
        # --------------------
        cluster_to_wh = defaultdict(list)
        for wh in leftovers_per_wh.keys():
            cluster_to_wh[cluster_map[wh]].append(wh)

        cluster_need = {}
        for c, wh_list in cluster_to_wh.items():
            total = sum(len(leftovers_per_wh[wh]) for wh in wh_list)
            cluster_need[c] = math.ceil(total / capacity)

        C_cost = sum(cluster_need.values())

        # --------------------
        # STEP 3: Hybrid allocation (Clusters + Zones) - only if it reduces total persons
        # --------------------
        # Assign clusters first, then leftover zones
        hybrid_assignments = []
        total_persons_hybrid = 0
        remaining_wh = set(leftovers_per_wh.keys())

        # Assign clusters
        for c, wh_list in cluster_to_wh.items():
            cluster_pkgs = []
            for wh in wh_list:
                cluster_pkgs.extend(leftovers_per_wh[wh])
            if cluster_pkgs:
                total_persons_hybrid += math.ceil(len(cluster_pkgs) / capacity)
        # Compute leftover warehouses not covered by clusters
        remaining_wh_after_clusters = set()
        for wh in leftovers_per_wh.keys():
            if all(wh not in cluster_to_wh[c] for c in cluster_to_wh):
                remaining_wh_after_clusters.add(wh)
        for z, wh_list in zone_to_wh.items():
            count_left = sum(len(leftovers_per_wh[wh]) for wh in wh_list if wh in remaining_wh_after_clusters)
            total_persons_hybrid += math.ceil(count_left / capacity)

        # Only use hybrid if total_persons_hybrid < Z_cost
        if total_persons_hybrid < Z_cost:
            # Assign clusters
            for c, wh_list in cluster_to_wh.items():
                cluster_pkgs = []
                for wh in wh_list:
                    cluster_pkgs.extend(leftovers_per_wh[wh])
                for i in range(0, len(cluster_pkgs), capacity):
                    person = f"Hc{person_counter}_{tid}"
                    person_counter += 1
                    for pkg_id in cluster_pkgs[i:i+capacity]:
                        wh_for_pkg = [wh for wh in wh_list if pkg_id in leftovers_per_wh[wh]][0]
                        hybrid_assignments.append({
                            'package_id': pkg_id,
                            'warehouse_id': wh_for_pkg,
                            'train_id': tid,
                            'person': person
                        })
            # Assign leftover zones
            zone_remaining_to_wh = defaultdict(list)
            for wh in leftovers_per_wh.keys():
                if all(wh not in cluster_to_wh[c] for c in cluster_to_wh):
                    zone_remaining_to_wh[zone_map[wh]].append(wh)
            for z, wh_list in zone_remaining_to_wh.items():
                for wh in wh_list:
                    pkgs = leftovers_per_wh[wh]
                    for i in range(0, len(pkgs), capacity):
                        person = f"Hc{person_counter}_{tid}"
                        person_counter += 1
                        for pkg_id in pkgs[i:i+capacity]:
                            hybrid_assignments.append({
                                'package_id': pkg_id,
                                'warehouse_id': wh,
                                'train_id': tid,
                                'person': person
                            })
            assignments_train.extend(hybrid_assignments)
        else:
            # Fallback: assign zones only
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

        all_assignments.extend(assignments_train)

    # --------------------
    # Build summary and per-train detail
    # --------------------
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
    summary_df["Total Persons"] = summary_df[warehouse_cols].sum(axis=1).apply(math.ceil).astype(int)
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
