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

            # --- DEBUG: show cost comparison before deciding allocation ---
            print(f"[DEBUG] Train: {tid}")
            print(f"  LB={LB}, Z_cost={Z_cost}, C_cost={C_cost}, Comb_cost={Comb_cost}")
            print(f"  Decision → ", end="")

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

        # Step 4: Allocate persons based on allocation plan
        for alloc_type, target, whs, num_persons in allocation_plan:
            for i in range(num_persons):
                person = f"Hc{person_idx + 1}_{tid}"
                whs_sorted = sorted(whs)
                for wh in whs_sorted:
                    for pkg in remaining_wh[wh]:
                        assignments_train.append({
                            'package_id': pkg,
                            'warehouse_id': wh,
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
