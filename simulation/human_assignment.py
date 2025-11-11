# simulation/human_assignment.py

import math
import pandas as pd
from collections import defaultdict
from itertools import combinations, chain

def infer_train_id_from_pkg(pkg_id, trains_df):
    """Infer train ID from package ID prefix."""
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
    Assign packages to humans based on priority:
    Workshop -> Zone -> Cluster -> Zone+Cluster combination
    """
    packages = packages_df.copy().reset_index(drop=True)
    warehouses = warehouses_df.copy().reset_index(drop=True)
    trains = trains_df.copy().reset_index(drop=True)

    packages['package_id'] = packages['package_id'].astype(str)
    packages['warehouse_id'] = packages['warehouse_id'].astype(str)
    packages['train_id'] = packages['package_id'].apply(lambda pid: infer_train_id_from_pkg(pid, trains))

    all_assignments = []

    # Group packages by train
    train_groups = packages.groupby('train_id')

    for tid, grp in train_groups:
        wh_to_pkgs = defaultdict(list)
        for _, row in grp.iterrows():
            wh_to_pkgs[row['warehouse_id']].append(row['package_id'])

        assigned_packages = set()
        person_idx = 0
        persons_train = []

        # Step 0: Workshop allocations
        step0_assignments = []
        remaining_wh_packages = {}
        for wh_id, pkgs in wh_to_pkgs.items():
            if len(pkgs) >= capacity:
                n_persons = len(pkgs) // capacity
                for i in range(n_persons):
                    person = f"Hc{person_idx+1}_{tid}"
                    assigned = pkgs[i*capacity:(i+1)*capacity]
                    step0_assignments.append({
                        'person': person,
                        'packages': assigned,
                        'warehouses': [wh_id]
                    })
                    assigned_packages.update(assigned)
                    person_idx += 1
                leftover = pkgs[n_persons*capacity:]
                if leftover:
                    remaining_wh_packages[wh_id] = leftover
            else:
                remaining_wh_packages[wh_id] = pkgs.copy()

        # Step 1: Compute LB and zone needs
        total_remaining = sum(len(pkgs) for pkgs in remaining_wh_packages.values())
        LB = math.ceil(total_remaining / capacity)

        # Prepare warehouse -> zone/cluster mappings
        wh_zone_map = {}
        wh_cluster_map = {}
        for _, w in warehouses.iterrows():
            wh_zone_map[w['warehouse_id']] = w['zone']
            wh_cluster_map[w['warehouse_id']] = w['cluster']

        # Organize remaining warehouses by zone
        zone_map = defaultdict(list)
        for wh, pkgs in remaining_wh_packages.items():
            if pkgs:  # only non-empty
                zone_map[wh_zone_map[wh]].append((wh, pkgs))

        # Compute zone need
        zone_need = {}
        for zone, wh_list in zone_map.items():
            zone_total = sum(len(pkgs) for _, pkgs in wh_list)
            zone_need[zone] = math.ceil(zone_total / capacity)
        Z_cost = sum(zone_need.values())

        # Step 2: Assign Zone if Z_cost == LB
        all_person_allocations = step0_assignments.copy()
        remaining_pkgs = {wh: pkgs.copy() for wh, pkgs in remaining_wh_packages.items() if pkgs}
        if Z_cost == LB:
            # Assign zone wise
            person_idx = len(all_person_allocations)
            for zone, wh_list in zone_map.items():
                current_person_pkgs = []
                current_wh = []
                person = f"Hc{person_idx+1}_{tid}"
                count = 0
                for wh, pkgs in wh_list:
                    pkgs_to_assign = [p for p in pkgs if p not in assigned_packages]
                    if not pkgs_to_assign:
                        continue
                    if count + len(pkgs_to_assign) <= capacity:
                        current_person_pkgs.extend(pkgs_to_assign)
                        current_wh.append(wh)
                        assigned_packages.update(pkgs_to_assign)
                        count += len(pkgs_to_assign)
                    else:
                        # Assign current person
                        if current_person_pkgs:
                            all_person_allocations.append({
                                'person': person,
                                'packages': current_person_pkgs,
                                'warehouses': current_wh.copy()
                            })
                            person_idx += 1
                        # Start new person
                        person = f"Hc{person_idx+1}_{tid}"
                        current_person_pkgs = pkgs_to_assign.copy()
                        current_wh = [wh]
                        assigned_packages.update(pkgs_to_assign)
                        count = len(pkgs_to_assign)
                if current_person_pkgs:
                    all_person_allocations.append({
                        'person': person,
                        'packages': current_person_pkgs,
                        'warehouses': current_wh.copy()
                    })
            continue  # LB satisfied

        # Step 3: Cluster allocation
        # Compute cluster need
        cluster_map = defaultdict(list)
        for wh, pkgs in remaining_pkgs.items():
            cluster_map[wh_cluster_map[wh]].append((wh, pkgs))

        cluster_need = {}
        for cluster, wh_list in cluster_map.items():
            total = sum(len(pkgs) for _, pkgs in wh_list)
            cluster_need[cluster] = math.ceil(total / capacity)
        cluster_cost = sum(cluster_need.values())

        # Decision logic
        if cluster_cost < Z_cost or cluster_cost == LB:
            # Assign cluster-wise
            person_idx = len(all_person_allocations)
            for cluster, wh_list in cluster_map.items():
                current_person_pkgs = []
                current_wh = []
                person = f"Hc{person_idx+1}_{tid}"
                count = 0
                for wh, pkgs in wh_list:
                    pkgs_to_assign = [p for p in pkgs if p not in assigned_packages]
                    if not pkgs_to_assign:
                        continue
                    if count + len(pkgs_to_assign) <= capacity:
                        current_person_pkgs.extend(pkgs_to_assign)
                        current_wh.append(wh)
                        assigned_packages.update(pkgs_to_assign)
                        count += len(pkgs_to_assign)
                    else:
                        if current_person_pkgs:
                            all_person_allocations.append({
                                'person': person,
                                'packages': current_person_pkgs,
                                'warehouses': current_wh.copy()
                            })
                            person_idx += 1
                        person = f"Hc{person_idx+1}_{tid}"
                        current_person_pkgs = pkgs_to_assign.copy()
                        current_wh = [wh]
                        assigned_packages.update(pkgs_to_assign)
                        count = len(pkgs_to_assign)
                if current_person_pkgs:
                    all_person_allocations.append({
                        'person': person,
                        'packages': current_person_pkgs,
                        'warehouses': current_wh.copy()
                    })
        else:
            # Step 4: Zone+Cluster combination
            # Generate all valid non-empty combinations
            zone_keys = list(zone_map.keys())
            cluster_keys = list(cluster_map.keys())
            min_persons = None
            best_comb = None

            for z_len in range(len(zone_keys)+1):
                for c_len in range(len(cluster_keys)+1):
                    if z_len == 0 and c_len == 0:
                        continue
                    z_combs = combinations(zone_keys, z_len)
                    c_combs = combinations(cluster_keys, c_len)
                    for z_sel in z_combs:
                        for c_sel in c_combs:
                            selected_wh = set()
                            # Collect selected warehouses
                            for z in z_sel:
                                selected_wh.update([wh for wh, _ in zone_map[z]])
                            for c in c_sel:
                                selected_wh.update([wh for wh, _ in cluster_map[c]])
                            total_packages = sum(len(remaining_pkgs[wh]) for wh in selected_wh if wh in remaining_pkgs)
                            est_persons = math.ceil(total_packages / capacity)
                            if min_persons is None or est_persons < min_persons:
                                min_persons = est_persons
                                best_comb = (z_sel, c_sel)

            # Assign best combination
            if best_comb:
                person_idx = len(all_person_allocations)
                z_sel, c_sel = best_comb
                wh_to_assign = set()
                for z in z_sel:
                    wh_to_assign.update([wh for wh, _ in zone_map[z]])
                for c in c_sel:
                    wh_to_assign.update([wh for wh, _ in cluster_map[c]])
                # Assign packages respecting capacity
                current_person_pkgs = []
                current_wh = []
                person = f"Hc{person_idx+1}_{tid}"
                count = 0
                for wh in sorted(wh_to_assign, key=lambda x: x):
                    pkgs_to_assign = [p for p in remaining_pkgs[wh] if p not in assigned_packages]
                    for p in pkgs_to_assign:
                        if count < capacity:
                            current_person_pkgs.append(p)
                            if wh not in current_wh:
                                current_wh.append(wh)
                            assigned_packages.add(p)
                            count += 1
                        else:
                            all_person_allocations.append({
                                'person': person,
                                'packages': current_person_pkgs.copy(),
                                'warehouses': current_wh.copy()
                            })
                            person_idx += 1
                            person = f"Hc{person_idx+1}_{tid}"
                            current_person_pkgs = [p]
                            current_wh = [wh]
                            assigned_packages.add(p)
                            count = 1
                if current_person_pkgs:
                    all_person_allocations.append({
                        'person': person,
                        'packages': current_person_pkgs.copy(),
                        'warehouses': current_wh.copy()
                    })

        # Step 5: Build final assignments
        for alloc in all_person_allocations:
            for wh in alloc['warehouses']:
                pkgs_in_wh = [p for p in alloc['packages'] if p in wh_to_pkgs.get(wh, [])]
                if not pkgs_in_wh:
                    continue
                for p in pkgs_in_wh:
                    all_assignments.append({
                        'package_id': p,
                        'warehouse_id': wh,
                        'train_id': tid,
                        'person': alloc['person']
                    })

    assignments_df = pd.DataFrame(all_assignments)

    # Build summary and per_train_detail
    summary_df = pd.DataFrame()
    per_train_detail = {}
    if not assignments_df.empty:
        summary_df = assignments_df.groupby(['train_id', 'warehouse_id']).size().unstack(fill_value=0)
        all_warehouses = list(warehouses['warehouse_id'])
        summary_df = summary_df.reindex(columns=all_warehouses, fill_value=0)
        summary_df = summary_df / capacity
        summary_df = summary_df.reset_index()
        warehouse_cols = [c for c in summary_df.columns if str(c).startswith('W')]
        summary_df['Total Persons'] = summary_df[warehouse_cols].sum(axis=1).apply(math.ceil).astype(int)
        summary_df[warehouse_cols] = summary_df[warehouse_cols].round(2)

        for tid, grp in assignments_df.groupby('train_id'):
            detail_rows = []
            for (wh, person), g in grp.groupby(['warehouse_id', 'person']):
                pkgs = list(g['package_id'])
                detail_rows.append({'warehouse': wh, 'person': person, 'packages': pkgs, 'count': len(pkgs)})
            per_train_detail[tid] = pd.DataFrame(detail_rows).sort_values(['warehouse', 'person']).reset_index(drop=True)

    metadata = {
        'total_packages': len(packages),
        'capacity': capacity,
        'total_persons': len(set(assignments_df['person'])) if not assignments_df.empty else 0
    }

    return assignments_df, summary_df, per_train_detail, metadata
