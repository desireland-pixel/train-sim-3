# simulation/human_assignment.py

import math
import pandas as pd
from collections import defaultdict

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
    Assign packages to humans based on workshop -> zone -> zone+cluster -> cluster priority.
    Capacity is max packages per person.
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
        # Step 0: Workshop allocation (if any warehouse has >= capacity)
        wh_to_pkgs = defaultdict(list)
        for _, row in grp.iterrows():
            wh_to_pkgs[row['warehouse_id']].append(row['package_id'])

        assigned_packages = set()
        person_idx = 0
        persons_train = []

        # Step 0: Direct warehouse allocations
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

        # Organize remaining warehouses by zone
        zone_map = defaultdict(list)
        wh_zone_map = {}
        wh_cluster_map = {}
        for _, w in warehouses.iterrows():
            wh_zone_map[w['warehouse_id']] = w['zone']
            wh_cluster_map[w['warehouse_id']] = w['cluster']

        for wh, pkgs in remaining_wh_packages.items():
            zone_map[wh_zone_map[wh]].append((wh, pkgs))

        # Step 2: Zone allocation
        zone_assignments = []
        for zone, wh_list in zone_map.items():
            # Assign whole warehouses to person until capacity
            zone_person_pkgs = []
            current_pkg_count = 0
            person = f"Hc{person_idx+1}_{tid}"
            current_wh = []
            for wh_id, pkgs in wh_list:
                # Skip already assigned
                pkgs_to_assign = [p for p in pkgs if p not in assigned_packages]
                if not pkgs_to_assign:
                    continue
                if current_pkg_count + len(pkgs_to_assign) <= capacity:
                    current_pkg_count += len(pkgs_to_assign)
                    zone_person_pkgs.extend(pkgs_to_assign)
                    current_wh.append(wh_id)
                    assigned_packages.update(pkgs_to_assign)
                else:
                    # Cannot fit all packages, assign current person and start new
                    if zone_person_pkgs:
                        zone_assignments.append({
                            'person': person,
                            'packages': zone_person_pkgs,
                            'warehouses': current_wh.copy()
                        })
                        person_idx += 1
                    # Start new person
                    person = f"Hc{person_idx+1}_{tid}"
                    zone_person_pkgs = pkgs_to_assign.copy()
                    current_pkg_count = len(pkgs_to_assign)
                    current_wh = [wh_id]
                    assigned_packages.update(pkgs_to_assign)
            if zone_person_pkgs:
                zone_assignments.append({
                    'person': person,
                    'packages': zone_person_pkgs,
                    'warehouses': current_wh.copy()
                })
                person_idx += 1

        # Combine step0 and zone allocations
        all_person_allocations = step0_assignments + zone_assignments

        # Step 3: Cluster allocation (only if LB not satisfied)
        total_assigned = sum(len(a['packages']) for a in all_person_allocations)
        if total_assigned < len(grp):
            # Remaining packages
            remaining_pkgs = [p for p in grp['package_id'].tolist() if p not in assigned_packages]
            if remaining_pkgs:
                cluster_map = defaultdict(list)
                for wh, pkgs in remaining_wh_packages.items():
                    unassigned = [p for p in pkgs if p not in assigned_packages]
                    if unassigned:
                        cluster_map[wh_cluster_map[wh]].append((wh, unassigned))
                # Assign per cluster
                for cluster, wh_list in cluster_map.items():
                    cluster_person_pkgs = []
                    current_pkg_count = 0
                    person = f"Hc{person_idx+1}_{tid}"
                    current_wh = []
                    for wh_id, pkgs in wh_list:
                        pkgs_to_assign = [p for p in pkgs if p not in assigned_packages]
                        if not pkgs_to_assign:
                            continue
                        if current_pkg_count + len(pkgs_to_assign) <= capacity:
                            current_pkg_count += len(pkgs_to_assign)
                            cluster_person_pkgs.extend(pkgs_to_assign)
                            current_wh.append(wh_id)
                            assigned_packages.update(pkgs_to_assign)
                        else:
                            # Assign current person and start new
                            if cluster_person_pkgs:
                                all_person_allocations.append({
                                    'person': person,
                                    'packages': cluster_person_pkgs,
                                    'warehouses': current_wh.copy()
                                })
                                person_idx += 1
                            # Start new person
                            person = f"Hc{person_idx+1}_{tid}"
                            cluster_person_pkgs = pkgs_to_assign.copy()
                            current_pkg_count = len(pkgs_to_assign)
                            current_wh = [wh_id]
                            assigned_packages.update(pkgs_to_assign)
                    if cluster_person_pkgs:
                        all_person_allocations.append({
                            'person': person,
                            'packages': cluster_person_pkgs,
                            'warehouses': current_wh.copy()
                        })
                        person_idx += 1

        # Build assignments_df
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
