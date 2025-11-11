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
    Assign packages to humans based on priority:
    Workshop -> Zone -> Cluster -> Zone+Cluster combination.
    """
    packages = packages_df.copy().reset_index(drop=True)
    warehouses = warehouses_df.copy().reset_index(drop=True)
    trains = trains_df.copy().reset_index(drop=True)

    packages['package_id'] = packages['package_id'].astype(str)
    packages['warehouse_id'] = packages['warehouse_id'].astype(str)
    packages['train_id'] = packages['package_id'].apply(lambda pid: infer_train_id_from_pkg(pid, trains))

    all_assignments = []

    # Mapping warehouse to zone and cluster
    wh_zone_map = dict(zip(warehouses['warehouse_id'], warehouses['zone']))
    wh_cluster_map = dict(zip(warehouses['warehouse_id'], warehouses['cluster']))

    # Group packages by train
    train_groups = packages.groupby('train_id')

    for tid, grp in train_groups:
        # ------------------------------
        # STEP 0: Workshop direct allocation
        # ------------------------------
        wh_to_pkgs = defaultdict(list)
        for _, row in grp.iterrows():
            wh_to_pkgs[row['warehouse_id']].append(row['package_id'])

        assigned_packages = set()
        person_idx = 0
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

        # If all packages are assigned in step 0
        if not remaining_wh_packages:
            all_person_allocations = step0_assignments
        else:
            # ------------------------------
            # STEP 1: Compute LB
            # ------------------------------
            total_remaining = sum(len(pkgs) for pkgs in remaining_wh_packages.values())
            LB = math.ceil(total_remaining / capacity)

            # ------------------------------
            # STEP 2: Compute Z_cost
            # ------------------------------
            zone_map = defaultdict(list)
            for wh, pkgs in remaining_wh_packages.items():
                zone_map[wh_zone_map[wh]].append((wh, pkgs))

            zone_cost = sum(math.ceil(sum(len(pkgs) for _, pkgs in wh_list) / capacity)
                            for wh_list in zone_map.values())

            # ------------------------------
            # STEP 2 DECISION: Zone-wise allocation if Z_cost == LB
            # ------------------------------
            if zone_cost == LB:
                zone_assignments = []
                for zone, wh_list in zone_map.items():
                    zone_assignments += _assign_zone_or_cluster(wh_list, assigned_packages, tid, person_idx, capacity)
                    person_idx = len(set(a['person'] for a in zone_assignments))
                    assigned_packages.update(pkg for a in zone_assignments for pkg in a['packages'])
                all_person_allocations = step0_assignments + zone_assignments

            else:
                # ------------------------------
                # STEP 3: Cluster cost
                # ------------------------------
                cluster_map = defaultdict(list)
                for wh, pkgs in remaining_wh_packages.items():
                    cluster_map[wh_cluster_map[wh]].append((wh, pkgs))

                cluster_cost = sum(math.ceil(sum(len(pkgs) for _, pkgs in wh_list) / capacity)
                                   for wh_list in cluster_map.values())

                # ------------------------------
                # STEP 3 DECISION: Cluster-wise allocation
                # ------------------------------
                if cluster_cost < zone_cost or cluster_cost == LB:
                    cluster_assignments = []
                    for cluster, wh_list in cluster_map.items():
                        cluster_assignments += _assign_zone_or_cluster(wh_list, assigned_packages, tid, person_idx, capacity)
                        person_idx = len(set(a['person'] for a in cluster_assignments))
                        assigned_packages.update(pkg for a in cluster_assignments for pkg in a['packages'])
                    all_person_allocations = step0_assignments + cluster_assignments

                else:
                    # ------------------------------
                    # STEP 4: Zone+Cluster combination
                    # Only try if it reduces total persons
                    # Otherwise fallback to Zone-wise
                    # ------------------------------
                    combined_assignments, combination_cost = _try_zone_cluster_combination(zone_map, cluster_map, assigned_packages, tid, person_idx, capacity)

                    if combination_cost < zone_cost:
                        all_person_allocations = step0_assignments + combined_assignments
                    else:
                        # fallback to Zone-wise
                        zone_assignments = []
                        for zone, wh_list in zone_map.items():
                            zone_assignments += _assign_zone_or_cluster(wh_list, assigned_packages, tid, person_idx, capacity)
                            person_idx = len(set(a['person'] for a in zone_assignments))
                            assigned_packages.update(pkg for a in zone_assignments for pkg in a['packages'])
                        all_person_allocations = step0_assignments + zone_assignments

        # ------------------------------
        # BUILD final assignments_df
        # ------------------------------
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


# ------------------------------
# HELPER FUNCTIONS
# ------------------------------

def _assign_zone_or_cluster(wh_list, assigned_packages, tid, person_idx, capacity):
    """
    Assign packages per zone or cluster, respecting warehouse order
    and per-person capacity.
    Returns list of allocations.
    """
    allocations = []
    current_pkg_count = 0
    current_person_pkgs = []
    current_wh = []
    person_counter = person_idx

    for wh_id, pkgs in wh_list:
        pkgs_to_assign = [p for p in pkgs if p not in assigned_packages]
        if not pkgs_to_assign:
            continue
        if current_pkg_count + len(pkgs_to_assign) <= capacity:
            current_pkg_count += len(pkgs_to_assign)
            current_person_pkgs.extend(pkgs_to_assign)
            current_wh.append(wh_id)
            assigned_packages.update(pkgs_to_assign)
        else:
            if current_person_pkgs:
                allocations.append({
                    'person': f"Hc{person_counter+1}_{tid}",
                    'packages': current_person_pkgs.copy(),
                    'warehouses': current_wh.copy()
                })
                person_counter += 1
            # start new person
            current_person_pkgs = pkgs_to_assign.copy()
            current_wh = [wh_id]
            current_pkg_count = len(pkgs_to_assign)
            assigned_packages.update(pkgs_to_assign)
    if current_person_pkgs:
        allocations.append({
            'person': f"Hc{person_counter+1}_{tid}",
            'packages': current_person_pkgs.copy(),
            'warehouses': current_wh.copy()
        })
    return allocations


def _try_zone_cluster_combination(zone_map, cluster_map, assigned_packages, tid, person_idx, capacity):
    """
    Try a zone+cluster combination that reduces total persons.
    Returns allocations and combination cost.
    """
    combined_allocations = []
    remaining_packages = set()
    for wh_list in zone_map.values():
        for wh, pkgs in wh_list:
            remaining_packages.update(pkgs)
    for wh_list in cluster_map.values():
        for wh, pkgs in wh_list:
            remaining_packages.update(pkgs)

    # Simple greedy approach: assign per cluster first, then fill zones
    person_counter = person_idx
    temp_assigned = set()
    for cluster, wh_list in cluster_map.items():
        allocs = _assign_zone_or_cluster(wh_list, temp_assigned, tid, person_counter, capacity)
        combined_allocations.extend(allocs)
        person_counter = len(set(a['person'] for a in combined_allocations))
        temp_assigned.update(pkg for a in allocs for pkg in a['packages'])

    # Now fill remaining packages zone-wise
    for zone, wh_list in zone_map.items():
        allocs = _assign_zone_or_cluster(wh_list, temp_assigned, tid, person_counter, capacity)
        combined_allocations.extend(allocs)
        person_counter = len(set(a['person'] for a in combined_allocations))
        temp_assigned.update(pkg for a in allocs for pkg in a['packages'])

    combination_cost = len(set(a['person'] for a in combined_allocations))
    return combined_allocations, combination_cost
