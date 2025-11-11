# simulation/human_assignment.py

import math
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


def _ceil_div(a, b):
    return (a + b - 1) // b


def _pack_wh_into_bins(warehouse_list, leftovers_per_wh, capacity):
    """
    Pack whole warehouses (each with len(pkgs) < capacity) into bins (persons) preserving warehouse_list order.
    Strategy:
      - For each warehouse in order, try to place the whole warehouse into the first bin where it fits.
      - If none fits, create a new bin and put the whole warehouse there.
    Returns list of bins; each bin is dict: {'warehouses': [wh1, wh2...], 'count': total_pkgs}
    """
    bins = []
    for wh in warehouse_list:
        pkgs = leftovers_per_wh.get(wh, [])
        if not pkgs:
            continue
        placed = False
        for b in bins:
            if b['count'] + len(pkgs) <= capacity:
                b['warehouses'].append(wh)
                b['count'] += len(pkgs)
                placed = True
                break
        if not placed:
            # start a new bin
            bins.append({'warehouses': [wh], 'count': len(pkgs)})
    return bins


def assign_packages(packages_df, trains_df, warehouses_df, capacity):
    """
    Returns: assignments_df, summary_df, per_train_detail, metadata
    - assignments_df: rows of (package_id, warehouse_id, train_id, person)
    - summary_df: aggregated persons per train/warehouse (divided by capacity as earlier)
    - per_train_detail: dict{train_id: DataFrame(warehouse, person, packages, count)}
    - metadata: dict
    """
    packages = packages_df.copy().reset_index(drop=True)
    trains = trains_df.copy().reset_index(drop=True)
    warehouses = warehouses_df.copy().reset_index(drop=True)

    # Ensure required columns exist
    if 'warehouse_id' not in warehouses.columns or 'zone' not in warehouses.columns or 'cluster' not in warehouses.columns:
        raise ValueError("warehouses_df must contain 'warehouse_id', 'zone', and 'cluster' columns")

    packages['package_id'] = packages['package_id'].astype(str)
    packages['warehouse_id'] = packages['warehouse_id'].astype(str)
    packages['train_id'] = packages['package_id'].apply(lambda pid: infer_train_id_from_pkg(pid, trains))

    all_assignments = []
    train_groups = packages.groupby('train_id')

    # preserve warehouse CSV order
    warehouse_order = list(warehouses['warehouse_id'])

    # precompute maps
    zone_map = warehouses.set_index('warehouse_id')['zone'].to_dict()
    cluster_map = warehouses.set_index('warehouse_id')['cluster'].to_dict()

    for tid, grp in train_groups:
        if len(grp) == 0:
            continue

        # build per-warehouse package lists (preserve package order as in input)
        wh_to_pkgs = {wh: [] for wh in warehouse_order}
        for _, row in grp.iterrows():
            wh = row['warehouse_id']
            if wh not in wh_to_pkgs:
                # if package refers to a warehouse not in CSV, still include it appended
                wh_to_pkgs.setdefault(wh, [])
            wh_to_pkgs[wh].append(row['package_id'])

        # STEP 0: Workshop direct allocation (warehouses with >= capacity)
        assignments_train = []
        person_counter = 1  # per train
        leftovers = {}  # wh -> list(pkg_ids) after removing full-capacity chunks

        for wh in warehouse_order:
            pkgs = wh_to_pkgs.get(wh, [])
            if not pkgs:
                continue
            if len(pkgs) >= capacity:
                full_bins = len(pkgs) // capacity
                for b in range(full_bins):
                    person = f"Hc{person_counter}_{tid}"
                    slice_pkgs = pkgs[b*capacity:(b+1)*capacity]
                    for pkg in slice_pkgs:
                        assignments_train.append({
                            'package_id': pkg, 'warehouse_id': wh, 'train_id': tid, 'person': person
                        })
                    person_counter += 1
                remain = len(pkgs) % capacity
                if remain:
                    leftovers[wh] = pkgs[-remain:]
            else:
                leftovers[wh] = pkgs.copy()

        # Also include warehouses not present in CSV order but present in wh_to_pkgs (edge cases)
        for wh, pkgs in wh_to_pkgs.items():
            if wh not in warehouse_order and pkgs:
                if len(pkgs) >= capacity:
                    full_bins = len(pkgs) // capacity
                    for b in range(full_bins):
                        person = f"Hc{person_counter}_{tid}"
                        slice_pkgs = pkgs[b*capacity:(b+1)*capacity]
                        for pkg in slice_pkgs:
                            assignments_train.append({
                                'package_id': pkg, 'warehouse_id': wh, 'train_id': tid, 'person': person
                            })
                        person_counter += 1
                    remain = len(pkgs) % capacity
                    if remain:
                        leftovers[wh] = pkgs[-remain:]
                else:
                    leftovers[wh] = pkgs.copy()

        # If no leftovers, proceed to next train
        if not any(leftovers.values()):
            all_assignments.extend(assignments_train)
            continue

        # STEP 1: compute LB
        total_leftover = sum(len(pkgs) for pkgs in leftovers.values())
        LB = _ceil_div(total_leftover, capacity)

        # build zone->warehouses (only those with leftovers) preserving CSV order
        zone_wh_ordered = defaultdict(list)
        for wh in warehouse_order:
            if wh in leftovers and leftovers[wh]:
                zone_wh_ordered[zone_map[wh]].append(wh)
        # also add any leftover warehouses not in warehouse_order
        for wh in leftovers:
            if wh not in warehouse_order and leftovers[wh]:
                zone_wh_ordered[zone_map.get(wh, None)].append(wh)

        # compute zone_need
        zone_need = {}
        for z, whs in zone_wh_ordered.items():
            cnt = sum(len(leftovers[wh]) for wh in whs)
            zone_need[z] = _ceil_div(cnt, capacity)
        Z_cost = sum(zone_need.values())

        # If zones satisfy LB (or are best), perform zone-first assignment
        use_plan = None  # 'zone', 'cluster', or 'hybrid'
        chosen_plan = None  # details for hybrid/cluster if needed

        if Z_cost <= LB:
            use_plan = 'zone'
        else:
            # compute clusters for leftovers
            cluster_wh_ordered = defaultdict(list)
            # preserve CSV order inside clusters
            for wh in warehouse_order:
                if wh in leftovers and leftovers[wh]:
                    cluster_wh_ordered[cluster_map[wh]].append(wh)
            for wh in leftovers:
                if wh not in warehouse_order and leftovers[wh]:
                    cluster_wh_ordered[cluster_map.get(wh, None)].append(wh)

            cluster_need = {}
            for c, whs in cluster_wh_ordered.items():
                cnt = sum(len(leftovers[wh]) for wh in whs)
                cluster_need[c] = _ceil_div(cnt, capacity)
            C_cost = sum(cluster_need.values())

            if C_cost < Z_cost:
                use_plan = 'cluster'
            elif C_cost > Z_cost:
                # zones are better; stick to zone priority (as requested)
                use_plan = 'zone'
            else:
                # C_cost == Z_cost (> LB) -> try hybrid combinations to reduce below Z_cost
                # Evaluate subsets of clusters and compute combined cost
                # Prepare cluster sets (set of warehouses)
                cluster_items = [(c, set(whs)) for c, whs in cluster_wh_ordered.items()]
                best = {'cost': Z_cost, 'clusters': None}
                # Consider all subsets of clusters (power set)
                n = len(cluster_items)
                # Only consider non-empty subsets (including empty will be zones only)
                for r in range(1, n+1):
                    for subset in combinations(cluster_items, r):
                        selected_clusters = [c for c, _ in subset]
                        selected_wh = set().union(*[s for _, s in subset])
                        # Ensure there is no overlap? clusters ideally are disjoint, but just continue if overlap present
                        # Compute cluster cost for selected clusters
                        cluster_cost_sel = 0
                        for csel, sset in subset:
                            cnt = sum(len(leftovers.get(wh, [])) for wh in sset)
                            cluster_cost_sel += _ceil_div(cnt, capacity)
                        # Remaining warehouses (not in selected clusters)
                        remaining_wh = [wh for wh in leftovers.keys() if leftovers[wh] and wh not in selected_wh]
                        # compute zone cost for remaining warehouses
                        zone_remaining = defaultdict(list)
                        for wh in remaining_wh:
                            zone_remaining[zone_map[wh]].append(wh)
                        zone_cost_rem = sum(_ceil_div(sum(len(leftovers[wh]) for wh in whs), capacity) for whs in zone_remaining.values())
                        total_cost = cluster_cost_sel + zone_cost_rem
                        if total_cost < best['cost']:
                            best = {'cost': total_cost, 'clusters': selected_clusters}
                if best['clusters'] and best['cost'] < Z_cost:
                    use_plan = 'hybrid'
                    chosen_plan = best
                else:
                    # hybrid doesn't help; fall back to zone priority
                    use_plan = 'zone'

        # ---------- Perform assignment according to decided plan ----------
        # Helper: assign bins (persons) given a list of warehouses (in order)
        def _assign_bins_for_wh_list(wh_list, leftovers_map, cur_person_counter):
            """
            Pack whole warehouses preserving order into bins (persons) and return assignments and updated counter.
            Returns: list of dicts (pkg assignment rows), new_counter
            """
            rows = []
            bins = _pack_wh_into_bins(wh_list, leftovers_map, capacity)
            # If some warehouse's pkgs > capacity (shouldn't happen here), we slice it (defensive)
            for b in bins:
                person = f"Hc{cur_person_counter}_{tid}"
                cur_person_counter += 1
                for wh in b['warehouses']:
                    pkgs = leftovers_map.get(wh, [])
                    # assign all pkgs of this warehouse (they should all be <= capacity sum in bin)
                    for pkg in pkgs:
                        rows.append({
                            'package_id': pkg, 'warehouse_id': wh, 'train_id': tid, 'person': person
                        })
                    # mark as assigned
                    leftovers_map[wh] = []
            return rows, cur_person_counter

        # Zone-first assignment
        if use_plan == 'zone':
            # iterate zones in stable order (sorted by zone name to be deterministic)
            for zone in sorted(zone_wh_ordered.keys()):
                whs = zone_wh_ordered[zone]
                if not whs:
                    continue
                rows, person_counter = _assign_bins_for_wh_list(whs, leftovers, person_counter)
                assignments_train.extend(rows)

        elif use_plan == 'cluster':
            # assign by cluster order (sorted by cluster name for determinism)
            # collect cluster_wh_ordered as earlier
            cluster_wh_ordered = defaultdict(list)
            for wh in warehouse_order:
                if wh in leftovers and leftovers[wh]:
                    cluster_wh_ordered[cluster_map[wh]].append(wh)
            for cluster in sorted(cluster_wh_ordered.keys()):
                whs = cluster_wh_ordered[cluster]
                if not whs:
                    continue
                rows, person_counter = _assign_bins_for_wh_list(whs, leftovers, person_counter)
                assignments_train.extend(rows)

        elif use_plan == 'hybrid':
            # chosen_plan has 'clusters' selected; assign those clusters first by cluster order
            selected_clusters = chosen_plan['clusters']
            # assign selected clusters
            for cluster in sorted(selected_clusters):
                whs = [wh for wh in warehouse_order if wh in leftovers and cluster_map[wh] == cluster]
                if not whs:
                    continue
                rows, person_counter = _assign_bins_for_wh_list(whs, leftovers, person_counter)
                assignments_train.extend(rows)
            # assign remaining by zones
            # rebuild zone_wh_ordered (only leftovers remain)
            zone_wh_ordered = defaultdict(list)
            for wh in warehouse_order:
                if wh in leftovers and leftovers[wh]:
                    zone_wh_ordered[zone_map[wh]].append(wh)
            for zone in sorted(zone_wh_ordered.keys()):
                whs = zone_wh_ordered[zone]
                if not whs:
                    continue
                rows, person_counter = _assign_bins_for_wh_list(whs, leftovers, person_counter)
                assignments_train.extend(rows)
        else:
            # fallback: zone
            for zone in sorted(zone_wh_ordered.keys()):
                whs = zone_wh_ordered[zone]
                if not whs:
                    continue
                rows, person_counter = _assign_bins_for_wh_list(whs, leftovers, person_counter)
                assignments_train.extend(rows)

        # Defensive check: any remaining pkgs left unassigned? They shouldn't be, but handle by slicing
        for wh, pkgs in list(leftovers.items()):
            if pkgs:
                # slice into capacity-chunks
                while pkgs:
                    person = f"Hc{person_counter}_{tid}"
                    take = pkgs[:capacity]
                    pkgs = pkgs[capacity:]
                    for pkg in take:
                        assignments_train.append({
                            'package_id': pkg, 'warehouse_id': wh, 'train_id': tid, 'person': person
                        })
                    person_counter += 1
                leftovers[wh] = []

        # append train assignments
        all_assignments.extend(assignments_train)

    # -------------------- Build DataFrames --------------------
    assignments_df = pd.DataFrame(all_assignments)
    if assignments_df.empty:
        summary_df = pd.DataFrame()
        per_train_detail = {}
        metadata = {'total_packages': 0, 'capacity': capacity, 'total_persons': 0}
        return assignments_df, summary_df, per_train_detail, metadata

    # Summary: persons per train x warehouse - we keep same behaviour as original (count / capacity)
    summary_df = assignments_df.groupby(["train_id", "warehouse_id"]).size().unstack(fill_value=0)
    all_warehouses = list(warehouses['warehouse_id'])
    # keep any warehouses that might not exist in warehouses_df? ensure columns align
    for col in assignments_df['warehouse_id'].unique():
        if col not in all_warehouses:
            all_warehouses.append(col)
    summary_df = summary_df.reindex(columns=all_warehouses, fill_value=0)
    summary_df = summary_df / capacity
    summary_df = summary_df.reset_index()

    warehouse_cols = [c for c in summary_df.columns if str(c).startswith("W")]
    if warehouse_cols:
        summary_df["Total Persons"] = summary_df[warehouse_cols].sum(axis=1).apply(math.ceil).astype(int)
        summary_df[warehouse_cols] = summary_df[warehouse_cols].round(2)
    else:
        summary_df["Total Persons"] = 0

    # per_train_detail: same format as before
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
