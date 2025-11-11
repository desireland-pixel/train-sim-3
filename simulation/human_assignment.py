# simulation/human_assignment.py
import math
import pandas as pd
from collections import defaultdict
from itertools import chain, combinations

def infer_train_id_from_pkg(pkg_id, trains_df):
    try:
        prefix = str(pkg_id)[:2]
        idx = int(prefix)
        if 1 <= idx <= len(trains_df):
            return trains_df.iloc[idx - 1]['train_id']
    except Exception:
        pass
    return "UNKNOWN"


def _powerset(iterable):
    "Return powerset of iterable (all subsets). Useful for enumerating cluster combinations."
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def _allocate_group_preserve_wh_order(wh_list, remaining_wh_pkgs, capacity, tid, start_person_idx):
    """
    Allocate packages for a group (zone or cluster) consisting of warehouses `wh_list`.
    - Try to assign whole warehouses to persons in CSV order, filling each person up to capacity.
    - Only split a warehouse across persons when unavoidable (i.e. leftover packages remain after attempting whole-warehouse placement).
    Returns:
        assignments (list of dict), next_person_idx (int), updated remaining_wh_pkgs (modified in place)
    """
    assignments = []
    person_idx = start_person_idx

    # Make shallow copy of counts for iteration
    # remaining_wh_pkgs maps wh -> list(pkg_id) (mutable)
    # We'll mutate remaining_wh_pkgs to remove assigned pkgs
    # Compute total and persons needed
    total_pkgs = sum(len(remaining_wh_pkgs.get(wh, [])) for wh in wh_list)
    if total_pkgs == 0:
        return assignments, person_idx

    num_persons_needed = math.ceil(total_pkgs / capacity)

    # We'll fill persons sequentially. For each person:
    #  - take whole warehouses while they fit in person's remaining capacity
    #  - if next warehouse doesn't fit and person already has something -> move to next person
    #  - after whole-warehouse pass, if some warehouses still have pkgs, we perform a final pass to split remaining pkgs across persons
    wh_queue = [wh for wh in wh_list if len(remaining_wh_pkgs.get(wh, [])) > 0]

    # First pass: allocate whole warehouses greedily into persons
    for _ in range(num_persons_needed):
        person_id = f"Hc{person_idx+1}_{tid}"
        cap_left = capacity
        made_assignment = False
        # traverse queue and try to take whole warehouses that fit
        new_queue = []
        for wh in wh_queue:
            pkgs = remaining_wh_pkgs.get(wh, [])
            if not pkgs:
                continue
            cnt = len(pkgs)
            if cnt <= cap_left:
                # Assign whole warehouse to this person (all remaining pkgs for wh)
                for pkg in pkgs:
                    assignments.append({
                        'package_id': pkg,
                        'warehouse_id': wh,
                        'train_id': tid,
                        'person': person_id
                    })
                cap_left -= cnt
                remaining_wh_pkgs[wh] = []  # consumed
                made_assignment = True
            else:
                # can't fit whole warehouse now, keep for later/pass
                new_queue.append(wh)
        wh_queue = new_queue
        person_idx += 1
        # continue to next person

    # After whole-warehouse allocation, there might still be warehouses with pkgs (if they were larger than any single capacity leftover)
    # Do a final pass to allocate remaining pkgs (split across persons as needed).
    # Recompute remaining pkgs total and assign sequentially filling persons to capacity.
    remaining_pkgs_flat = []
    remaining_wh_for_split = []
    for wh in wh_list:
        pkgs = remaining_wh_pkgs.get(wh, [])
        if pkgs:
            remaining_wh_for_split.append((wh, pkgs.copy()))
            remaining_pkgs_flat.extend([(wh, p) for p in pkgs])

    if remaining_pkgs_flat:
        # Determine how many persons already used (person_idx currently is start_person_idx + num_persons_needed)
        # We'll re-create persons starting from (start_person_idx) to fill exact capacities while avoiding duplication of person ids used earlier.
        # However since we already created persons in first pass, we should continue with same numbering but ensure we don't duplicate
        # the same person ids twice. Simpler: compute persons needed = ceil(total_remaining / capacity) and create persons new for the split
        total_remaining = len(remaining_pkgs_flat)
        num_persons_for_remaining = math.ceil(total_remaining / capacity)
        # Assign sequentially
        flat_idx = 0
        for i in range(num_persons_for_remaining):
            person_id = f"Hc{person_idx+1}_{tid}"
            person_idx += 1
            take = remaining_pkgs_flat[flat_idx: flat_idx + capacity]
            flat_idx += capacity
            for wh, pkg in take:
                assignments.append({
                    'package_id': pkg,
                    'warehouse_id': wh,
                    'train_id': tid,
                    'person': person_id
                })
                # remove assigned pkg from remaining_wh_pkgs
                if pkg in remaining_wh_pkgs[wh]:
                    remaining_wh_pkgs[wh].remove(pkg)
    # return
    return assignments, person_idx


def assign_packages(packages_df, trains_df, warehouses_df, capacity):
    """
    Returns: assignments_df, summary_df, per_train_detail, metadata
    - assignments_df: rows of package->warehouse->person
    - per_train_detail: dict train_id -> DataFrame(rows: warehouse, person, packages, count)
    """
    packages = packages_df.copy().reset_index(drop=True)
    trains = trains_df.copy().reset_index(drop=True)
    warehouses = warehouses_df.copy().reset_index(drop=True)

    # normalize types
    packages['package_id'] = packages['package_id'].astype(str)
    packages['warehouse_id'] = packages['warehouse_id'].astype(str)
    packages['train_id'] = packages['package_id'].apply(lambda pid: infer_train_id_from_pkg(pid, trains))

    # Prepare maps and warehouse order (CSV order)
    warehouse_order = list(warehouses['warehouse_id'])
    wh_zone_map = warehouses.set_index('warehouse_id')['zone'].to_dict()
    wh_cluster_map = warehouses.set_index('warehouse_id')['cluster'].to_dict()

    all_assignments = []
    train_groups = packages.groupby('train_id')

    for tid, grp in train_groups:
        if len(grp) == 0:
            continue

        # Build wh -> list(pkg) in CSV order
        wh_to_pkgs = {wh: [] for wh in warehouse_order}
        for _, row in grp.iterrows():
            wh = row['warehouse_id']
            if wh not in wh_to_pkgs:
                # if CSV didn't list it (defensive), add at end
                wh_to_pkgs[wh] = []
                warehouse_order.append(wh)
            wh_to_pkgs[wh].append(row['package_id'])

        # Step 0: workshop direct allocation (consume full capacity chunks per warehouse)
        assignments_train = []
        person_counter = 0  # counts persons assigned for this train, used in Hc numbering

        remaining_wh_pkgs = {}
        for wh in warehouse_order:
            pkgs = wh_to_pkgs.get(wh, [])
            if not pkgs:
                continue
            # full persons for this warehouse
            if len(pkgs) >= capacity:
                full = len(pkgs) // capacity
                for f in range(full):
                    person_id = f"Hc{person_counter+1}_{tid}"
                    person_counter += 1
                    chunk = pkgs[f*capacity:(f+1)*capacity]
                    for pkg in chunk:
                        assignments_train.append({
                            'package_id': pkg,
                            'warehouse_id': wh,
                            'train_id': tid,
                            'person': person_id
                        })
                leftover = len(pkgs) % capacity
                if leftover:
                    remaining_wh_pkgs[wh] = pkgs[-leftover:]
                else:
                    remaining_wh_pkgs[wh] = []
            else:
                # all are leftovers
                remaining_wh_pkgs[wh] = pkgs.copy()

        # Remove warehouses with empty lists from remaining_wh_pkgs for clarity
        remaining_wh_pkgs = {wh: pkgs for wh, pkgs in remaining_wh_pkgs.items() if pkgs}

        if not remaining_wh_pkgs:
            # no leftovers, just collect assignments_train
            all_assignments.extend(assignments_train)
            continue

        # Step 1: compute LB on leftovers
        total_leftover = sum(len(pkgs) for pkgs in remaining_wh_pkgs.values())
        LB = math.ceil(total_leftover / capacity)

        # Step 2: zone needs (only considering warehouses that have leftovers)
        zone_wh_map = defaultdict(list)
        for wh in warehouse_order:
            if wh in remaining_wh_pkgs:
                zone = wh_zone_map.get(wh)
                zone_wh_map[zone].append(wh)

        zone_need = {}
        for zone, whs in zone_wh_map.items():
            cnt = sum(len(remaining_wh_pkgs[wh]) for wh in whs)
            zone_need[zone] = math.ceil(cnt / capacity)
        Z_cost = sum(zone_need.values())

        # If zone-first is sufficient (Z_cost <= LB) -> allocate by zones
        # (we use <= LB because we prefer zones when equal or better)
        chosen_plan = None  # 'zones', 'clusters', or ('hybrid', chosen_clusters_set)
        chosen_clusters_set = None

        if Z_cost <= LB:
            chosen_plan = 'zones'
        else:
            # Step 3: evaluate clusters
            cluster_wh_map = defaultdict(list)
            for wh in warehouse_order:
                if wh in remaining_wh_pkgs:
                    cl = wh_cluster_map.get(wh)
                    cluster_wh_map[cl].append(wh)
            cluster_need = {}
            for cl, whs in cluster_wh_map.items():
                cnt = sum(len(remaining_wh_pkgs[wh]) for wh in whs)
                cluster_need[cl] = math.ceil(cnt / capacity)
            C_cost = sum(cluster_need.values())

            if C_cost < Z_cost:
                chosen_plan = 'clusters'
            else:
                # Try hybrid: enumerate cluster subsets and compute persons:
                clusters_keys = list(cluster_wh_map.keys())
                best_combo = None
                best_combo_cost = None
                best_combo_covered_wh = None
                # iterate powerset of clusters (excluding empty), but include empty to allow zone fallback
                for subset in _powerset(clusters_keys):
                    # skip empty subset if we want to consider pure zones separately; but allow empty to compute zone-only cost
                    chosen_clusters = set(subset)
                    # compute covered warehouses by clusters
                    covered_wh = set()
                    for cl in chosen_clusters:
                        covered_wh.update(cluster_wh_map[cl])
                    # compute cost for clusters part
                    clusters_part_cost = 0
                    for cl in chosen_clusters:
                        cnt = sum(len(remaining_wh_pkgs[wh]) for wh in cluster_wh_map[cl])
                        clusters_part_cost += math.ceil(cnt / capacity) if cnt > 0 else 0
                    # remaining warehouses -> group by zone (only those not in covered_wh)
                    remaining_wh_for_zones = [wh for wh in remaining_wh_pkgs.keys() if wh not in covered_wh]
                    zone_map_for_remaining = defaultdict(int)
                    for wh in remaining_wh_for_zones:
                        zone_map_for_remaining[wh_zone_map.get(wh)] += len(remaining_wh_pkgs[wh])
                    zones_part_cost = sum(math.ceil(cnt / capacity) for cnt in zone_map_for_remaining.values())
                    total_cost = clusters_part_cost + zones_part_cost
                    # record best (min cost). tie-breaker: prefer fewer clusters (i.e., prefer zone-heavy)
                    if best_combo_cost is None or total_cost < best_combo_cost or (total_cost == best_combo_cost and (best_combo is not None and len(chosen_clusters) < len(best_combo))):
                        best_combo_cost = total_cost
                        best_combo = chosen_clusters.copy()
                        best_combo_covered_wh = covered_wh.copy()
                # Decide: accept hybrid only if best_combo_cost < Z_cost (we want to improve over zone cost)
                if best_combo_cost is not None and best_combo_cost < Z_cost:
                    chosen_plan = 'hybrid'
                    chosen_clusters_set = best_combo
                else:
                    # fallback to zone priority
                    chosen_plan = 'zones'

        # Now perform actual allocations according to chosen_plan.
        # We'll use the _allocate_group_preserve_wh_order helper for each zone/cluster group.
        # Note: helper consumes remaining_wh_pkgs (removes assigned pkgs), so no duplication.
        if chosen_plan == 'zones':
            # allocate per zone in CSV order of warehouses (respect zone order derived from warehouse_order)
            # ensure deterministic order of zones: by first appearance in warehouse_order
            seen_zones = []
            for wh in warehouse_order:
                if wh in remaining_wh_pkgs:
                    z = wh_zone_map.get(wh)
                    if z not in seen_zones:
                        seen_zones.append(z)
            for z in seen_zones:
                whs = [wh for wh in warehouse_order if wh in remaining_wh_pkgs and wh_zone_map.get(wh) == z]
                if not whs:
                    continue
                allocs, person_counter = _allocate_group_preserve_wh_order(whs, remaining_wh_pkgs, capacity, tid, person_counter)
                assignments_train.extend(allocs)
        elif chosen_plan == 'clusters':
            # allocate per cluster in CSV order (cluster order by first appearance in warehouse_order)
            seen_clusters = []
            for wh in warehouse_order:
                if wh in remaining_wh_pkgs:
                    cl = wh_cluster_map.get(wh)
                    if cl not in seen_clusters:
                        seen_clusters.append(cl)
            for cl in seen_clusters:
                whs = [wh for wh in warehouse_order if wh in remaining_wh_pkgs and wh_cluster_map.get(wh) == cl]
                if not whs:
                    continue
                allocs, person_counter = _allocate_group_preserve_wh_order(whs, remaining_wh_pkgs, capacity, tid, person_counter)
                assignments_train.extend(allocs)
        elif chosen_plan == 'hybrid':
            # allocate clusters chosen_clusters_set first (in CSV order)
            # then allocate leftover by zones
            # cluster set ordering: order clusters by first warehouse appearance in warehouse_order
            cluster_firstpos = {}
            for cl in chosen_clusters_set:
                whs = [wh for wh in warehouse_order if wh_cluster_map.get(wh) == cl and wh in remaining_wh_pkgs]
                if whs:
                    cluster_firstpos[cl] = min(warehouse_order.index(wh) for wh in whs)
                else:
                    cluster_firstpos[cl] = float('inf')
            clusters_ordered = sorted(list(chosen_clusters_set), key=lambda c: cluster_firstpos.get(c, float('inf')))
            for cl in clusters_ordered:
                whs = [wh for wh in warehouse_order if wh in remaining_wh_pkgs and wh_cluster_map.get(wh) == cl]
                if not whs:
                    continue
                allocs, person_counter = _allocate_group_preserve_wh_order(whs, remaining_wh_pkgs, capacity, tid, person_counter)
                assignments_train.extend(allocs)
            # finally allocate remaining by zones (zone-first)
            seen_zones = []
            for wh in warehouse_order:
                if wh in remaining_wh_pkgs and remaining_wh_pkgs.get(wh):
                    z = wh_zone_map.get(wh)
                    if z not in seen_zones:
                        seen_zones.append(z)
            for z in seen_zones:
                whs = [wh for wh in warehouse_order if wh in remaining_wh_pkgs and wh_zone_map.get(wh) == z]
                if not whs:
                    continue
                allocs, person_counter = _allocate_group_preserve_wh_order(whs, remaining_wh_pkgs, capacity, tid, person_counter)
                assignments_train.extend(allocs)
        else:
            # defensive fallback: assign by zones
            seen_zones = []
            for wh in warehouse_order:
                if wh in remaining_wh_pkgs:
                    z = wh_zone_map.get(wh)
                    if z not in seen_zones:
                        seen_zones.append(z)
            for z in seen_zones:
                whs = [wh for wh in warehouse_order if wh in remaining_wh_pkgs and wh_zone_map.get(wh) == z]
                if not whs:
                    continue
                allocs, person_counter = _allocate_group_preserve_wh_order(whs, remaining_wh_pkgs, capacity, tid, person_counter)
                assignments_train.extend(allocs)

        # Ensure no leftover packages remain (should be none)
        # In rare edge cases if something left, allocate them individually
        for wh in list(remaining_wh_pkgs.keys()):
            pkgs = remaining_wh_pkgs.get(wh, [])
            while pkgs:
                person_id = f"Hc{person_counter+1}_{tid}"
                person_counter += 1
                take = pkgs[:capacity]
                pkgs = pkgs[capacity:]
                remaining_wh_pkgs[wh] = pkgs
                for pkg in take:
                    assignments_train.append({
                        'package_id': pkg,
                        'warehouse_id': wh,
                        'train_id': tid,
                        'person': person_id
                    })

        # collect assignments for this train
        all_assignments.extend(assignments_train)

    # Build outputs (same shapes as before)
    assignments_df = pd.DataFrame(all_assignments)
    if assignments_df.empty:
        summary_df = pd.DataFrame()
        per_train_detail = {}
        metadata = {'total_packages': 0, 'capacity': capacity, 'total_persons': 0}
        return assignments_df, summary_df, per_train_detail, metadata

    # summary: persons per train_id x warehouse (in person-units = ceil(packages/capacity) aggregated)
    summary_df = assignments_df.groupby(["train_id", "warehouse_id"]).size().unstack(fill_value=0)
    # reindex to original warehouse list if available
    try:
        all_warehouses = list(warehouses_df["warehouse_id"])
    except Exception:
        all_warehouses = summary_df.columns.tolist()
    summary_df = summary_df.reindex(columns=all_warehouses, fill_value=0)
    summary_df = summary_df / capacity
    summary_df = summary_df.reset_index()

    warehouse_cols = [c for c in summary_df.columns if str(c).startswith("W")]
    summary_df["Total Persons"] = summary_df[warehouse_cols].sum(axis=1).apply(math.ceil)
    summary_df[warehouse_cols] = summary_df[warehouse_cols].round(2)

    per_train_detail = {}
    for tid, grp in assignments_df.groupby('train_id'):
        detail_rows = []
        # group by warehouse and person to get packages per person per warehouse
        for (wh, person), g in grp.groupby(['warehouse_id', 'person']):
            pkgs = list(g['package_id'])
            detail_rows.append({'warehouse': wh, 'person': person, 'packages': pkgs, 'count': len(pkgs)})
        per_train_detail[tid] = pd.DataFrame(detail_rows).sort_values(['warehouse', 'person']).reset_index(drop=True)

    metadata = {
        'total_packages': len(packages),
        'capacity': capacity,
        'total_persons': int(assignments_df['person'].nunique())
    }

    return assignments_df, summary_df, per_train_detail, metadata
