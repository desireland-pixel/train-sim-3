# simulation/human_assignment.py

import math
import itertools
import pandas as pd
from collections import defaultdict, OrderedDict

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


def assign_packages(packages_df, trains_df, warehouses_df, capacity):
    """
    Returns: assignments_df, summary_df, per_train_detail, metadata
    assignments_df columns: package_id, warehouse_id, train_id, person
    """
    # defensive copies
    packages = packages_df.copy().reset_index(drop=True)
    trains = trains_df.copy().reset_index(drop=True)
    warehouses = warehouses_df.copy().reset_index(drop=True)

    # ensure cols are strings
    packages['package_id'] = packages['package_id'].astype(str)
    packages['warehouse_id'] = packages['warehouse_id'].astype(str)
    packages['train_id'] = packages['package_id'].apply(lambda pid: infer_train_id_from_pkg(pid, trains))

    # warehouse order from CSV
    warehouse_order = list(warehouses['warehouse_id'])

    # maps for zone/cluster lookup
    zone_map = warehouses.set_index('warehouse_id')['zone'].to_dict()
    cluster_map = warehouses.set_index('warehouse_id')['cluster'].to_dict()

    all_assignments = []
    train_groups = packages.groupby('train_id')

    for tid, grp in train_groups:
        if grp.empty:
            continue

        # build per-warehouse package lists preserving CSV order
        wh_to_pkgs = OrderedDict((wh, []) for wh in warehouse_order)
        for _, row in grp.iterrows():
            wh = row['warehouse_id']
            if wh not in wh_to_pkgs:
                wh_to_pkgs[wh] = []
            wh_to_pkgs[wh].append(row['package_id'])

        assignments_train = []
        person_count = 0  # used for Hc numbering per train

        # -----------------------
        # STEP 0: Workshop direct allocations (warehouse >= capacity)
        # Iterate in CSV warehouse order
        # -----------------------
        leftovers = OrderedDict()
        for wh in warehouse_order:
            pkgs = wh_to_pkgs.get(wh, [])
            if not pkgs:
                continue
            n = len(pkgs)
            if n >= capacity:
                full = n // capacity
                for _ in range(full):
                    person_count += 1
                    person = f"Hc{person_count}_{tid}"
                    take = pkgs[:capacity]
                    pkgs = pkgs[capacity:]
                    for p in take:
                        assignments_train.append({
                            'package_id': p,
                            'warehouse_id': wh,
                            'train_id': tid,
                            'person': person
                        })
                # remaining (less than capacity) stored as leftover
            if pkgs:
                leftovers[wh] = pkgs

        # if no leftovers -> done for this train
        if not leftovers:
            all_assignments.extend(assignments_train)
            continue

        # -----------------------
        # Compute LB (minimum persons needed)
        # -----------------------
        total_leftover = sum(len(v) for v in leftovers.values())
        LB = math.ceil(total_leftover / capacity)

        # -----------------------
        # Build zone -> warehouses (in CSV order)
        # -----------------------
        zone_to_wh = OrderedDict()
        for wh in warehouse_order:
            if wh in leftovers:
                z = zone_map.get(wh)
                zone_to_wh.setdefault(z, []).append(wh)

        # compute zone needs
        zone_need = {}
        for z, whs in zone_to_wh.items():
            cnt = sum(len(leftovers[wh]) for wh in whs)
            zone_need[z] = _ceil_div(cnt, capacity)
        Z_cost = sum(zone_need.values())

        # -----------------------
        # If zone-only meets LB or is <= LB -> assign zones (zone-first)
        # Zone assignment uses warehouse-first packing (see explanation)
        # -----------------------
        def assign_zone_group(zone_whs):
            """Assign packages inside a zone using warehouse-first packing.
               Returns list of assignment dicts and updates leftovers by removing assigned pkgs."""
            nonlocal person_count
            results = []
            # We'll iterate warehouses in CSV order (zone_whs already in CSV order)
            # For warehouse-first packing:
            # start a current person load (list) and current load count
            curr_person_pkgs = []
            curr_person_whs = []
            curr_load = 0
            for wh in zone_whs:
                wh_pkgs = leftovers.get(wh, [])
                if not wh_pkgs:
                    continue
                wh_count = len(wh_pkgs)
                # If warehouse fits entirely into current person -> assign whole warehouse to current person
                if curr_load + wh_count <= capacity and curr_load > 0:
                    # add whole warehouse to current person
                    curr_person_pkgs.extend([(p, wh) for p in wh_pkgs])
                    curr_load += wh_count
                    leftovers[wh] = []  # removed
                elif wh_count <= capacity and curr_load == 0:
                    # start a fresh person and assign this whole warehouse to them (warehouse-first)
                    person_count += 1
                    person = f"Hc{person_count}_{tid}"
                    for p in wh_pkgs:
                        results.append({'package_id': p, 'warehouse_id': wh, 'train_id': tid, 'person': person})
                    leftovers[wh] = []
                else:
                    # The warehouse cannot be fully merged into current person without exceeding capacity,
                    # and either current person is empty or wh_count < capacity but would exceed.
                    # If current person has some packages, finalize current person first.
                    if curr_load > 0:
                        person_count += 1
                        person = f"Hc{person_count}_{tid}"
                        for p, pwh in curr_person_pkgs:
                            results.append({'package_id': p, 'warehouse_id': pwh, 'train_id': tid, 'person': person})
                        curr_person_pkgs = []
                        curr_load = 0
                    # Now assign warehouse wholly to a new person if it fits; else (wh_count > capacity) split (shouldn't happen because step0 handled >=capacity)
                    if len(wh_pkgs) <= capacity:
                        person_count += 1
                        person = f"Hc{person_count}_{tid}"
                        for p in wh_pkgs:
                            results.append({'package_id': p, 'warehouse_id': wh, 'train_id': tid, 'person': person})
                        leftovers[wh] = []
                    else:
                        # Split warehouse across multiple persons (only if wh_count > capacity - unexpected in normal flow)
                        idx = 0
                        while idx < len(wh_pkgs):
                            person_count += 1
                            person = f"Hc{person_count}_{tid}"
                            take = wh_pkgs[idx: idx + capacity]
                            for p in take:
                                results.append({'package_id': p, 'warehouse_id': wh, 'train_id': tid, 'person': person})
                            idx += capacity
                        leftovers[wh] = []

            # finalize any remaining curr_person_pkgs
            if curr_person_pkgs:
                person_count += 1
                person = f"Hc{person_count}_{tid}"
                for p, pwh in curr_person_pkgs:
                    results.append({'package_id': p, 'warehouse_id': pwh, 'train_id': tid, 'person': person})
                curr_person_pkgs = []
            return results

        # -----------------------
        # Helper: assign cluster group similar to zone (warehouse-first, but group by cluster)
        # -----------------------
        def assign_cluster_group(cluster_whs):
            nonlocal person_count
            results = []
            # same logic as zone but cluster_whs must be in CSV order
            curr_person_pkgs = []
            curr_load = 0
            for wh in cluster_whs:
                wh_pkgs = leftovers.get(wh, [])
                if not wh_pkgs:
                    continue
                wh_count = len(wh_pkgs)
                if curr_load + wh_count <= capacity and curr_load > 0:
                    curr_person_pkgs.extend([(p, wh) for p in wh_pkgs])
                    curr_load += wh_count
                    leftovers[wh] = []
                elif wh_count <= capacity and curr_load == 0:
                    person_count += 1
                    person = f"Hc{person_count}_{tid}"
                    for p in wh_pkgs:
                        results.append({'package_id': p, 'warehouse_id': wh, 'train_id': tid, 'person': person})
                    leftovers[wh] = []
                else:
                    if curr_load > 0:
                        person_count += 1
                        person = f"Hc{person_count}_{tid}"
                        for p, pwh in curr_person_pkgs:
                            results.append({'package_id': p, 'warehouse_id': pwh, 'train_id': tid, 'person': person})
                        curr_person_pkgs = []
                        curr_load = 0
                    if wh_count <= capacity:
                        person_count += 1
                        person = f"Hc{person_count}_{tid}"
                        for p in wh_pkgs:
                            results.append({'package_id': p, 'warehouse_id': wh, 'train_id': tid, 'person': person})
                        leftovers[wh] = []
                    else:
                        idx = 0
                        while idx < len(wh_pkgs):
                            person_count += 1
                            person = f"Hc{person_count}_{tid}"
                            take = wh_pkgs[idx: idx + capacity]
                            for p in take:
                                results.append({'package_id': p, 'warehouse_id': wh, 'train_id': tid, 'person': person})
                            idx += capacity
                        leftovers[wh] = []
            if curr_person_pkgs:
                person_count += 1
                person = f"Hc{person_count}_{tid}"
                for p, pwh in curr_person_pkgs:
                    results.append({'package_id': p, 'warehouse_id': pwh, 'train_id': tid, 'person': person})
            return results

        # -----------------------
        # Decide allocation strategy:
        # - If zone cost <= LB -> zone-first allocation
        # - Else compute cluster cost -> if cluster_cost < zone_cost -> cluster-only
        # - Else try zone+cluster combinations (subsets of clusters) to see if any combo reduces cost below Z_cost.
        #   If a combination reduces persons (prefer lower total persons), use it.
        # - Otherwise fallback to zone-priority (zone allocation)
        # -----------------------

        if Z_cost <= LB:
            # assign by zones
            for z, whs in zone_to_wh.items():
                # assign warehouses in zone using warehouse-first packing
                assigned = assign_zone_group(whs)
                assignments_train.extend(assigned)
            all_assignments.extend(assignments_train)
            continue

        # build cluster -> wh (CSV order)
        cluster_to_wh = OrderedDict()
        for wh in warehouse_order:
            if wh in leftovers:
                cl = cluster_map.get(wh)
                cluster_to_wh.setdefault(cl, []).append(wh)

        cluster_need = {}
        for cl, whs in cluster_to_wh.items():
            cnt = sum(len(leftovers[wh]) for wh in whs)
            cluster_need[cl] = _ceil_div(cnt, capacity)
        C_cost = sum(cluster_need.values())

        # If clusters are strictly better than zones, use clusters
        if C_cost < Z_cost:
            # assign clusters in CSV order of warehouses
            for cl, whs in cluster_to_wh.items():
                assigned = assign_cluster_group(whs)
                assignments_train.extend(assigned)
            all_assignments.extend(assignments_train)
            continue

        # else C_cost >= Z_cost -> try hybrid combinations of clusters subsets + zones
        # enumerate all subsets of clusters (there will usually be very few)
        best_plan = None
        best_cost = Z_cost  # baseline = zone cost
        clusters_list = list(cluster_to_wh.keys())

        # produce set of warehouses covered by a cluster subset
        for r in range(1, len(clusters_list) + 1):
            for subset in itertools.combinations(clusters_list, r):
                covered_wh = set()
                for cl in subset:
                    covered_wh.update(cluster_to_wh.get(cl, []))
                # remaining warehouses (that still need handling) will be assigned by zones
                remaining_wh = [wh for wh in leftovers.keys() if wh not in covered_wh]
                # compute persons needed for selected clusters
                cluster_persons = 0
                for cl in subset:
                    cnt = sum(len(leftovers[wh]) for wh in cluster_to_wh.get(cl, []))
                    cluster_persons += _ceil_div(cnt, capacity)
                # compute persons for remaining by zones
                # build map zone->list(remaining wh)
                rem_zone_map = defaultdict(list)
                for wh in remaining_wh:
                    rem_zone_map[zone_map.get(wh)].append(wh)
                zone_persons = 0
                for z, whs in rem_zone_map.items():
                    cnt = sum(len(leftovers[wh]) for wh in whs)
                    zone_persons += _ceil_div(cnt, capacity)
                total_persons = cluster_persons + zone_persons
                # choose plan if total_persons < best_cost (strict improvement)
                if total_persons < best_cost:
                    best_cost = total_persons
                    best_plan = {
                        'clusters': set(subset),
                        'remaining_zones': rem_zone_map,
                        'covered_wh': covered_wh,
                        'cluster_persons': cluster_persons,
                        'zone_persons': zone_persons,
                        'total_persons': total_persons
                    }

        # If we found a hybrid that reduces persons compared to zones, use it.
        if best_plan is not None:
            # assign selected clusters first (in CSV order of warehouses inside each cluster)
            for cl in clusters_list:
                if cl not in best_plan['clusters']:
                    continue
                whs = cluster_to_wh.get(cl, [])
                assigned = assign_cluster_group(whs)
                assignments_train.extend(assigned)
            # assign remaining warehouses by zones (for each zone in CSV-derived order)
            # Build zone order from warehouse_order
            zone_order = []
            for wh in warehouse_order:
                if wh in leftovers:
                    z = zone_map.get(wh)
                    if z not in zone_order:
                        zone_order.append(z)
            for z in zone_order:
                # only assign if zone has remaining warehouses not covered by clusters
                whs = [wh for wh in zone_to_wh.get(z, []) if wh in leftovers]
                if not whs:
                    continue
                assigned = assign_zone_group(whs)
                assignments_train.extend(assigned)
            all_assignments.extend(assignments_train)
            continue

        # else fallback to zone-priority (even if zone_cost > LB)
        for z, whs in zone_to_wh.items():
            assigned = assign_zone_group(whs)
            assignments_train.extend(assigned)

        all_assignments.extend(assignments_train)

    # -----------------------
    # Build outputs
    # -----------------------
    assignments_df = pd.DataFrame(all_assignments)
    if assignments_df.empty:
        summary_df = pd.DataFrame()
        per_train_detail = {}
        metadata = {'total_packages': 0, 'capacity': capacity, 'total_persons': 0}
        return assignments_df, summary_df, per_train_detail, metadata

    # Summary (train x warehouse) in persons (we follow your previous UI: persons = ceil(count/capacity))
    summary_df = assignments_df.groupby(["train_id", "warehouse_id"]).size().unstack(fill_value=0)
    all_wh_list = list(warehouses_df["warehouse_id"])
    summary_df = summary_df.reindex(columns=all_wh_list, fill_value=0)
    # convert counts -> persons needed per warehouse fraction (matching existing UI)
    summary_df = summary_df / capacity
    summary_df = summary_df.reset_index()

    warehouse_cols = [c for c in summary_df.columns if str(c).startswith("W")]
    summary_df["Total Persons"] = summary_df[warehouse_cols].sum(axis=1).apply(math.ceil).astype(int)
    summary_df[warehouse_cols] = summary_df[warehouse_cols].round(2)

    # per_train_detail (same shape as before)
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
