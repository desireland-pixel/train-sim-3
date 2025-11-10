# simulation/human_assignment.py

import math
import numpy as np
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

def _chunk_list(lst, n):
    """Yield successive chunks of size n from lst (last chunk can be smaller)."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def assign_packages(packages_df, trains_df, warehouses_df, capacity):
    """
    New assignment algorithm implementing the decision tree:
      - Step0: warehouse direct allocation (full groups)
      - Step1: LB calculation if leftovers remain
      - Step2: zone calculation
      - Step3: if Z_cost == LB -> assign by zones
      - Step4: otherwise evaluate cluster combos and choose minimal collectors
      - Step5: produce final assignments in order: Step0 allocations, then zone or cluster allocations

    Inputs:
      packages_df: DataFrame with columns package_id, warehouse_id, generated_time (train inferred)
      trains_df: DataFrame with trains info (used only to infer train_id)
      warehouses_df: DataFrame with warehouse_id, zone, cluster (zone/cluster names assumed available)
      capacity: int, max packages a person can carry

    Returns:
      assignments_df, summary_df, per_train_detail, metadata
    """

    packages = packages_df.copy().reset_index(drop=True)
    trains = trains_df.copy().reset_index(drop=True)
    warehouses = warehouses_df.copy().reset_index(drop=True)

    # Normalize types
    packages['package_id'] = packages['package_id'].astype(str)
    packages['warehouse_id'] = packages['warehouse_id'].astype(str)
    packages['train_id'] = packages['package_id'].apply(lambda pid: infer_train_id_from_pkg(pid, trains))

    all_assignments = []
    train_groups = packages.groupby('train_id')

    for tid, grp in train_groups:
        # All packages to be assigned for this train
        pkg_list_all = list(grp.to_dict('records'))
        total_packages_train = len(pkg_list_all)
        if total_packages_train == 0:
            continue

        # Total persons required (upper bound based on capacity)
        Hn_train = math.ceil(total_packages_train / capacity)
        # pre-create person names
        persons_train = [f"Hc{i+1}_{tid}" for i in range(Hn_train)]

        # Organize packages by warehouse
        wh_to_pkgs = defaultdict(list)
        for row in pkg_list_all:
            wh_to_pkgs[row['warehouse_id']].append(row)

        # STEP 0: Warehouse direct allocations (full capacity groups)
        assignments_train = []
        leftovers_per_warehouse = {}  # warehouse_id -> list of leftover package rows
        step0_persons = []  # list of person names created in step0
        person_counter = 0

        for wh, pkgs in wh_to_pkgs.items():
            n = len(pkgs)
            full_groups = n // capacity
            idx = 0
            for _ in range(full_groups):
                if person_counter >= len(persons_train):
                    raise RuntimeError("Insufficient person slots computed — logic error.")
                person = persons_train[person_counter]
                step0_persons.append(person)
                group_pkgs = pkgs[idx: idx + capacity]
                for p in group_pkgs:
                    assignments_train.append({
                        'package_id': p['package_id'],
                        'warehouse_id': wh,
                        'train_id': tid,
                        'person': person
                    })
                idx += capacity
                person_counter += 1
            # leftover packages for further allocation
            rem = pkgs[idx:]
            if rem:
                leftovers_per_warehouse[wh] = rem

        # If no leftovers, we are done for this train
        if not leftovers_per_warehouse:
            all_assignments.extend(assignments_train)
            continue

        # STEP 1: Lower bound LB
        total_leftovers = sum(len(v) for v in leftovers_per_warehouse.values())
        LB = math.ceil(total_leftovers / capacity)

        # Prepare zone mapping and cluster mapping from warehouses_df
        # Normalize zone and cluster column names (accept either 'zone'/'cluster' or 'Zone'/'Cluster')
        wdf = warehouses.copy()
        # ensure string columns exist
        if 'zone' not in wdf.columns and 'Zone' in wdf.columns:
            wdf = wdf.rename(columns={'Zone': 'zone'})
        if 'cluster' not in wdf.columns and 'Cluster' in wdf.columns:
            wdf = wdf.rename(columns={'Cluster': 'cluster'})

        zone_of = dict(zip(wdf['warehouse_id'].astype(str), wdf['zone'].astype(str)))
        cluster_of = dict(zip(wdf['warehouse_id'].astype(str), wdf['cluster'].astype(str)))

        # STEP 2: Zone totals & needs
        zone_totals = defaultdict(int)
        for wh, pkgs in leftovers_per_warehouse.items():
            z = zone_of.get(wh, None)
            zone_totals[z] += len(pkgs)

        zone_need = {z: math.ceil(cnt / capacity) for z, cnt in zone_totals.items()}
        Z_cost = sum(zone_need.values())

        # If Z_cost equals LB -> allocate by zones (STEP 3)
        use_clusters = False
        chosen_cluster_combination = None
        chosen_zone_need = zone_need.copy()
        if Z_cost != LB:
            # STEP 4: Cluster evaluation
            # compute cluster totals & needs based on leftovers
            cluster_members = defaultdict(list)  # cluster -> list of warehouses
            for wh in leftovers_per_warehouse.keys():
                c = cluster_of.get(wh, None)
                cluster_members[c].append(wh)

            cluster_totals = {}
            for c, whs in cluster_members.items():
                cnt = sum(len(leftovers_per_warehouse[w]) for w in whs)
                cluster_totals[c] = cnt

            cluster_need = {c: math.ceil(cnt / capacity) for c, cnt in cluster_totals.items()}

            # Evaluate combinations of clusters (only clusters that appear in leftovers)
            candidate_clusters = list(cluster_totals.keys())
            best_option = None
            best_cost = None
            best_cover_whs = None  # warehouses covered by chosen clusters

            # consider also the empty combination (no clusters chosen -> fallback to zone cost)
            best_cost = Z_cost
            best_option = ('zones', None)
            best_cover_whs = set()

            # Try all non-empty combinations
            for r in range(1, len(candidate_clusters) + 1):
                for comb in combinations(candidate_clusters, r):
                    # clusters in comb do not overlap by definition (clusters are labels),
                    # but they might share warehouses in data - treat union of their warehouses as covered
                    covered_whs = set()
                    comb_cost = 0
                    for c in comb:
                        # If cluster has total 0 (shouldn't), skip
                        comb_cost += cluster_need.get(c, 0)
                        # collect warehouses in this cluster (but only those that have leftovers)
                        for wh in cluster_members.get(c, []):
                            covered_whs.add(wh)

                    # for warehouses not covered by the clusters, we must allocate by zone for their zones
                    remaining_whs = set(leftovers_per_warehouse.keys()) - covered_whs
                    remaining_zone_totals = defaultdict(int)
                    for wh in remaining_whs:
                        z = zone_of.get(wh, None)
                        remaining_zone_totals[z] += len(leftovers_per_warehouse[wh])
                    remaining_zone_need = sum(math.ceil(cnt / capacity) for cnt in remaining_zone_totals.values())

                    total_cost = comb_cost + remaining_zone_need

                    if best_cost is None or total_cost < best_cost:
                        best_cost = total_cost
                        best_option = ('clusters', comb)
                        best_cover_whs = covered_whs

            # If cluster option found strictly smaller than Z_cost, use clusters
            if best_option[0] == 'clusters' and best_cost < Z_cost:
                use_clusters = True
                chosen_cluster_combination = best_option[1]
                # build final zone need for remaining warehouses (zones not covered)
                remaining_whs = set(leftovers_per_warehouse.keys()) - best_cover_whs
                final_zone_totals = defaultdict(int)
                for wh in remaining_whs:
                    z = zone_of.get(wh, None)
                    final_zone_totals[z] += len(leftovers_per_warehouse[wh])
                chosen_zone_need = {z: math.ceil(cnt / capacity) for z, cnt in final_zone_totals.items()}
            else:
                use_clusters = False
                chosen_cluster_combination = None
                chosen_zone_need = zone_need.copy()
        else:
            # Z_cost == LB -> use zones
            use_clusters = False
            chosen_cluster_combination = None
            chosen_zone_need = zone_need.copy()

        # Now we know the allocation breakdown:
        # - step0_persons already assigned (person_counter positions used)
        # - we must assign additional persons for either zones or clusters to cover leftovers
        # Create list of remaining person slots
        remaining_persons = persons_train[person_counter:]
        rem_person_idx = 0

        # Helper to allocate a list of packages among N persons (names), splitting into capacity chunks
        def _allocate_pkgs_to_persons(pkgs, person_names):
            """Return list of (person_name, [pkg_rows]) assignments. pkgs is list of pkg rows."""
            chunks = list(_chunk_list(pkgs, capacity))
            # If more chunks than person_names, we must reuse person_names sequentially (shouldn't happen because person count matches)
            assignments = []
            # If too few persons, we still assign using the available persons round-robin (defensive)
            if not person_names:
                raise RuntimeError("No person available to allocate packages.")
            for i, ch in enumerate(chunks):
                person = person_names[i % len(person_names)]
                assignments.append((person, ch))
            return assignments

        # STEP 5a: First, add Step0 assignments already recorded (assignments_train)
        # (already present in assignments_train)
        # STEP 5b: Assign leftovers according to chosen mode (zones or clusters)
        # We'll build entity -> list of package rows for allocation
        entity_to_pkgs = {}  # entity_key -> list of pkgs ; entity_key is either "ZONE:<zone>" or "CLUSTER:<cluster>" or "WAREHOUSE:<wh>"
        entity_person_counts = {}  # entity_key -> number of persons to allocate

        # Add per-warehouse leftovers as base
        for wh, pkgs in leftovers_per_warehouse.items():
            # Will be grouped into zones/clusters below
            pass

        # Determine how many persons per zone (chosen_zone_need) and per cluster (cluster_need when used)
        # Build mapping from entity to included warehouses (for packaging)
        if use_clusters and chosen_cluster_combination:
            # clusters chosen
            # clusters cover some warehouses (only those that had leftovers)
            covered_by_clusters = set()
            for c in chosen_cluster_combination:
                whs = [w for w in leftovers_per_warehouse.keys() if cluster_of.get(w) == c]
                covered_by_clusters.update(whs)
                # gather pkgs
                pkgs_combined = []
                for w in whs:
                    pkgs_combined.extend(leftovers_per_warehouse.get(w, []))
                entity_key = f"CLUSTER:{c}"
                entity_to_pkgs[entity_key] = pkgs_combined
                entity_person_counts[entity_key] = math.ceil(len(pkgs_combined) / capacity) if pkgs_combined else 0

            # remaining warehouses -> distribute by zone (chosen_zone_need)
            remaining_whs = set(leftovers_per_warehouse.keys()) - covered_by_clusters
            zone_to_whs = defaultdict(list)
            for w in remaining_whs:
                z = zone_of.get(w, None)
                zone_to_whs[z].append(w)
            for z, whs in zone_to_whs.items():
                pkgs_combined = []
                for w in whs:
                    pkgs_combined.extend(leftovers_per_warehouse.get(w, []))
                entity_key = f"ZONE:{z}"
                entity_to_pkgs[entity_key] = pkgs_combined
                entity_person_counts[entity_key] = math.ceil(len(pkgs_combined) / capacity) if pkgs_combined else 0
        else:
            # Use zones (possibly all zones)
            # Build zone -> warehouses with leftovers
            zone_to_whs = defaultdict(list)
            for w in leftovers_per_warehouse.keys():
                z = zone_of.get(w, None)
                zone_to_whs[z].append(w)
            for z, whs in zone_to_whs.items():
                pkgs_combined = []
                for w in whs:
                    pkgs_combined.extend(leftovers_per_warehouse.get(w, []))
                entity_key = f"ZONE:{z}"
                entity_to_pkgs[entity_key] = pkgs_combined
                entity_person_counts[entity_key] = math.ceil(len(pkgs_combined) / capacity) if pkgs_combined else 0

        # Now allocate persons to entities (respecting persons_train list length)
        entity_to_persons = {}
        for entity, pkgs in entity_to_pkgs.items():
            num_persons_needed = entity_person_counts.get(entity, 0)
            # take next num_persons_needed from remaining_persons
            persons_assigned = []
            for _ in range(num_persons_needed):
                if rem_person_idx >= len(remaining_persons):
                    # defensive: if out of generated persons, create additional names (shouldn't normally happen)
                    extra_idx = len(persons_train) + 1
                    new_person = f"Hc{extra_idx}_{tid}"
                    remaining_persons.append(new_person)
                persons_assigned.append(remaining_persons[rem_person_idx])
                rem_person_idx += 1
            entity_to_persons[entity] = persons_assigned

        # Now actually split packages within each entity among its persons and produce assignments
        for entity, pkgs in entity_to_pkgs.items():
            persons_for_entity = entity_to_persons.get(entity, [])
            # pkgs is list of package rows; ensure deterministic order (sort by package_id)
            pkgs_sorted = sorted(pkgs, key=lambda p: p['package_id'])
            allocs = _allocate_pkgs_to_persons(pkgs_sorted, persons_for_entity)
            for person, pkg_rows in allocs:
                for p in pkg_rows:
                    assignments_train.append({
                        'package_id': p['package_id'],
                        'warehouse_id': p['warehouse_id'],
                        'train_id': tid,
                        'person': person
                    })

        # Append assignments for this train
        all_assignments.extend(assignments_train)

    # End per-train loop

    assignments_df = pd.DataFrame(all_assignments)
    if assignments_df.empty:
        summary_df = pd.DataFrame()
        per_train_detail = {}
        metadata = {'total_packages': 0, 'capacity': capacity, 'total_persons': 0}
        return assignments_df, summary_df, per_train_detail, metadata

    # Build summary_df (same style as previous code)
    summary_df = assignments_df.groupby(["train_id", "warehouse_id"]).size().unstack(fill_value=0)
    all_warehouses = list(warehouses_df["warehouse_id"])
    summary_df = summary_df.reindex(columns=all_warehouses, fill_value=0)
    summary_df = summary_df / capacity
    summary_df = summary_df.reset_index()

    warehouse_cols = [c for c in summary_df.columns if str(c).startswith("W")]
    summary_df["Total Persons"] = np.ceil(summary_df[warehouse_cols].sum(axis=1)).astype(int)
    summary_df[warehouse_cols] = summary_df[warehouse_cols].round(2)

    # Build per_train_detail
    per_train_detail = {}
    for tid, grp in assignments_df.groupby('train_id'):
        detail_rows = []
        # group by warehouse and person (stable deterministic ordering)
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
