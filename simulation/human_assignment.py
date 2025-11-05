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

    for tid, grp in train_groups:
        total_packages_train = len(grp)
        if total_packages_train == 0:
            continue

        Hn_train = math.ceil(total_packages_train / capacity)
        persons_train = [f"Hc{i+1}_{tid}" for i in range(Hn_train)]

        assignments_train = []
        person_idx = 0
        leftovers_per_warehouse = {}

        wh_to_pkgs_train = defaultdict(list)
        for _, row in grp.iterrows():
            wh_to_pkgs_train[row['warehouse_id']].append({'package_id': row['package_id'], 'train_id': row['train_id']})

        for wh, pkg_list in wh_to_pkgs_train.items():
            idx = 0
            n = len(pkg_list)
            f_i = n // capacity
            for f in range(f_i):
                person = persons_train[person_idx]
                for k in range(capacity):
                    pkg = pkg_list[idx]
                    assignments_train.append({
                        'package_id': pkg['package_id'],
                        'warehouse_id': wh,
                        'train_id': pkg['train_id'],
                        'person': person
                    })
                    idx += 1
                person_idx += 1
            leftover_pkgs = pkg_list[idx:]
            if leftover_pkgs:
                leftovers_per_warehouse[wh] = leftover_pkgs

        if leftovers_per_warehouse:
            bins = [{'person': persons_train[i], 'used': 0, 'allocs': []} for i in range(person_idx, len(persons_train))]

            leftover_items = [(wh, len(pkgs), pkgs) for wh, pkgs in leftovers_per_warehouse.items()]
            leftover_items.sort(key=lambda x: x[1], reverse=True)

            for wh, count, pkgs in leftover_items:
                best_bin = None
                best_after = None
                for b in bins:
                    if b['used'] + count <= capacity:
                        after = b['used'] + count
                        if best_after is None or after < best_after:
                            best_after = after
                            best_bin = b
                if best_bin:
                    best_bin['used'] += count
                    best_bin['allocs'].append((wh, pkgs))
                else:
                    remaining = pkgs.copy()
                    for b in bins:
                        avail = capacity - b['used']
                        if avail <= 0:
                            continue
                        take = min(avail, len(remaining))
                        take_pkgs = remaining[:take]
                        b['allocs'].append((wh, take_pkgs))
                        b['used'] += take
                        remaining = remaining[take:]
                        if not remaining:
                            break
                    if remaining:
                        raise RuntimeError("Unable to pack leftovers into persons — inconsistent Hn.")

            for b in bins:
                person = b['person']
                for wh, alloc_pkgs in b['allocs']:
                    for pkg in alloc_pkgs:
                        assignments_train.append({
                            'package_id': pkg['package_id'],
                            'warehouse_id': wh,
                            'train_id': pkg['train_id'],
                            'person': person
                        })

        all_assignments.extend(assignments_train)

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
        'total_persons': sum([math.ceil(len(grp)/capacity) for tid, grp in train_groups])
    }

    return assignments_df, summary_df, per_train_detail, metadata
