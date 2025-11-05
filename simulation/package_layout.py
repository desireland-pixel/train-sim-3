# simulation/package_layout.py
# places packages near warehouses, same logic as before
def compute_package_positions(packages_df, warehouses_df, current_time, offset_x=12, col_spacing=12, row_spacing=25, max_cols=5):
    positions = []
    if packages_df is None or packages_df.empty:
        return positions
    for wh_id, group in packages_df.groupby("warehouse_id"):
        wh = warehouses_df[warehouses_df.warehouse_id == wh_id].iloc[0]
        for idx, (_, pkg) in enumerate(group.iterrows()):
            if current_time >= pkg.generated_time:
                col = idx % max_cols
                row = idx // max_cols
                x = wh.x + offset_x + col * col_spacing
                y = wh.y + row * row_spacing
                positions.append((pkg.package_id, wh_id, x, y))
    return positions
