# app.py
import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go
from pathlib import Path
from modules.assignment import assign_packages
from modules.movement import load_points, build_route, interpolate_position

st.set_page_config(layout="wide", page_title="Train-Warehouse Simulation")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# -------------------------
# Load data (CSVs)
# -------------------------
def load_csv(filename):
    path = DATA_DIR / filename
    if path.exists():
        try:
            df = pd.read_csv(path)
            if df.empty or len(df.columns) == 0:
                return pd.DataFrame()
            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

trains = load_csv("trains.csv")
warehouses = load_csv("warehouses.csv")
packages = load_csv("packages.csv")
persons = load_csv("persons.csv")

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Simulation Settings")
max_packages_per_person = st.sidebar.number_input("Max packages a person can carry", 1, 10, 5)
current_time = st.sidebar.number_input("Current time (minutes)", 0, 60, 0)

# -------------------------
# Orders per train inputs
# -------------------------
st.sidebar.markdown("### Orders per Train")

# Initialize a dictionary in session state to hold dynamic input values
if 'dynamic_orders' not in st.session_state:
    st.session_state.dynamic_orders = {}

# This list will store the *values* entered by the user in the correct order
train_orders = [] 
train_ids = trains['train_id'].tolist() # Get the list of train IDs once

# Iterate through the actual train IDs from your DataFrame
for train_id in train_ids:
    # Create a unique key for the widget in session state
    input_key = f"orders_for_{train_id}"
    
    # Set a default value in session state if it doesn't exist
    if input_key not in st.session_state.dynamic_orders:
        st.session_state.dynamic_orders[input_key] = 0

    # Create the number input dynamically in the sidebar
    # The 'key' argument ensures Streamlit manages the value across reruns
    current_value = st.sidebar.number_input(
        f"{train_id} Orders",
        min_value=0,
        max_value=20,
        value=st.session_state.dynamic_orders[input_key],
        key=input_key # Link the widget to the session state key
    )
    
    # Append the value to the list used by the generation logic below
    train_orders.append(current_value)

# -------------------------
# Generate Packages Button
# -------------------------
if st.sidebar.button("Generate Packages"):
    gen_packages = []

    # For each train, use the manual order inputs from sidebar
    for i, train_id in enumerate(trains.train_id, 1):
        n_orders = train_orders[i - 1]  # take order count directly from sidebar
        if n_orders > 0:
            start_time = int(trains.loc[trains.train_id == train_id, "start_time"].values[0])
            for j in range(1, n_orders + 1):
                pkg_id = f"{i:02d}{j:02d}"  # 0101, 0102 ... etc.
                warehouse_id = np.random.choice(warehouses.warehouse_id)  # random W1–W6
                gen_packages.append({
                    "package_id": pkg_id,
                    "warehouse_id": warehouse_id,
                    "generated_time": start_time - 10
                })

    # Convert only if we actually generated packages
    if gen_packages:
        packages = pd.DataFrame(gen_packages)
        st.session_state["packages"] = packages
        packages.to_csv(DATA_DIR / "packages.csv", index=False)
        st.session_state["pkg_text"] = packages[["package_id", "warehouse_id", "generated_time"]]
    else:
        st.warning("No orders entered — no packages generated.")
        st.session_state.pop("packages", None)
        st.session_state.pop("pkg_text", None)

# -------------------------
# Page title
# -------------------------
st.title("🚉 Train–Warehouse Simulation")
st.markdown(f"**Simulation Time: {current_time} min**")

# -------------------------
# Simulation visuals
# -------------------------
fig = go.Figure()

# Warehouses
fig.add_trace(go.Scatter(
    x=warehouses.x, y=warehouses.y,
    mode="markers+text",
    text=warehouses.warehouse_id,
    name="Warehouses",
    marker=dict(size=15, color="green", symbol="square"),
    textposition="top center",
    textfont=dict(color="black")
))

# Platforms (5 fixed)
platforms = pd.DataFrame({
    'platform': [1,2,3,4,5],
    'x': [200,200,200,200,200],
    'y': [150,100,50,0,-50]
})
fig.add_trace(go.Scatter(
    x=platforms.x, y=platforms.y,
    mode="markers+text",
    text=[f"P{i}" for i in platforms.platform],
    name="Platforms",
    marker=dict(size=18, color="blue")
))

# Trains movement
train_positions = []
for _, r in trains.iterrows():
    if current_time < r.start_time:
        x, y = r.x_source, r.y_source
    elif current_time > r.arrive_time:
        x, y = r.x_platform, r.y_platform
    else:
        frac = (current_time - r.start_time) / (r.arrive_time - r.start_time)
        x = r.x_source + frac * (r.x_platform - r.x_source)
        y = r.y_source + frac * (r.y_platform - r.y_source)
    train_positions.append((r.train_id, x, y))

fig.add_trace(go.Scatter(
    x=[x for _,x,_ in train_positions],
    y=[y for _,_,y in train_positions],
    text=[tid for tid,_,_ in train_positions],
    mode="markers+text",
    name="Trains",
    marker=dict(size=20, color="red"),
    textfont=dict(color="black"),
    textposition="middle left"
))

# -------------------------
# Packages display
# -------------------------
base_hour = 9
base_minute = 0

if "packages" in st.session_state:
    packages = st.session_state["packages"]
    
# Group packages by warehouse to arrange them neatly
offset_x = 12
col_spacing = 12
row_spacing = 25
max_cols = 5

for wh_id, group in packages.groupby("warehouse_id"):
    wh = warehouses[warehouses.warehouse_id == wh_id].iloc[0]

    for idx, (_, pkg) in enumerate(group.iterrows()):
        if current_time >= pkg.generated_time:
            col = idx % max_cols
            row = idx // max_cols

            # position right of warehouse
            x = wh.x + offset_x + col * col_spacing
            y = wh.y + row * row_spacing

            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                text=[pkg.package_id],
                textposition="bottom center",
                marker=dict(size=8, color="#D2B48C", symbol="square",
                            line=dict(color="black", width=0.25)),
                name="Packages",
                showlegend=(len(fig.data) == 3)
            ))

# -------------------------
# Worker movement
# -------------------------
points = load_points(DATA_DIR)
walk_speed = 50  # units per minute (tunable)

if "per_train_detail" in st.session_state:
    per_train_detail = st.session_state["per_train_detail"]

    for train_id, detail in per_train_detail.items():
        if detail.empty:
            continue
        platform = int(trains.loc[trains.train_id == train_id, "platform"].values[0])

        for _, row in detail.iterrows():
            warehouse_id = row["warehouse"]
            person_id = row["person"]

            # Build route dynamically
            route = build_route(warehouse_id, platform, points, warehouses)
            
            # Compute current position (based on simulation time)
            x, y = interpolate_position(route, current_time, walk_speed)

            # Add to plot as orange marker
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                text=[person_id],
                textposition="top center",
                name="Humans",
                marker=dict(size=10, color="orange", symbol="circle"),
                showlegend=False
            ))

# -------------------------
# Clock on top right
# -------------------------
total_minutes = base_minute + current_time
display_hour = base_hour + total_minutes // 60
display_minute = total_minutes % 60
clock_str = f"{display_hour:02d}:{display_minute:02d}"
st.markdown(f"""
<div style='text-align: right; font-size:48px;'>
    ⏰ {clock_str}
</div>
""", unsafe_allow_html=True)

# Show chart
st.plotly_chart(fig, use_container_width=True)
# Fix chart axis ranges
fig.update_xaxes(range=[0, 500])
fig.update_yaxes(range=[-100, 200])
fig.update_layout(autosize=False)

# -------------------------
# Show package text summary
# -------------------------
if "pkg_text" in st.session_state:
    pkg_text = st.session_state["pkg_text"].copy()

    # Format generated_time as HH:MM (base 09:00)
    base_hour = 9
    pkg_text["generated_time"] = pkg_text["generated_time"].apply(
        lambda t: f"{base_hour + t // 60:02d}:{t % 60:02d}"
    )

    st.markdown("**Generated Packages:**")
    st.dataframe(pkg_text)

# -------------------------
# Assign Packages Button & Results
# -------------------------
assign_clicked = st.sidebar.button("Assign Packages")

# Only generate assignments if clicked or if already exist in session_state
if assign_clicked or ("summary_df" in st.session_state and "per_train_detail" in st.session_state):
    if assign_clicked:
        if "packages" not in st.session_state:
            st.warning("No packages available. Generate packages first.")
        else:
            pkgs = st.session_state["packages"].copy()
            if 'package_id' not in pkgs.columns or 'warehouse_id' not in pkgs.columns:
                st.error("packages table missing required columns ('package_id', 'warehouse_id').")
            else:
                assignments_df, summary_df, per_train_detail, meta = assign_packages(
                    pkgs, trains, warehouses, int(max_packages_per_person)
                )
                # Persist results for future reruns
                st.session_state['assignments_df'] = assignments_df
                st.session_state['summary_df'] = summary_df
                st.session_state['per_train_detail'] = per_train_detail
                st.session_state['assignment_meta'] = meta
                st.success(f"Assigned {meta['total_packages']} packages -> {meta['total_persons']} persons")

    # Use session_state for display (works after reruns)
    summary_df = st.session_state["summary_df"]
    per_train_detail = st.session_state["per_train_detail"]

    # --- Show summary table ---
    st.markdown("**Assignment Summary (train × warehouse):**")
    st.dataframe(summary_df.fillna(0).set_index('train_id'))

    # -------------------------
    # Drill-down Buttons
    # -------------------------
    train_options = list(summary_df['train_id'])
    if "selected_train" not in st.session_state:
        st.session_state["selected_train"] = train_options[0] if train_options else None

    cols = st.columns(len(train_options))
    for i, train_id in enumerate(train_options):
        with cols[i]:
            if st.button(f"🚆 {train_id}", key=f"train_{train_id}"):
                st.session_state["selected_train"] = train_id

    # Show details for selected train
    selected_train = st.session_state["selected_train"]
    if selected_train:
        st.markdown(f"**Details for {selected_train}:**")
        detail = per_train_detail.get(selected_train, pd.DataFrame())
        if detail.empty:
            st.info("No assignment details for selected train.")
        else:
            detail_disp = detail.copy()
            detail_disp["packages"] = detail_disp["packages"].apply(lambda lst: ",".join(lst))
            detail_disp = detail_disp[["warehouse", "person", "packages", "count"]]
            detail_disp = detail_disp.rename(
                columns={
                    "warehouse": "Warehouse",
                    "person": "Person",
                    "packages": "Package IDs",
                    "count": "Count",
                }
            )
            st.dataframe(detail_disp)

# -------------------------
# Info text
# -------------------------
st.info("Use the button in the sidebar to move time forward or backward.")
