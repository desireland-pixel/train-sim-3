# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from simulation.data_loader import load_all
from simulation.time_controller import clamp_time
from simulation.train_movement import compute_train_positions
from simulation.human_assignment import assign_packages
from simulation.human_routes import build_route, interpolate_position
from simulation.package_layout import compute_package_positions
from simulation.visual_elements import (
    draw_warehouses,
    draw_platforms,
    draw_trains,
    draw_packages,
    draw_humans
)

# -------------------------
# STREAMLIT CONFIG
# -------------------------
st.set_page_config(layout="wide", page_title="Warehouse & Train Simulator")
st.title("🚉 Train / Warehouse / Human Movement Simulation")

# -------------------------
# LOAD DATA
# -------------------------
data = load_all()
trains = data["trains"]
warehouses = data["warehouses"]
packages = data["packages"]
persons = data["persons"]
points = data["points"]

# Default walking speed
HUMAN_WALK_SPEED = 50  # units/minute

# -------------------------
# SIDEBAR CONTROLS
# -------------------------
st.sidebar.header("Simulation Settings")

# Max packages per human
max_packages_per_person = st.sidebar.number_input(
    "Max packages a person can carry", 1, 10, 5
)

# Simulation time slider (1 min step, with + / – buttons)
current_time = st.sidebar.number_input(
    "Simulation Time (minutes)", min_value=0, max_value=1440, value=0, step=1
)
current_time = clamp_time(current_time)

# Orders per train
st.sidebar.markdown("### Orders per Train")
train_orders = []
train_ids = trains['train_id'].tolist() if not trains.empty else []

if 'dynamic_orders' not in st.session_state:
    st.session_state.dynamic_orders = {}

for train_id in train_ids:
    input_key = f"orders_for_{train_id}"
    if input_key not in st.session_state.dynamic_orders:
        st.session_state.dynamic_orders[input_key] = 0

    val = st.sidebar.number_input(
        f"{train_id} Orders",
        min_value=0,
        max_value=20,
        value=st.session_state.dynamic_orders[input_key],
        key=input_key
    )
    train_orders.append(val)

# Sidebar buttons
generate_packages_clicked = st.sidebar.button("Generate Packages")
assign_packages_clicked = st.sidebar.button("Assign Packages")

# -------------------------
# PACKAGE GENERATION
# -------------------------
if generate_packages_clicked:
    gen_packages = []
    for i, train_id in enumerate(train_ids, 1):
        n_orders = train_orders[i - 1]
        if n_orders > 0:
            start_time = int(trains.loc[trains.train_id == train_id, "start_time"].values[0])
            for j in range(1, n_orders + 1):
                pkg_id = f"{i:02d}{j:02d}"
                warehouse_id = pd.Series(['W1','W2','W3','W4','W5','W6']).sample(1).values[0]
                gen_packages.append({
                    "package_id": pkg_id,
                    "warehouse_id": warehouse_id,
                    "generated_time": start_time - 10
                })

    if gen_packages:
        packages = pd.DataFrame(gen_packages)
        st.session_state["packages"] = packages
        packages.to_csv("data/packages.csv", index=False)
        st.success(f"Generated {len(packages)} packages")
    else:
        st.warning("No packages generated. Please enter order counts.")

# -------------------------
# PACKAGE ASSIGNMENT
# -------------------------
if assign_packages_clicked:
    if "packages" not in st.session_state or st.session_state["packages"].empty:
        st.warning("No packages available. Generate packages first.")
    else:
        pkgs = st.session_state["packages"].copy()
        assignments_df, summary_df, per_train_detail, meta = assign_packages(
            pkgs, trains, warehouses, capacity=int(max_packages_per_person)
        )
        st.session_state['assignments_df'] = assignments_df
        st.session_state['summary_df'] = summary_df
        st.session_state['per_train_detail'] = per_train_detail
        st.success(f"Assigned {meta['total_packages']} packages -> {meta['total_persons']} persons")

# -------------------------
# SIMULATION POSITIONS
# -------------------------
# Train positions
train_positions = compute_train_positions(trains, current_time) if not trains.empty else []

# Package positions
packages_to_show = st.session_state.get("packages", packages)
package_positions = compute_package_positions(packages_to_show, warehouses, current_time)

# Human positions (future: will compute based on routes)
human_positions = []

# -------------------------
# DRAW SCENE
# -------------------------
fig = go.Figure()
draw_warehouses(fig, warehouses)

# Platforms are fixed
platforms = pd.DataFrame({
    'platform': [1,2,3,4,5],
    'x': [200,200,200,200,200],
    'y': [150,100,50,0,-50]
})
draw_platforms(fig, platforms)

draw_trains(fig, train_positions)
draw_packages(fig, package_positions)
draw_humans(fig, human_positions)

fig.update_layout(
    width=900,
    height=600,
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    plot_bgcolor="white"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# SUMMARY / ASSIGNMENT DISPLAY
# -------------------------
if "summary_df" in st.session_state:
    st.markdown("**Assignment Summary (train × warehouse)**")
    st.dataframe(st.session_state["summary_df"].fillna(0).set_index('train_id'))

if "per_train_detail" in st.session_state:
    train_options = list(st.session_state["per_train_detail"].keys())
    if train_options:
        selected_train = st.selectbox("Select Train for Details", train_options)
        detail = st.session_state["per_train_detail"][selected_train]
        if not detail.empty:
            detail_disp = detail.copy()
            detail_disp["packages"] = detail_disp["packages"].apply(lambda lst: ",".join(lst))
            detail_disp = detail_disp.rename(columns={
                "warehouse": "Warehouse",
                "person": "Person",
                "packages": "Package IDs",
                "count": "Count"
            })
            st.dataframe(detail_disp)
