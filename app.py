# app.py
import streamlit as st
import pandas as pd
import numpy as np
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

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(layout="wide", page_title="Train / Warehouse / Human Simulation")
st.title("🚉 Train / Warehouse / Human Movement Simulation")

# -----------------------------
# LOAD DATA
# -----------------------------
data = load_all()
trains = data["trains"]
warehouses = data["warehouses"]
packages = data["packages"]
persons = data["persons"]
points = data["points"]

# Default walking speed (units per minute)
HUMAN_WALK_SPEED = 50

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("Simulation Settings")

# Max packages a person can carry
max_packages_per_person = st.sidebar.number_input("Max packages a person", 1, 10, 5)

# Orders per train
st.sidebar.markdown("### Orders per Train")
if 'dynamic_orders' not in st.session_state:
    st.session_state.dynamic_orders = {}

train_orders = []
train_ids = trains['train_id'].tolist()
for train_id in train_ids:
    input_key = f"orders_for_{train_id}"
    if input_key not in st.session_state.dynamic_orders:
        st.session_state.dynamic_orders[input_key] = 0
    current_value = st.sidebar.number_input(
        f"{train_id} Orders",
        min_value=0,
        max_value=20,
        value=st.session_state.dynamic_orders[input_key],
        key=input_key
    )
    train_orders.append(current_value)

# -----------------------------
# PACKAGE GENERATION BUTTON
# -----------------------------
if st.sidebar.button("Generate Packages"):
    gen_packages = []
    for i, train_id in enumerate(train_ids, 1):
        n_orders = train_orders[i - 1]
        if n_orders > 0:
            start_time = int(trains.loc[trains.train_id == train_id, "start_time"].values[0])
            for j in range(1, n_orders + 1):
                pkg_id = f"{i:02d}{j:02d}"
                warehouse_id = np.random.choice(warehouses.warehouse_id)
                gen_packages.append({
                    "package_id": pkg_id,
                    "warehouse_id": warehouse_id,
                    "generated_time": start_time - 10
                })
    if gen_packages:
        st.session_state["packages"] = pd.DataFrame(gen_packages)
        st.success(f"Generated {len(gen_packages)} packages.")
    else:
        st.warning("No packages generated.")

# -----------------------------
# ASSIGN PACKAGES BUTTON
# -----------------------------
assign_clicked = st.sidebar.button("Assign Packages")
if assign_clicked:
    if "packages" not in st.session_state:
        st.warning("No packages available. Generate packages first.")
    else:
        assignments_df, summary_df, per_train_detail, meta = assign_packages(
            st.session_state["packages"], trains, warehouses, max_packages_per_person
        )
        st.session_state["assignments_df"] = assignments_df
        st.session_state["summary_df"] = summary_df
        st.session_state["per_train_detail"] = per_train_detail
        st.session_state["assignment_meta"] = meta
        st.success(f"Assigned {meta['total_packages']} packages to {meta['total_persons']} persons.")

# -----------------------------
# SIMULATION TIME SLIDER
# -----------------------------
time = st.sidebar.number_input(
    "Simulation Time (minutes)",
    min_value=0, max_value=1440, value=0, step=1
)
time = clamp_time(time)

# -----------------------------
# COMPUTE TRAIN POSITIONS
# -----------------------------
if "start_time" in trains.columns and "arrive_time" in trains.columns:
    train_positions = compute_train_positions(trains, time)
else:
    train_positions = []

# -----------------------------
# PACKAGE POSITIONS
# -----------------------------
package_positions = []
if "packages" in st.session_state:
    package_positions = compute_package_positions(st.session_state["packages"], warehouses, time)

# -----------------------------
# HUMAN MOVEMENT PLACEHOLDER
# -----------------------------
human_positions = []  # future: populate based on assignment + routes

# -----------------------------
# PLOTLY FIGURE
# -----------------------------
fig = go.Figure()
draw_warehouses(fig, warehouses)

# Draw platforms from fixed coordinates
platforms = pd.DataFrame({
    'platform': [1, 2, 3, 4, 5],
    'x': [200, 200, 200, 200, 200],
    'y': [150, 100, 50, 0, -50]
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

# -----------------------------
# ASSIGNMENT SUMMARY
# -----------------------------
if "summary_df" in st.session_state:
    summary_df = st.session_state["summary_df"]
    st.markdown("**Assignment Summary (train × warehouse)**")
    st.dataframe(summary_df.fillna(0).set_index('train_id'))

# -----------------------------
# TRAIN BUTTONS + DETAILS
# -----------------------------
if "per_train_detail" in st.session_state:
    per_train_detail = st.session_state["per_train_detail"]
    train_options = [tid for tid, df in per_train_detail.items() if not df.empty]

    if train_options:
        st.markdown("**Select Train to see details:**")
        cols = st.columns(len(train_options))
        for i, train_id in enumerate(train_options):
            with cols[i]:
                if st.button(f"🚆 {train_id}", key=f"train_{train_id}"):
                    st.session_state["selected_train"] = train_id

        selected_train = st.session_state.get("selected_train", train_options[0])
        detail = per_train_detail[selected_train]
        if not detail.empty:
            detail_disp = detail.copy()
            detail_disp["packages"] = detail_disp["packages"].apply(lambda lst: ",".join(lst))
            detail_disp = detail_disp.rename(columns={
                "warehouse": "Warehouse",
                "person": "Person",
                "packages": "Package IDs",
                "count": "Count"
            })
            st.markdown(f"**Details for {selected_train}:**")
            st.dataframe(detail_disp)
