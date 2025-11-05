# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta, datetime

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

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("Simulation Settings")

# Simulation Time Slider at the top
if "sim_time" not in st.session_state:
    st.session_state.sim_time = 0

time = st.sidebar.number_input(
    "Simulation Time (minutes)",
    min_value=0, max_value=1440,
    value=st.session_state.sim_time,
    step=1
)
time = clamp_time(time)
st.session_state.sim_time = time

# Digital Clock
start_hour = 9
sim_clock = (datetime(2000,1,1,start_hour,0) + timedelta(minutes=time)).strftime("%H:%M")
st.sidebar.markdown(f"**Time:** {sim_clock}")

# Max packages per person
max_packages_per_person = st.sidebar.number_input("Max packages a person", 1, 10, 5)

# Orders per train
st.sidebar.markdown("### Orders per Train")
if 'dynamic_orders' not in st.session_state:
    st.session_state.dynamic_orders = {}

train_orders = []
train_ids = trains['train_id'].tolist()
orders_changed = False
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
    if current_value != st.session_state.dynamic_orders[input_key]:
        orders_changed = True
        st.session_state.dynamic_orders[input_key] = current_value
    train_orders.append(current_value)

# -----------------------------
# MESSAGES BEFORE ANY ACTION
# -----------------------------
if time == 0:
    st.info("Change simulation time to see the changes")
if all(o == 0 for o in train_orders):
    st.warning("Add no of orders on the left side")
elif orders_changed:
    st.warning("No of orders have been changed, click 'Generate packages'")

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
        st.success(f"{len(gen_packages)} packages generated, click 'Assign packages' to distribute the packages")
        # Show assignment summary table immediately (empty, as not assigned yet)
        assignments_df, summary_df, per_train_detail, meta = assign_packages(
            st.session_state["packages"], trains, warehouses, max_packages_per_person
        )
        st.session_state["assignments_df"] = assignments_df
        st.session_state["summary_df"] = summary_df
        st.session_state["per_train_detail"] = per_train_detail
        st.session_state["assignment_meta"] = meta
        st.markdown("**Assignment Summary (train × warehouse)**")
        st.dataframe(summary_df.fillna(0).set_index('train_id'))
    else:
        st.warning("No packages generated.")

# -----------------------------
# ASSIGN PACKAGES BUTTON
# -----------------------------
if st.sidebar.button("Assign Packages"):
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
human_positions = []

# -----------------------------
# PLOTLY FIGURE
# -----------------------------
fig = go.Figure()
draw_warehouses(fig, warehouses)

# Draw fixed platforms
platforms = pd.DataFrame({
    'platform': [1, 2, 3, 4, 5],
    'x': [200, 200, 200, 200, 200],
    'y': [150, 100, 50, 0, -50]
})
draw_platforms(fig, platforms)

draw_trains(fig, train_positions)
draw_packages(fig, package_positions)
draw_humans(fig, human_positions)

# FIX AXES to prevent expansion animation
fig.update_xaxes(range=[-50, 300])
fig.update_yaxes(range=[-50, 200])

fig.update_layout(
    width=900,
    height=600,
    plot_bgcolor="white",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False)
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# TRAIN DETAIL BUTTONS
# -----------------------------
if "per_train_detail" in st.session_state:
    per_train_detail = st.session_state["per_train_detail"]
    # Use the sidebar order of train_ids to maintain order
    train_options = [tid for tid in train_ids if tid in per_train_detail and not per_train_detail[tid].empty]

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
