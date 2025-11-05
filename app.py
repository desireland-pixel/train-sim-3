# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

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
# STREAMLIT PAGE CONFIG
# -------------------------
st.set_page_config(layout="wide", page_title="Train / Warehouse / Human Simulation")
st.title("🚉 Train / Warehouse / Human Movement Simulation")

# -------------------------
# Load data
# -------------------------
data = load_all()
trains = data["trains"]
warehouses = data["warehouses"]
packages = data["packages"]
persons = data["persons"]
points = data["points"]

train_ids = trains['train_id'].tolist() if not trains.empty else []

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Simulation Settings")

# Max packages per person
max_packages_per_person = st.sidebar.number_input(
    "Max packages a person can carry", min_value=1, max_value=10, value=5
)

# Orders per train
st.sidebar.markdown("### Orders per Train")
if 'dynamic_orders' not in st.session_state:
    st.session_state.dynamic_orders = {tid: 0 for tid in train_ids}

train_orders = []
for train_id in train_ids:
    val = st.sidebar.number_input(
        f"{train_id} Orders",
        min_value=0, max_value=20,
        value=st.session_state.dynamic_orders.get(train_id, 0),
        key=f"orders_{train_id}"
    )
    st.session_state.dynamic_orders[train_id] = val
    train_orders.append(val)

# +/- buttons + number_input for simulation time
if 'sim_time' not in st.session_state:
    st.session_state.sim_time = 0

col1, col2 = st.sidebar.columns([1,1])
if col1.button("-"):
    st.session_state.sim_time = max(0, st.session_state.sim_time - 1)
if col2.button("+"):
    st.session_state.sim_time = min(1440, st.session_state.sim_time + 1)

time = st.sidebar.number_input(
    "Simulation Time (minutes)",
    min_value=0, max_value=1440,
    value=st.session_state.sim_time,
    step=1
)
st.session_state.sim_time = time
time = int(clamp_time(time))

# -------------------------
# Package generation & assignment
# -------------------------
# Button: Generate Packages
if st.sidebar.button("Generate Packages"):
    gen_packages = []
    for i, train_id in enumerate(train_ids, 1):
        n_orders = train_orders[i-1]
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
        packages = pd.DataFrame(gen_packages)
        st.session_state["packages"] = packages
        st.success(f"{len(packages)} packages generated, click 'Assign Packages' to distribute them")
        st.session_state["assignments_df"] = None  # clear old assignments
    else:
        st.warning("No orders entered — no packages generated.")

# Button: Assign Packages
assign_clicked = st.sidebar.button("Assign Packages")
if assign_clicked and "packages" in st.session_state:
    assignments_df, summary_df, per_train_detail, meta = assign_packages(
        st.session_state["packages"], trains, warehouses, max_packages_per_person
    )
    st.session_state["assignments_df"] = assignments_df
    st.session_state["summary_df"] = summary_df
    st.session_state["per_train_detail"] = per_train_detail
    st.session_state["assignment_meta"] = meta
    st.success(f"Assigned {meta['total_packages']} packages → {meta['total_persons']} persons")

# -------------------------
# Show messages for orders / time
# -------------------------
msg_area = st.empty()
if time == 0:
    if sum(train_orders) == 0:
        msg_area.info("Change simulation time to see the changes\nAdd number of orders on the left side")
    else:
        msg_area.info("Change simulation time to see the changes")
elif sum(train_orders) == 0:
    msg_area.info("Add number of orders on the left side")

# -------------------------
# Clock above the graph
# -------------------------
sim_clock = (datetime(2000,1,1,9,0) + timedelta(minutes=time)).strftime("%H:%M")
st.markdown(f"<h3 style='text-align:center'>Time: {sim_clock}</h3>", unsafe_allow_html=True)

# -------------------------
# Compute positions
# -------------------------
train_positions = compute_train_positions(trains, time) if not trains.empty else []

packages_df = st.session_state.get("packages", pd.DataFrame())
package_positions = compute_package_positions(packages_df, warehouses, time)

# Human positions (empty for now)
human_positions = []

# -------------------------
# DRAW SCENE
# -------------------------
fig = go.Figure()
draw_warehouses(fig, warehouses)

# Platforms are fixed
platforms_df = pd.DataFrame({
    'platform': [1,2,3,4,5],
    'x': [200,200,200,200,200],
    'y': [150,100,50,0,-50]
})
draw_platforms(fig, platforms_df)
draw_trains(fig, train_positions)
draw_packages(fig, package_positions)
draw_humans(fig, human_positions)

fig.update_layout(
    width=900,
    height=600,
    xaxis=dict(visible=False, range=[-50, 500]),
    yaxis=dict(visible=False, range=[-100, 200]),
    plot_bgcolor="white",
    autosize=False
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Show Generated Packages Table
# -------------------------
if not packages_df.empty:
    st.markdown("**Generated Packages:**")
    pkg_text = packages_df.copy()
    pkg_text["generated_time"] = pkg_text["generated_time"].apply(
        lambda t: f"{9 + t//60:02d}:{t%60:02d}"
    )
    st.dataframe(pkg_text)

# -------------------------
# Show Assignment Summary
# -------------------------
summary_df = st.session_state.get("summary_df", pd.DataFrame())
per_train_detail = st.session_state.get("per_train_detail", {})

if not summary_df.empty:
    st.markdown("**Assignment Summary (train × warehouse):**")
    st.dataframe(summary_df.fillna(0).set_index('train_id'))

# -------------------------
# Train buttons for details
# -------------------------
train_with_orders = [tid for tid in train_ids if tid in per_train_detail]
if train_with_orders:
    cols = st.columns(len(train_with_orders))
    for i, train_id in enumerate(train_with_orders):
        with cols[i]:
            if st.button(f"🚆 {train_id}", key=f"train_btn_{train_id}"):
                st.session_state["selected_train"] = train_id

# Show details for selected train
selected_train = st.session_state.get("selected_train", train_with_orders[0] if train_with_orders else None)
if selected_train and selected_train in per_train_detail:
    st.markdown(f"**Details for {selected_train}:**")
    detail = per_train_detail[selected_train]
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
