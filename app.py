# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta

from simulation.data_loader import load_all
from simulation.time_controller import clamp_time
from simulation.train_movement import compute_train_positions
from simulation.human_assignment import assign_packages
from simulation.human_routes import build_route, interpolate_position
from simulation.package_layout import compute_package_positions
from simulation.visual_elements import draw_warehouses, draw_platforms, draw_trains, draw_packages, draw_humans

st.set_page_config(layout="wide", page_title="Train-Warehouse Simulation")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# -------------------------
# Load data
# -------------------------
data = load_all()
trains = data["trains"]
warehouses = data["warehouses"]
packages = data["packages"]
persons = data["persons"]
points = data["points"]

# Platforms fixed
platforms = pd.DataFrame({
    'platform': [1, 2, 3, 4, 5],
    'x': [200, 200, 200, 200, 200],
    'y': [150, 100, 50, 0, -50]
})

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Simulation Settings")

# Simulation time slider at top
if "sim_time" not in st.session_state:
    st.session_state.sim_time = 0
time = st.sidebar.number_input(
    "Simulation Time (minutes)", min_value=0, max_value=1440,
    value=st.session_state.sim_time, step=1
)
st.session_state.sim_time = time
time = clamp_time(time)

max_packages_per_person = st.sidebar.number_input("Max packages a person can carry", 1, 10, 5)

st.sidebar.markdown("### Orders per Train")
if "dynamic_orders" not in st.session_state:
    st.session_state.dynamic_orders = {}
train_orders = []
train_ids = trains['train_id'].tolist()
for train_id in train_ids:
    key = f"orders_{train_id}"
    if key not in st.session_state.dynamic_orders:
        st.session_state.dynamic_orders[key] = 0
    val = st.sidebar.number_input(
        f"{train_id} Orders", min_value=0, max_value=20,
        value=st.session_state.dynamic_orders[key],
        key=key
    )
    train_orders.append(val)
    st.session_state.dynamic_orders[key] = val

generate_clicked = st.sidebar.button("Generate Packages")
assign_clicked = st.sidebar.button("Assign Packages")

# -------------------------
# Messages handling
# -------------------------
msg_time = ""
msg_orders = ""
msg_generated = ""
msg_assigned = ""

if time == 0:
    msg_time = "Change simulation time to see the magic!"
if all(o == 0 for o in train_orders):
    msg_orders = "Add no. of order(s) on the left side"

# Detect change in orders
if "prev_orders" not in st.session_state:
    st.session_state.prev_orders = train_orders.copy()
elif st.session_state.prev_orders != train_orders:
    msg_orders = "No. of order(s) has/have been changed, click 'Generate Packages'"
    st.session_state.prev_orders = train_orders.copy()

# -------------------------
# Package generation
# -------------------------
if generate_clicked:
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
        packages = pd.DataFrame(gen_packages)
        st.session_state["packages"] = packages
        msg_generated = f"{len(packages)} packages generated, click 'Assign Packages' to distribute the packages."
    else:
        packages = pd.DataFrame()
        st.session_state.pop("packages", None)

# -------------------------
# Assign packages
# -------------------------
assignments_df = pd.DataFrame()
summary_df = pd.DataFrame()
per_train_detail = {}
if assign_clicked or ("assignments_df" in st.session_state):
    if "packages" in st.session_state:
        pkgs = st.session_state["packages"]
        assignments_df, summary_df, per_train_detail, meta = assign_packages(
            pkgs, trains, warehouses, capacity=int(max_packages_per_person)
        )
        st.session_state["assignments_df"] = assignments_df
        st.session_state["summary_df"] = summary_df
        st.session_state["per_train_detail"] = per_train_detail
        msg_assigned = f"Assigned {meta['total_packages']} packages → {meta['total_persons']} persons."

# -------------------------
# Package Positions
# -------------------------
package_positions = compute_package_positions(st.session_state.get("packages", pd.DataFrame()), warehouses, time)

# -------------------------
# Human positions (empty for now)
# -------------------------
human_positions = []

# -------------------------
# Train positions
# -------------------------
train_positions = compute_train_positions(trains, time) if not trains.empty else []

# -------------------------
# Page title
# -------------------------
st.title("🚉 Train–Warehouse Simulation")
#st.markdown(f"**Simulation Time: {current_time} min**")

# -------------------------
# DIGITAL CLOCK ABOVE GRAPH
# -------------------------
sim_clock = (datetime(2000,1,1,9,0) + timedelta(minutes=time)).strftime("%H:%M")
st.markdown(f"### Time: {sim_clock}")

# -------------------------
# Status Message - 1
# -------------------------
if msg_time:
    st.info(msg_time)
    
# -------------------------
# PLOTLY FIGURE
# -------------------------
fig = go.Figure()
draw_warehouses(fig, warehouses)
draw_platforms(fig, platforms)
draw_trains(fig, train_positions)
draw_packages(fig, package_positions)
draw_humans(fig, human_positions)

fig.update_xaxes(range=[-50, 500])
fig.update_yaxes(range=[-100, 200])
fig.update_layout(
    width=900,
    height=600,
    plot_bgcolor="white",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    autosize=False
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Status Message - 2
# -------------------------
if msg_orders:
    st.warning(msg_orders)
if msg_generated:
    st.success(msg_generated)

# -------------------------
# Show generated packages
# -------------------------
if "packages" in st.session_state and not st.session_state["packages"].empty:
    pkg_text = st.session_state["packages"].copy()
    pkg_text["generated_time"] = pkg_text["generated_time"].apply(
        lambda t: f"{9 + t//60:02d}:{t%60:02d}"
    )
    st.markdown("**Generated Packages:**")
    st.dataframe(pkg_text)

# -------------------------
# Show assignment message
# -------------------------
if msg_assigned:
    st.success(msg_assigned)

# -------------------------------------------------------------
# Common Strategy: Determine the final, ordered list of trains with details
# -------------------------------------------------------------

ordered_train_ids_with_details = []

if "per_train_detail" in st.session_state:
    per_train_detail = st.session_state["per_train_detail"]
    
    if per_train_detail:
        # Filter the original list to only include those present in per_train_detail
        # This preserves the original order (A C B E D) while only selecting the valid ones (e.g., A C B)
        ordered_train_ids_with_details = [
            train_id for train_id in train_ids if train_id in per_train_detail
        ]

# -------------------------------------------------------------
# Assignment Summary (train × warehouse) - Uses common list
# -------------------------------------------------------------
if "summary_df" in st.session_state and not st.session_state["summary_df"].empty:
    st.markdown("**Assignment Summary (train × warehouse)**")
    
    summary_df_display = st.session_state["summary_df"].fillna(0).set_index('train_id')
    
    # Use the common strategy list to enforce order
    summary_df_display = summary_df_display.reindex(ordered_train_ids_with_details)
    
    # Optional safety net: remove rows that might be empty if a train_id had no summary data
    summary_df_display = summary_df_display.dropna(axis=0, how='all')
    
    st.dataframe(summary_df_display)

# -------------------------------------------------------------
# Train detail buttons - Uses common list
# -------------------------------------------------------------
if ordered_train_ids_with_details:
    st.markdown("**Select Train to see details:**")
    
    # Create columns exactly matching the common list length
    cols = st.columns(len(ordered_train_ids_with_details))
    
    # Iterate over the common list
    for i, train_id in enumerate(ordered_train_ids_with_details):
        with cols[i]:
            if st.button(f"🚆 {train_id}", key=f"train_btn_{train_id}"):
                st.session_state["selected_train"] = train_id
    # -------------------------
        
    selected_train = st.session_state.get("selected_train", list(per_train_detail.keys())[0])
    detail = per_train_detail.get(selected_train, pd.DataFrame())
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
