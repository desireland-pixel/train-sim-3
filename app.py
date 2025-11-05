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

# --------------------------------------------------------
# STREAMLIT PAGE SETUP
# --------------------------------------------------------
st.set_page_config(layout="wide", page_title="Train / Warehouse / Human Simulator")
st.title("🚉 Train / Warehouse / Human Movement Simulation")

# --------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------
data = load_all()
trains = data["trains"]
warehouses = data["warehouses"]
packages = data["packages"]
persons = data["persons"]
points = data["points"]

# --------------------------------------------------------
# PARAMETERS
# --------------------------------------------------------
HUMAN_WALK_SPEED = 50  # units per minute

# --------------------------------------------------------
# TIME CONTROL
# --------------------------------------------------------
time = st.slider("Simulation Time (minutes)", 0, 1440, 100, 1)
time = clamp_time(time)

# --------------------------------------------------------
# INITIALIZE FIGURE
# --------------------------------------------------------
fig = go.Figure()

# --------------------------------------------------------
# DRAW STATIC ELEMENTS
# --------------------------------------------------------
# Warehouses
draw_warehouses(fig, warehouses)

# Platforms (fixed)
platforms = pd.DataFrame({
    'platform': [1, 2, 3, 4, 5],
    'x': [200, 200, 200, 200, 200],
    'y': [150, 100, 50, 0, -50]
})
draw_platforms(fig, platforms)

# --------------------------------------------------------
# TRAIN MOVEMENT
# --------------------------------------------------------
if "start_time" in trains.columns and "arrive_time" in trains.columns:
    train_positions = compute_train_positions(trains, time)
else:
    train_positions = []
draw_trains(fig, train_positions)

# --------------------------------------------------------
# PACKAGE ASSIGNMENT
# --------------------------------------------------------
if packages is not None and not packages.empty:
    assignments_df, summary_df, per_train_detail, meta = assign_packages(
        packages, trains, warehouses, capacity=5
    )
else:
    assignments_df = None

# --------------------------------------------------------
# PACKAGE PLACEMENT
# --------------------------------------------------------
package_positions = compute_package_positions(packages, warehouses, time)
draw_packages(fig, package_positions)

# --------------------------------------------------------
# HUMAN MOVEMENT (placeholder)
# --------------------------------------------------------
human_positions = []  # future logic to compute positions based on routes & assignments
draw_humans(fig, human_positions)

# --------------------------------------------------------
# FIGURE LAYOUT
# --------------------------------------------------------
fig.update_layout(
    width=900,
    height=600,
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    plot_bgcolor="white",
    title=f"Simulation Time: {time} min"
)

# --------------------------------------------------------
# SHOW FIGURE
# --------------------------------------------------------
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------
# PACKAGE ASSIGNMENT SUMMARY (optional)
# --------------------------------------------------------
if assignments_df is not None and not assignments_df.empty:
    st.markdown("**Package Assignment Summary:**")
    st.dataframe(summary_df.fillna(0).set_index('train_id'))
