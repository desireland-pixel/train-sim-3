import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from simulation.data_loader import load_static_data, load_timetable
from simulation.time_controller import SimulationClock
from simulation.train_movement import compute_train_positions
from simulation.human_assignment import assign_packages_to_warehouses
from simulation.human_routes import compute_human_positions
from simulation.package_layout import compute_package_positions
from simulation.visual_elements import (
    draw_warehouses,
    draw_platforms,
    draw_trains,
    draw_packages,
    draw_humans,
)


# --------------------------------------------------------
# Streamlit Page Setup
# --------------------------------------------------------
st.set_page_config(layout="wide", page_title="Train Simulation v2")

st.title("Train & Warehouse Operation Simulator")


# --------------------------------------------------------
# Data Loading
# --------------------------------------------------------
warehouses, platforms, points = load_static_data("data/")
timetable = load_timetable("data/trains.csv")

# UI: Select a simulation time
selected_time = st.slider("Simulation Time (minutes)", 0, 1440, 50, step=1)

clock = SimulationClock(selected_time)


# --------------------------------------------------------
# Train Movement
# --------------------------------------------------------
train_positions = compute_train_positions(timetable, clock.current_time)


# --------------------------------------------------------
# Order & Human Assignment
# --------------------------------------------------------
# (Later: UI form → user adds orders → orders assigned to warehouse)
# For now: we use a placeholder "no orders" state

orders = []  # placeholder for now

warehouse_load = assign_packages_to_warehouses(orders, warehouses)


# --------------------------------------------------------
# Package Placement Around Warehouses
# --------------------------------------------------------
package_positions = compute_package_positions(warehouse_load, warehouses)


# --------------------------------------------------------
# Human Movement (Future Phase)
# --------------------------------------------------------
human_positions = compute_human_positions(clock.current_time, points, warehouse_load)


# --------------------------------------------------------
# Visualization
# --------------------------------------------------------
fig = go.Figure()

draw_warehouses(fig, warehouses)
draw_platforms(fig, platforms)
draw_trains(fig, train_positions)
draw_packages(fig, package_positions)
draw_humans(fig, human_positions)

fig.update_layout(
    width=900,
    height=700,
    showlegend=False,
    plot_bgcolor="white",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
)

st.plotly_chart(fig, use_container_width=True)
