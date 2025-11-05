import streamlit as st
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

platforms = pd.DataFrame({
    'platform': [1,2,3,4,5],
    'x': [200,200,200,200,200],
    'y': [150,100,50,0,-50]
})
draw_platforms(fig, platforms)

# --------------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------------
st.set_page_config(layout="wide", page_title="Warehouse & Train Simulator")
st.title("Train / Warehouse / Human Movement Simulation")

data = load_all()
trains = data["trains"]
warehouses = data["warehouses"]
packages = data["packages"]
persons = data["persons"]
points = data["points"]

# Default walking speed (units per minute)
HUMAN_WALK_SPEED = 50

# --------------------------------------------------------
# Time Controls
# --------------------------------------------------------
time = st.slider("Simulation Time (minutes)", 0, 1440, 100, 1)
time = clamp_time(time)

# --------------------------------------------------------
# Train Movement
# --------------------------------------------------------
if "start_time" in trains.columns and "arrive_time" in trains.columns:
    train_positions = compute_train_positions(trains, time)
else:
    train_positions = []

# --------------------------------------------------------
# Package Assignment (uses your existing assignment engine)
# --------------------------------------------------------
if not packages.empty:
    assignments_df, summary_df, per_train_detail, meta = assign_packages(
        packages, trains, warehouses, capacity=5
    )
else:
    assignments_df = None

# --------------------------------------------------------
# Package Placement on Map
# --------------------------------------------------------
package_positions = compute_package_positions(packages, warehouses, time)

# --------------------------------------------------------
# Human Movement (future: when we activate routes)
# --------------------------------------------------------
human_positions = []  # <--- stays empty for now (no movement yet)

# --------------------------------------------------------
# DRAW SCENE
# --------------------------------------------------------
fig = go.Figure()
draw_warehouses(fig, warehouses)
draw_platforms(fig, trains)  # platforms are in trains CSV in your version
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
