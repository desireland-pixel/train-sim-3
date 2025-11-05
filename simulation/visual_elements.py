# simulation/visual_elements.py
import plotly.graph_objects as go
import pandas as pd

def draw_warehouses(fig, warehouses_df):
    fig.add_trace(go.Scatter(
        x=warehouses_df.x, y=warehouses_df.y,
        mode="markers+text",
        text=warehouses_df.warehouse_id,
        name="Warehouses",
        marker=dict(size=15, color="green", symbol="square"),
        textposition="top center",
        textfont=dict(color="black")
    ))

def draw_platforms(fig, platforms_df):
    fig.add_trace(go.Scatter(
        x=platforms_df.x, y=platforms_df.y,
        mode="markers+text",
        text=[f"P{i}" for i in platforms_df.platform],
        name="Platforms",
        marker=dict(size=18, color="blue")
    ))

def draw_trains(fig, train_positions):
    fig.add_trace(go.Scatter(
        x=[x for _, x, _ in train_positions],
        y=[y for _, _, y in train_positions],
        text=[tid for tid, _, _ in train_positions],
        mode="markers+text",
        name="Trains",
        marker=dict(size=20, color="red"),
        textfont=dict(color="black"),
        textposition="middle left"
    ))

def draw_packages(fig, package_positions):
    for pkg_id, wh_id, x, y in package_positions:
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            text=[pkg_id],
            textposition="bottom center",
            marker=dict(size=8, color="#D2B48C", symbol="square",
                        line=dict(color="black", width=0.25)),
            name="Packages",
            showlegend=True
        ))

def draw_humans(fig, human_positions):
    for person_id, x, y in human_positions:
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            text=[person_id],
            textposition="top center",
            marker=dict(size=10, color="orange", symbol="circle"),
            name="Humans",
            showlegend=False
        ))
