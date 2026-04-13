"""Reusable Plotly chart functions for FPL Analytics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.express as px
import plotly.graph_objects as go

if TYPE_CHECKING:
    import pandas as pd


def team_strength_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter plot of attack vs defence ratings for all teams."""
    fig = px.scatter(
        df,
        x="attack",
        y="defence",
        text="team",
        title="Team Strength — Attack vs Defence Ratings",
        labels={"attack": "Attack Rating", "defence": "Defence Rating"},
    )
    fig.update_traces(textposition="top center", marker=dict(size=10))
    fig.update_layout(template="plotly_white")
    return fig


def fixture_difficulty_heatmap(df: pd.DataFrame, teams: list[str], gws: list[int]) -> go.Figure:
    """Heatmap of fixture difficulty: teams × gameweeks."""
    fig = go.Figure(
        data=go.Heatmap(
            z=df.values,
            x=[f"GW{gw}" for gw in gws],
            y=teams,
            colorscale="RdYlGn_r",
            text=df.values,
            texttemplate="%{text:.2f}",
            hovertemplate="Team: %{y}<br>GW: %{x}<br>Difficulty: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Fixture Difficulty Calendar",
        template="plotly_white",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def projection_bar_chart(df: pd.DataFrame, n: int = 20) -> go.Figure:
    """Bar chart of top N players by projected points."""
    top = df.nlargest(n, "total_xpts")
    fig = px.bar(
        top,
        x="name",
        y="total_xpts",
        color="position",
        title=f"Top {n} Players by Projected Points",
        labels={"total_xpts": "Projected Points", "name": "Player"},
    )
    fig.update_layout(template="plotly_white", xaxis_tickangle=-45)
    return fig
