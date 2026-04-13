"""Page 3 — Team Strengths.

Scatter plot and bar charts of Dixon-Coles attack/defence ratings.
Optional comparison with a user-pasted reference dataset.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.data_loader import load_model

st.set_page_config(page_title="Team Strengths · FPL Analytics", layout="wide")
st.title("💪 Team Strengths")

# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("Loading model…"):
    try:
        ts = load_model()
    except Exception as e:
        st.error(f"Could not load team strengths: {e}")
        st.stop()

df = ts.to_dataframe()

# ── Model quality metrics ─────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Teams modelled", len(df))
c2.metric("Log-likelihood", f"{ts.log_likelihood:.1f}")
c3.metric("Home advantage", f"{ts.home_advantage:.3f}")
c4.metric("Rho (ρ)", f"{ts.rho:.4f}")
st.caption(
    "Home advantage > 1 means home teams score more goals. "
    "Rho adjusts 0-0, 1-0, 0-1, 1-1 probabilities away from pure Poisson."
)

st.divider()

# ── Scatter: attack vs defence ────────────────────────────────────────────────
st.subheader("Attack vs Defence")

fig_scatter = px.scatter(
    df,
    x="attack",
    y="defence",
    text="team",
    color="overall",
    color_continuous_scale="RdYlGn",
    size=[1] * len(df),
    size_max=15,
    hover_data={"attack": ":.3f", "defence": ":.3f", "overall": ":.3f"},
    labels={"attack": "Attack Rating", "defence": "Defence Rating", "overall": "Overall"},
    title="Team Strength Ratings (attack vs defence)",
)
fig_scatter.update_traces(textposition="top center", marker=dict(size=12))
fig_scatter.add_hline(y=df["defence"].mean(), line_dash="dot", line_color="grey", opacity=0.5)
fig_scatter.add_vline(x=df["attack"].mean(), line_dash="dot", line_color="grey", opacity=0.5)
fig_scatter.update_layout(template="plotly_white", height=500)
st.plotly_chart(fig_scatter, use_container_width=True)

# ── Bar charts ────────────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("Attack Rankings")
    fig_att = px.bar(
        df.sort_values("attack", ascending=True),
        x="attack", y="team",
        orientation="h",
        color="attack",
        color_continuous_scale="RdYlGn",
        labels={"attack": "Attack Rating", "team": ""},
    )
    fig_att.update_layout(template="plotly_white", height=550, showlegend=False,
                          coloraxis_showscale=False)
    st.plotly_chart(fig_att, use_container_width=True)

with col_r:
    st.subheader("Defence Rankings")
    st.caption("Lower = stronger defence")
    fig_def = px.bar(
        df.sort_values("defence", ascending=False),
        x="defence", y="team",
        orientation="h",
        color="defence",
        color_continuous_scale="RdYlGn_r",
        labels={"defence": "Defence Rating", "team": ""},
    )
    fig_def.update_layout(template="plotly_white", height=550, showlegend=False,
                          coloraxis_showscale=False)
    st.plotly_chart(fig_def, use_container_width=True)

# ── Raw data table ────────────────────────────────────────────────────────────
st.divider()
with st.expander("Show raw ratings table"):
    st.dataframe(
        df.style.format({"attack": "{:.4f}", "defence": "{:.4f}", "overall": "{:.4f}"}),
        use_container_width=True,
    )

# ── Comparison with external source ──────────────────────────────────────────
st.divider()
st.subheader("Compare with external ratings (e.g. Solio)")
st.caption(
    "Paste a CSV with columns `team,attack,defence` into the text box below "
    "to compare against our ratings side by side."
)

csv_input = st.text_area("Paste CSV here (team,attack,defence)", height=150)
if csv_input.strip():
    try:
        import io
        other_df = pd.read_csv(io.StringIO(csv_input))
        comparison = ts.compare(other_df)
        st.dataframe(
            comparison.style.format({
                c: "{:.4f}" for c in comparison.select_dtypes("float").columns
            }).background_gradient(
                subset=["attack_diff", "defence_diff"], cmap="RdYlGn"
            ),
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")
