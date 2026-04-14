"""Page 6 — Price Change Alerts.

Embeds LiveFPL's price change tracker directly in the dashboard.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Price Change Alerts · FPL Analytics", layout="wide")
st.title("💰 Price Change Alerts")

st.caption("Data provided by [LiveFPL](https://www.livefpl.net/prices)")

components.iframe("https://www.livefpl.net/prices", height=900, scrolling=True)
