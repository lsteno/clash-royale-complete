
import streamlit as st
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Set Page Config
st.set_page_config(
    page_title="DreamerV3 Clash Flagship Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4461; }
    .stProgress .st-bo { background-color: #00ff00; }
    h1, h2, h3 { color: #00d4ff !important; font-family: 'Inter', sans-serif; }
</style>
""", unsafe_allow_html=True)

# Path to logdir
LOGDIR = Path("./logdir/clash_v3_flagship")

def load_metrics():
    """Parse metrics.jsonl from dreamer.py logger."""
    metrics_file = LOGDIR / "metrics.jsonl"
    if not metrics_file.exists():
        return None
    
    data = []
    with open(metrics_file, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    return pd.DataFrame(data)

def get_latest_screencaps():
    """Find recent .npz files in train_eps and extract images."""
    eps_dir = LOGDIR / "train_eps"
    if not eps_dir.exists():
        return []
    
    files = sorted(list(eps_dir.glob("*.npz")), key=os.path.getmtime, reverse=True)
    images = []
    for f in files[:4]: # Last 4 episodes
        try:
            with np.load(f) as data:
                if 'image' in data:
                    # npz stores (T, C, H, W) or (T, H, W, C)
                    img = data['image'][-1] # Last frame
                    if img.shape[0] == 3: # C,H,W
                         img = img.transpose(1, 2, 0)
                    images.append((f.stem, img))
        except:
            continue
    return images

# --- SIDEBAR ---
st.sidebar.title("🛡️ Flagship Controls")
st.sidebar.info("Monitoring 4 Parallel Emulators\nXL World Model Architecture")
auto_refresh = st.sidebar.checkbox("Auto Refresh (5s)", value=True)

if auto_refresh:
    time.sleep(5)
    st.rerun()

# --- MAIN DASHBOARD ---
st.title("🛡️ DreamerV3: Clash Royale Command Center")

df = load_metrics()

if df is not None and not df.empty:
    # 1. Top Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    
    total_steps = df['step'].max() if 'step' in df else 0
    c1.metric("Total Env Steps", f"{total_steps:,}")
    
    if 'train_return' in df:
        latest_ret = df['train_return'].dropna().iloc[-1]
        c2.metric("Latest Return", f"{latest_ret:.1f}")
    
    if 'train_length' in df:
        avg_len = df['train_length'].mean()
        c3.metric("Avg Episode Length", f"{avg_len:.0f}")

    if 'step' in df:
        # Simple FPS estimate (last 10 rows)
        if len(df) > 10:
            last_steps = df['step'].iloc[-1] - df['step'].iloc[-10]
            c4.metric("Throughput", f"{last_steps/10:.1f} steps/log")

    # 2. Charts
    st.subheader("📈 Training Progress")
    colA, colB = st.columns(2)
    
    if 'train_return' in df:
        with colA:
            st.line_chart(df.set_index('step')['train_return'].dropna(), use_container_width=True)
            st.caption("Environment Returns over Steps")
            
    if 'model_loss' in df:
        with colB:
            st.line_chart(df.set_index('step')['model_loss'].dropna(), use_container_width=True)
            st.caption("World Model Loss (Dynamic Learning)")

    # 3. Live Replay Buffer
    st.subheader("🎥 Live Agent Perspective")
    screens = get_latest_screencaps()
    if screens:
        cols = st.columns(len(screens))
        for i, (name, img) in enumerate(screens):
            with cols[i]:
                # Rescale if needed
                if img.max() <= 1.0: img = (img * 255).astype(np.uint8)
                st.image(img, caption=f"Episode: {name[:8]}...", use_container_width=True)
    else:
        st.warning("No replay data found yet. Prefill in progress...")

else:
    st.warning("Waiting for first log entry... The model is currently initializing the 4 emulators and prefilling the buffer.")
    st.progress(0)
    st.info("Logdir: " + str(LOGDIR.absolute()))

st.divider()
st.caption("Antigravity Agentic Coding v1.0 | Clash Royale Flagship Project")
