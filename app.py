# app.py — Smart Parking AI System
# Model: EfficientNetV2B3
# Accuracy: 99.33%

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Parking AI",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
.hero-title {
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00d4ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}
.hero-sub {
    text-align: center;
    color: #888;
    font-size: 1.1rem;
    margin-bottom: 1rem;
}
.result-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 2px solid #00d4ff;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
}
.vehicle-name {
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    margin: 10px 0;
}
.confidence-score {
    font-size: 3rem;
    font-weight: 900;
    color: #00d4ff;
}
.park-card {
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    margin-top: 15px;
}
.zone-name {
    font-size: 2.5rem;
    font-weight: 900;
}
.zone-reason {
    font-size: 0.95rem;
    margin-top: 8px;
}
.unknown-card {
    background: linear-gradient(135deg, #2a1a1a, #3a1a1a);
    border: 2px solid #ff4444;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── PARKING ZONES ─────────────────────────────────────────────
PARKING_ZONES = {
    "suv": {
        "zone":   "Zone A",
        "reason": "Wide bays — extra door space",
        "color":  "#00d4ff"
    },
    "pickup_truck": {
        "zone":   "Zone B",
        "reason": "Extra long bays — truck bed space",
        "color":  "#ff6b6b"
    },
    "sedan": {
        "zone":   "Zone C",
        "reason": "Standard bays — optimal fit",
        "color":  "#00cc88"
    },
    "hatchback": {
        "zone":   "Zone D",
        "reason": "Compact bays — space efficient",
        "color":  "#aa44ff"
    },
    "motorcycle": {
        "zone":   "Zone F",
        "reason": "Dedicated motorcycle bays",
        "color":  "#ff9900"
    },
}

# ── CONFIDENCE THRESHOLD ──────────────────────────────────────
THRESHOLD = 85.0

# ── LOAD MODEL ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = tf.keras.models.load_model('model/car_model.h5')
    with open('model/class_indices.json') as f:
        ci = json.load(f)
    idx_to_class = {v: k for k, v in ci.items()}
    return m, idx_to_class

model, idx_to_class = load_model()

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🅿️ Parking Zone Guide")
    st.markdown("---")
    for key, info in PARKING_ZONES.items():
        st.markdown(f"""
        <div style="background:#1a1a2e;border-radius:10px;
                    padding:10px;margin:5px 0;
                    border-left:4px solid {info['color']}">
            <b style="color:white">
                {info['zone']} — {key.replace('_',' ').title()}
            </b><br>
            <small style="color:#aaa">{info['reason']}</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧠 Model Info")
    st.markdown("""
    - **Architecture:** EfficientNetV2B3
    - **Classes:** 5 vehicle types
    - **Accuracy:** 99.33%
    - **Input Size:** 300 × 300
    - **Framework:** TensorFlow
    """)

# ── HEADER ────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🅿️ Smart Parking AI</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">'
    'Upload a vehicle image — AI detects type and assigns parking zone'
    '</p>',
    unsafe_allow_html=True
)
st.markdown("---")

# ── MAIN LAYOUT ───────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

# ── COLUMN 1 — Upload ─────────────────────────────────────────
with col1:
    st.markdown("### 📷 Upload Vehicle Image")
    uploaded = st.file_uploader(
        "Choose a vehicle image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Vehicle",
                 use_column_width=True)
        w, h = img.size
        st.caption(f"📐 Original Size: {w} × {h} px")

# ── COLUMN 2 — Results ────────────────────────────────────────
with col2:
    st.markdown("### 🤖 Detection Result")

    if uploaded:
        # ── Preprocess ────────────────────────────────────────
        img_resized = np.array(img.resize((300, 300)))  # ✅ 300x300
        img_array   = preprocess_input(img_resized)     # ✅ EfficientNetV2
        img_array   = np.expand_dims(img_array, axis=0)

        # ── Predict ───────────────────────────────────────────
        with st.spinner("🔍 Detecting vehicle..."):
            preds      = model.predict(img_array)[0]
            top_idx    = int(np.argmax(preds))
            vehicle    = idx_to_class[top_idx]
            confidence = preds[top_idx] * 100
            zone_info  = PARKING_ZONES[vehicle]

        # ── Confidence Check ──────────────────────────────────
        if confidence < THRESHOLD:
            st.markdown(f"""
            <div class="unknown-card">
                <div style="font-size:3rem">⚠️</div>
                <div style="font-size:1.8rem;font-weight:800;
                            color:#ff4444;margin:10px 0">
                    Unknown Vehicle
                </div>
                <div style="color:#aaa;font-size:0.95rem">
                    Confidence too low ({confidence:.1f}%)
                    to make a reliable prediction
                </div>
                <div style="color:#ff9900;font-weight:600;
                            margin-top:10px">
                    Please contact parking attendant
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # ── Vehicle Result ─────────────────────────────────
            st.markdown(f"""
            <div class="result-card">
                <div style="color:#aaa;font-size:0.9rem">
                    Detected Vehicle
                </div>
                <div class="vehicle-name">
                    {vehicle.replace('_', ' ').title()}
                </div>
                <div class="confidence-score">
                    {confidence:.1f}%
                </div>
                <div style="color:#aaa;font-size:0.85rem">
                    Confidence Score
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Parking Zone ──────────────────────────────────
            st.markdown(f"""
            <div class="park-card" style="
                 background:linear-gradient(135deg,
                 {zone_info['color']}22,
                 {zone_info['color']}44);
                 border:2px solid {zone_info['color']};">
                <div style="color:#aaa;font-size:0.9rem">
                    Assigned Parking Zone
                </div>
                <div class="zone-name"
                     style="color:{zone_info['color']}">
                    {zone_info['zone']}
                </div>
                <div class="zone-reason" style="color:#ccc">
                    {zone_info['reason']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── All Probabilities ──────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 All Class Probabilities")
        for idx, prob in sorted(
            enumerate(preds),
            key=lambda x: x[1],
            reverse=True
        ):
            cls   = idx_to_class[idx]
            label = cls.replace('_', ' ').title()
            st.progress(
                float(prob),
                text=f"{label}: {prob*100:.1f}%"
            )

    else:
        st.markdown("""
        <div style="border:2px dashed #333;
                    border-radius:15px;
                    padding:80px 30px;
                    text-align:center;
                    color:#555">
            <div style="font-size:3rem">🅿️</div>
            <div style="margin-top:10px;font-size:1.1rem">
                Upload a vehicle image to get started
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("---")
f1, f2, f3 = st.columns(3)
f1.markdown("<center>🧠 EfficientNetV2B3</center>",
            unsafe_allow_html=True)
f2.markdown("<center>🚗 5 Vehicle Classes</center>",
            unsafe_allow_html=True)
f3.markdown("<center>⚡ 99.33% Accuracy</center>",
            unsafe_allow_html=True)