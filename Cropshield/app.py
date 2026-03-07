import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="CropShield", page_icon="🌿", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');
* { font-family: 'Syne', sans-serif; }
.block-container { padding-top: 2rem; max-width: 1100px; }
.hero {
    background: linear-gradient(135deg, #0d2818 0%, #0d1117 50%, #1a0d2e 100%);
    border: 1px solid #1a3a2a; border-radius: 20px;
    padding: 2.5rem; margin-bottom: 2rem;
}
.hero-title {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #4ade80, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0;
}
.stat-row { display: flex; gap: 1.5rem; margin-top: 1.5rem; flex-wrap: wrap; }
.stat {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 0.7rem 1.2rem;
}
.stat-val { font-family: 'DM Mono', monospace; font-size: 1.3rem; font-weight: 700; color: #4ade80; }
.stat-lbl { font-size: 0.72rem; color: #475569; text-transform: uppercase; letter-spacing: 1px; }
.section-title { font-size: 0.75rem; color: #475569; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.8rem; }
.result-card {
    background: linear-gradient(135deg, #0d2818, #0d1117);
    border: 1px solid #1a3a2a; border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
}
.plant-label { font-size: 0.85rem; color: #64748b; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.3rem; }
.condition-healthy { font-size: 1.8rem; font-weight: 800; color: #4ade80; }
.condition-disease  { font-size: 1.8rem; font-weight: 800; color: #f87171; }
.badge-h { background: rgba(74,222,128,0.15); color: #4ade80; border-radius: 20px; padding: 3px 14px; font-size: 0.75rem; font-weight: 700; }
.badge-d { background: rgba(248,113,113,0.15); color: #f87171; border-radius: 20px; padding: 3px 14px; font-size: 0.75rem; font-weight: 700; }
.speed  { font-family: 'DM Mono'; background: rgba(96,165,250,0.1); color: #60a5fa; border-radius: 20px; padding: 3px 12px; font-size: 0.75rem; }
.conf-label { display: flex; justify-content: space-between; font-size: 0.8rem; color: #475569; margin-bottom: 4px; }
.conf-val   { color: #4ade80; font-family: 'DM Mono'; }
.treat-ok  { background: rgba(74,222,128,0.08);  border: 1px solid rgba(74,222,128,0.2);  border-radius: 10px; padding: 1rem; color: #86efac; font-size: 0.9rem; line-height: 1.6; }
.treat-bad { background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.2); border-radius: 10px; padding: 1rem; color: #fca5a5; font-size: 0.9rem; line-height: 1.6; }
.top5row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #1e293b; font-size: 0.85rem; }
.bar-bg  { width: 80px; background: #1e293b; border-radius: 99px; height: 4px; margin: 0 10px; }
.empty-state { background: #0d1117; border: 2px dashed #1e3a2a; border-radius: 16px; padding: 4rem 2rem; text-align: center; }
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000"

# ── Hero ───────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">🌿 CropShield</p>
    <p style="color:#64748b; margin-top:0.5rem">Real-time plant disease detection · EfficientNetB0 + ONNX + FastAPI</p>
    <div class="stat-row">
        <div class="stat"><div class="stat-val">99.64%</div><div class="stat-lbl">Accuracy</div></div>
        <div class="stat"><div class="stat-val">38</div><div class="stat-lbl">Disease Classes</div></div>
        <div class="stat"><div class="stat-val">~21ms</div><div class="stat-lbl">Inference Time</div></div>
        <div class="stat"><div class="stat-val">54K</div><div class="stat-lbl">Training Images</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── API Check ──────────────────────────────────────────────
try:
    ok = requests.get(f"{API_URL}/health", timeout=3).json().get("status") == "healthy"
except:
    ok = False
if not ok:
    st.error("⚠️ API not running! Run: `python main.py`")
    st.stop()

# ── Layout ─────────────────────────────────────────────────
col_l, col_r = st.columns([1, 1.2], gap="large")

with col_l:
    st.markdown('<p class="section-title">Upload Leaf Image</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("leaf", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_container_width=True,
                 caption=f"{uploaded.name} ({image.size[0]}×{image.size[1]}px)")
        st.markdown('<p class="section-title" style="margin-top:1rem">Supported Plants</p>', unsafe_allow_html=True)
        plants = ["🍎 Apple","🍇 Grape","🍅 Tomato","🥔 Potato","🌽 Corn","🍑 Peach","🫐 Blueberry","🍒 Cherry"]
        cols = st.columns(4)
        for i, p in enumerate(plants):
            cols[i%4].markdown(f'<div style="background:#111827;border-radius:8px;padding:4px;text-align:center;font-size:0.72rem;color:#64748b;margin:2px">{p}</div>', unsafe_allow_html=True)

with col_r:
    if uploaded:
        st.markdown('<p class="section-title">Analysis Result</p>', unsafe_allow_html=True)
        with st.spinner("🔬 Analyzing..."):
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            buf.seek(0)
            try:
                resp   = requests.post(f"{API_URL}/predict",
                                       files={"file": ("leaf.jpg", buf, "image/jpeg")}, timeout=30)
                result = resp.json()
                if resp.status_code == 200:
                    healthy   = result["is_healthy"]
                    conf      = result["confidence"]
                    plant     = result["plant"]
                    condition = result["condition"]
                    treatment = result["treatment"]
                    speed     = result["inference_ms"]
                    top5      = result["top5"]

                    # header row
                    h1, h2 = st.columns([2,1])
                    with h1:
                        st.markdown(f'<p class="plant-label">{plant}</p>', unsafe_allow_html=True)
                        css = "condition-healthy" if healthy else "condition-disease"
                        st.markdown(f'<p class="{css}">{condition}</p>', unsafe_allow_html=True)
                    with h2:
                        badge = '<span class="badge-h">HEALTHY</span>' if healthy else '<span class="badge-d">DISEASE DETECTED</span>'
                        st.markdown(f'<div style="text-align:right;margin-top:0.5rem">{badge}<br><br><span class="speed">⚡ {speed}ms</span></div>', unsafe_allow_html=True)

                    # confidence bar
                    st.markdown(f"""
                    <div style="margin-top:1rem">
                        <div class="conf-label"><span>Confidence</span><span class="conf-val">{conf}%</span></div>
                        <div style="background:#1e293b;border-radius:99px;height:8px">
                            <div style="width:{conf}%;height:8px;background:linear-gradient(90deg,#4ade80,#a78bfa);border-radius:99px"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # treatment
                    st.markdown("<br>", unsafe_allow_html=True)
                    tc = "treat-ok" if healthy else "treat-bad"
                    icon = "✅" if healthy else "💊"
                    lbl  = "Status" if healthy else "Recommended Treatment"
                    st.markdown(f'<div class="{tc}"><strong>{icon} {lbl}</strong><br>{treatment}</div>', unsafe_allow_html=True)

                    # top 5
                    st.markdown('<p class="section-title" style="margin-top:1.5rem">Top 5 Predictions</p>', unsafe_allow_html=True)
                    for i, item in enumerate(top5):
                        name  = item["class"].replace("___"," → ").replace("_"," ")
                        c     = item["confidence"]
                        color = "#4ade80" if i == 0 else "#475569"
                        st.markdown(f"""
                        <div class="top5row">
                            <span style="color:{color};flex:1">{name}</span>
                            <div class="bar-bg"><div style="width:{c}%;height:4px;background:{color};border-radius:99px"></div></div>
                            <span style="font-family:DM Mono;font-size:0.8rem;color:{color};min-width:55px;text-align:right">{c}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error(f"Error: {result.get('detail','Unknown')}")
            except Exception as e:
                st.error(f"Connection error: {e}")
    else:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size:3rem">🌿</div>
            <div style="color:#4ade80;font-size:1.1rem;font-weight:700;margin-top:1rem">Upload a leaf image to get started</div>
            <div style="color:#475569;font-size:0.85rem;margin-top:0.5rem">Supports Apple, Tomato, Potato, Corn, Grape and more</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div style="display:flex;justify-content:space-between;color:#334155;font-size:0.8rem"><span>🌿 CropShield — EfficientNetB0 · ONNX · FastAPI · Streamlit</span><span>99.64% accuracy · 38 classes · 21ms inference</span></div>', unsafe_allow_html=True)