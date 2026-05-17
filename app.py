"""
AI Object Detection System - FYP
Uses YOLOv8 pretrained on COCO dataset (80 classes)
Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
import tempfile
import os

# ─────────────────────────────────────────
#  Page config  (must be first Streamlit call)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Object Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0a0a0f;
    --surface:   #12121a;
    --border:    #2a2a3d;
    --accent:    #7c3aed;
    --accent2:   #06b6d4;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --success:   #10b981;
    --warning:   #f59e0b;
}

/* ── Global ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #0a0a0f 0%, #1a0533 50%, #0a0a0f 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(124,58,237,.18) 0%, transparent 60%),
                radial-gradient(ellipse at 30% 50%, rgba(6,182,212,.10) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(2rem, 4vw, 3rem);
    background: linear-gradient(90deg, #a78bfa, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 .5rem 0;
    line-height: 1.1;
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: .85rem;
    color: var(--muted);
    letter-spacing: .05em;
    margin: 0;
}
.badge {
    display: inline-flex; align-items: center; gap: .4rem;
    background: rgba(124,58,237,.2);
    border: 1px solid rgba(124,58,237,.4);
    color: #a78bfa;
    padding: .25rem .75rem;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: .7rem;
    letter-spacing: .08em;
    margin-bottom: 1rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-weight: 600;
    font-size: .95rem;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
    display: flex; align-items: center; gap: .5rem;
}

/* ── Detection result box ── */
.det-box {
    background: rgba(16,185,129,.05);
    border: 1px solid rgba(16,185,129,.2);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: .4rem 0;
    display: flex; align-items: center; justify-content: space-between;
}
.det-label { font-weight: 600; font-size: .9rem; }
.det-conf  { font-family: 'Space Mono', monospace; font-size: .8rem; color: var(--success); }

/* ── Stats ── */
.stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem;
    text-align: center;
}
.stat-val  { font-size: 2rem; font-weight: 800; color: #a78bfa; line-height: 1; }
.stat-desc { font-size: .75rem; color: var(--muted); margin-top: .3rem; letter-spacing: .04em; }

/* ── Streamlit widgets override ── */
.stSlider > div { padding: 0; }
[data-testid="stFileUploader"] {
    background: var(--surface);
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 1rem;
    transition: border-color .2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent); }

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--accent), #9d4edd);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: .8rem;
    letter-spacing: .06em;
    padding: .6rem 1.5rem;
    width: 100%;
    transition: opacity .2s, transform .1s;
}
div[data-testid="stButton"] > button:hover { opacity: .88; transform: translateY(-1px); }
div[data-testid="stButton"] > button:active { transform: translateY(0); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Space Mono', monospace;
    font-size: .8rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--muted);
}

/* ── Progress / spinner ── */
.stProgress > div > div { background: linear-gradient(90deg, var(--accent), var(--accent2)); }
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Info / warning boxes ── */
.stAlert { border-radius: 10px; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] { gap: .5rem; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    font-size: .8rem;
    letter-spacing: .04em;
}
.stTabs [aria-selected="true"] {
    background: rgba(124,58,237,.2) !important;
    border-color: var(--accent) !important;
    color: #a78bfa !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #0ea5e9, var(--accent2)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: .8rem;
    letter-spacing: .06em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  Load YOLOv8 model (cached)
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_name: str = "yolov8n.pt"):
    """Load YOLOv8 model once and cache it."""
    try:
        from ultralytics import YOLO
        model = YOLO(model_name)
        return model, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────
#  COCO class → colour mapping (80 classes)
# ─────────────────────────────────────────
COCO_COLORS = {
    "person": (255, 87, 87), "bicycle": (87, 255, 87), "car": (87, 87, 255),
    "motorcycle": (255, 165, 0), "airplane": (0, 191, 255), "bus": (255, 20, 147),
    "train": (50, 205, 50), "truck": (255, 215, 0), "boat": (0, 255, 255),
    "traffic light": (255, 69, 0), "fire hydrant": (220, 20, 60), "stop sign": (178, 34, 34),
    "parking meter": (218, 165, 32), "bench": (107, 142, 35), "bird": (65, 105, 225),
    "cat": (255, 140, 0), "dog": (148, 0, 211), "horse": (0, 128, 0),
    "sheep": (128, 128, 0), "cow": (0, 128, 128), "elephant": (128, 0, 128),
    "bear": (64, 0, 128), "zebra": (0, 64, 128), "giraffe": (128, 64, 0),
    "backpack": (0, 255, 128), "umbrella": (128, 255, 0), "handbag": (255, 128, 0),
    "tie": (0, 128, 255), "suitcase": (128, 0, 255), "frisbee": (255, 0, 128),
    "skis": (0, 200, 200), "snowboard": (200, 0, 200), "sports ball": (200, 200, 0),
    "kite": (100, 200, 100), "baseball bat": (200, 100, 100), "baseball glove": (100, 100, 200),
    "skateboard": (150, 75, 0), "surfboard": (0, 150, 75), "tennis racket": (75, 0, 150),
    "bottle": (0, 75, 150), "wine glass": (150, 0, 75), "cup": (75, 150, 0),
    "fork": (200, 50, 50), "knife": (50, 200, 50), "spoon": (50, 50, 200),
    "bowl": (200, 150, 0), "banana": (255, 230, 0), "apple": (255, 0, 0),
    "sandwich": (200, 100, 0), "orange": (255, 128, 0), "broccoli": (0, 200, 0),
    "carrot": (255, 100, 0), "hot dog": (200, 50, 100), "pizza": (200, 150, 50),
    "donut": (255, 192, 203), "cake": (255, 20, 147), "chair": (70, 130, 180),
    "couch": (135, 206, 235), "potted plant": (34, 139, 34), "bed": (139, 0, 139),
    "dining table": (184, 134, 11), "toilet": (128, 128, 128), "tv": (0, 0, 205),
    "laptop": (0, 191, 255), "mouse": (100, 149, 237), "remote": (72, 209, 204),
    "keyboard": (30, 144, 255), "cell phone": (0, 255, 127), "microwave": (127, 255, 0),
    "oven": (255, 165, 0), "toaster": (210, 105, 30), "sink": (64, 224, 208),
    "refrigerator": (0, 128, 255), "book": (128, 0, 64), "clock": (255, 215, 0),
    "vase": (148, 103, 189), "scissors": (214, 39, 40), "teddy bear": (197, 176, 213),
    "hair drier": (196, 156, 148), "toothbrush": (23, 190, 207),
}

def get_color(label: str):
    return COCO_COLORS.get(label.lower(), (124, 58, 237))


# ─────────────────────────────────────────
#  Detection logic
# ─────────────────────────────────────────
def run_detection(model, image: np.ndarray, conf_thresh: float, iou_thresh: float):
    """Run YOLOv8 inference and return annotated image + result data."""
    results = model(image, conf=conf_thresh, iou=iou_thresh, verbose=False)
    result  = results[0]

    annotated = image.copy()
    detections = []

    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = model.names[cls]
            color = get_color(label)

            # Draw filled rectangle (semi-transparent) + border
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.12, annotated, 0.88, 0, annotated)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label background
            text     = f"{label}  {conf:.0%}"
            font     = cv2.FONT_HERSHEY_SIMPLEX
            scale    = max(0.45, min(0.7, (x2 - x1) / 180))
            thickness = 1
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            pad = 5
            cv2.rectangle(annotated, (x1, y1 - th - pad * 2), (x1 + tw + pad * 2, y1), color, -1)
            cv2.putText(annotated, text, (x1 + pad, y1 - pad), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

            detections.append({"label": label, "confidence": conf, "bbox": (x1, y1, x2, y2)})

    return annotated, detections


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ─────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:.5rem 0 1.5rem 0;'>
      <div style='font-size:2.5rem;'>🔍</div>
      <div style='font-family:"Space Mono",monospace; font-size:.7rem;
                  letter-spacing:.15em; color:#64748b; margin-top:.3rem;'>
        OBJECT DETECTION
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Detection Settings")
    conf_thresh = st.slider("Confidence Threshold", 0.10, 0.95, 0.40, 0.05,
                            help="Minimum confidence to show a detection")
    iou_thresh  = st.slider("IOU Threshold (NMS)", 0.10, 0.95, 0.45, 0.05,
                            help="Non-Maximum Suppression overlap threshold")

    st.markdown("---")
    st.markdown("### 🧠 Model")
    model_choice = st.selectbox("YOLOv8 Variant",
                                ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
                                index=0,
                                help="n=nano (fastest), s=small, m=medium (most accurate)")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    <div style='font-size:.82rem; color:#64748b; line-height:1.7;'>
    <b>Model:</b> YOLOv8 (Ultralytics)<br>
    <b>Dataset:</b> COCO (80 classes)<br>
    <b>Task:</b> Object Detection<br>
    <b>Project:</b> Final Year Project
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:"Space Mono",monospace; font-size:.65rem;
                color:#374151; text-align:center; letter-spacing:.06em;'>
    BUILT WITH ULTRALYTICS + STREAMLIT
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
#  Load model
# ─────────────────────────────────────────
with st.spinner(f"Loading {model_choice}..."):
    model, model_err = load_model(model_choice)


# ─────────────────────────────────────────
#  Hero header
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="badge">▸ YOLOV8 · COCO DATASET · 80 CLASSES</div>
  <h1 class="hero-title">AI Object Detection System</h1>
  <p class="hero-sub">Upload an image or video — detect everything in real time using YOLOv8</p>
</div>
""", unsafe_allow_html=True)

if model_err:
    st.error(f"❌ Model failed to load: {model_err}\n\nRun `pip install ultralytics` and restart.")
    st.stop()

# ─────────────────────────────────────────
#  Tabs: Image | Video | About
# ─────────────────────────────────────────
tab_img, tab_vid, tab_about = st.tabs(["🖼️  Image Detection", "🎬  Video Detection", "📘  About & Classes"])


# ════════════════════════════════════════
#  TAB 1 — Image Detection
# ════════════════════════════════════════
with tab_img:
    uploaded = st.file_uploader(
        "Drop an image here or click to browse",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="img_upload"
    )

    if uploaded:
        orig_pil = Image.open(uploaded)
        w, h     = orig_pil.size

        # ── Layout: original | result ──────────────
        col_orig, col_result = st.columns(2, gap="medium")

        with col_orig:
            st.markdown('<div class="card-title">📥 Original Image</div>', unsafe_allow_html=True)
            st.image(orig_pil, use_container_width=True)
            st.markdown(
                f'<div style="font-family:\'Space Mono\',monospace; font-size:.75rem; '
                f'color:#64748b; margin-top:.5rem;">{w} × {h} px · {uploaded.type}</div>',
                unsafe_allow_html=True
            )

        # ── Run detection ──────────────────────────
        with col_result:
            st.markdown('<div class="card-title">🎯 Detection Result</div>', unsafe_allow_html=True)
            with st.spinner("Running YOLOv8 inference..."):
                t0      = time.perf_counter()
                cv_img  = pil_to_cv2(orig_pil)
                ann_cv, detections = run_detection(model, cv_img, conf_thresh, iou_thresh)
                elapsed = (time.perf_counter() - t0) * 1000
                ann_pil = cv2_to_pil(ann_cv)

            st.image(ann_pil, use_container_width=True)

            # Download button
            st.download_button(
                "⬇️  Download Result",
                data=pil_to_bytes(ann_pil),
                file_name="detection_result.png",
                mime="image/png",
                use_container_width=True
            )

        # ── Stats row ─────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        unique_classes = list({d["label"] for d in detections})
        avg_conf = (sum(d["confidence"] for d in detections) / len(detections) * 100) if detections else 0

        st.markdown(f"""
        <div class="stat-grid">
          <div class="stat-card">
            <div class="stat-val">{len(detections)}</div>
            <div class="stat-desc">OBJECTS DETECTED</div>
          </div>
          <div class="stat-card">
            <div class="stat-val">{len(unique_classes)}</div>
            <div class="stat-desc">UNIQUE CLASSES</div>
          </div>
          <div class="stat-card">
            <div class="stat-val">{elapsed:.0f}<span style="font-size:1rem">ms</span></div>
            <div class="stat-desc">INFERENCE TIME</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Detection list ─────────────────────────
        if detections:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="card-title">📋 Detected Objects</div>', unsafe_allow_html=True)

            # Sort by confidence descending
            for d in sorted(detections, key=lambda x: -x["confidence"]):
                x1, y1, x2, y2 = d["bbox"]
                color = get_color(d["label"])
                hex_c = "#{:02x}{:02x}{:02x}".format(*color[::-1])  # BGR→RGB
                st.markdown(f"""
                <div class="det-box">
                  <span class="det-label">
                    <span style="display:inline-block;width:10px;height:10px;
                                 border-radius:2px;background:{hex_c};
                                 margin-right:8px;vertical-align:middle;"></span>
                    {d['label'].upper()}
                  </span>
                  <div style="text-align:right;">
                    <span class="det-conf">{d['confidence']:.1%}</span><br>
                    <span style="font-family:'Space Mono',monospace;font-size:.65rem;color:#374151;">
                      [{x1},{y1}] → [{x2},{y2}]
                    </span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No objects detected above the current confidence threshold. Try lowering it in the sidebar.")

    else:
        # Empty state
        st.markdown("""
        <div style="text-align:center; padding:4rem 2rem; border:2px dashed #2a2a3d;
                    border-radius:16px; margin-top:1rem;">
          <div style="font-size:4rem; margin-bottom:1rem;">📸</div>
          <div style="font-size:1.1rem; font-weight:600; margin-bottom:.5rem;">No image uploaded yet</div>
          <div style="color:#64748b; font-size:.9rem;">
            Upload a JPG, PNG, or WebP image above to start detecting objects
          </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════
#  TAB 2 — Video Detection
# ════════════════════════════════════════
with tab_vid:
    st.markdown("""
    <div style="background:rgba(124,58,237,.07);border:1px solid rgba(124,58,237,.2);
                border-radius:10px;padding:1rem 1.25rem;margin-bottom:1rem;font-size:.88rem;color:#94a3b8;">
    ⚠️ <b>Note:</b> Video processing runs frame-by-frame. Large files may take a while.
    For live webcam, use the button below (requires camera permission).
    </div>
    """, unsafe_allow_html=True)

    vid_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"], key="vid_upload")

    if vid_file:
        # Save to temp file so OpenCV can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(vid_file.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25
        cap.release()

        st.info(f"📹 Video loaded: **{total_frames} frames** at **{fps:.1f} FPS**")

        frame_skip = st.slider("Process every Nth frame (speed vs quality)", 1, 10, 3)
        max_frames = st.slider("Max frames to process", 10, min(300, total_frames), min(60, total_frames))

        if st.button("▶  Run Detection on Video"):
            cap = cv2.VideoCapture(tmp_path)
            output_frames = []
            prog = st.progress(0)
            status_txt = st.empty()
            processed = 0
            frame_idx = 0

            while cap.isOpened() and processed < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_skip == 0:
                    ann_frame, _ = run_detection(model, frame, conf_thresh, iou_thresh)
                    output_frames.append(cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB))
                    processed += 1
                    prog.progress(processed / max_frames)
                    status_txt.text(f"Processing frame {processed}/{max_frames}…")
                frame_idx += 1

            cap.release()
            os.unlink(tmp_path)
            prog.empty()
            status_txt.empty()

            st.success(f"✅ Processed {len(output_frames)} frames!")

            # Show as a slider of frames
            if output_frames:
                frame_sel = st.slider("Browse result frames", 0, len(output_frames) - 1, 0)
                st.image(output_frames[frame_sel], use_container_width=True,
                         caption=f"Frame {frame_sel + 1} of {len(output_frames)}")
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 2rem;border:2px dashed #2a2a3d;
                    border-radius:16px;margin-top:1rem;">
          <div style="font-size:3.5rem;margin-bottom:1rem;">🎬</div>
          <div style="font-size:1.1rem;font-weight:600;margin-bottom:.5rem;">No video uploaded</div>
          <div style="color:#64748b;font-size:.9rem;">Upload MP4, AVI, or MOV to run frame-by-frame detection</div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════
#  TAB 3 — About & Classes
# ════════════════════════════════════════
with tab_about:
    col_a, col_b = st.columns([1.2, 1], gap="large")

    with col_a:
        st.markdown("""
        <div class="card">
          <div class="card-title">🎓 Project Overview</div>
          <p style="line-height:1.8; font-size:.9rem; color:#cbd5e1;">
            This Final Year Project demonstrates real-time object detection using
            <b>YOLOv8</b> (You Only Look Once v8) by Ultralytics, pretrained on the
            <b>COCO dataset</b> (Common Objects in Context) with <b>80 object classes</b>.
          </p>
          <p style="line-height:1.8; font-size:.9rem; color:#cbd5e1;">
            The application provides a complete end-to-end pipeline: image/video upload →
            neural network inference → annotated output with bounding boxes, class labels,
            and confidence scores.
          </p>
        </div>

        <div class="card">
          <div class="card-title">⚡ Tech Stack</div>
          <div style="font-size:.88rem; line-height:2;">
            <span style="color:#a78bfa;">▸</span> <b>Model:</b> YOLOv8n / YOLOv8s / YOLOv8m<br>
            <span style="color:#a78bfa;">▸</span> <b>Framework:</b> Ultralytics + PyTorch<br>
            <span style="color:#a78bfa;">▸</span> <b>UI:</b> Streamlit<br>
            <span style="color:#a78bfa;">▸</span> <b>Image Processing:</b> OpenCV + Pillow<br>
            <span style="color:#a78bfa;">▸</span> <b>Language:</b> Python 3.9+
          </div>
        </div>

        <div class="card">
          <div class="card-title">📊 Model Comparison</div>
          <table style="width:100%;font-size:.82rem;border-collapse:collapse;">
            <tr style="color:#64748b;font-family:'Space Mono',monospace;font-size:.72rem;">
              <th style="text-align:left;padding:.4rem;">VARIANT</th>
              <th>PARAMS</th><th>SPEED</th><th>MAP50</th>
            </tr>
            <tr style="border-top:1px solid #2a2a3d;">
              <td style="padding:.5rem 0;color:#a78bfa;font-weight:600;">YOLOv8n</td>
              <td style="text-align:center;">3.2M</td>
              <td style="text-align:center;color:#10b981;">⚡ Fastest</td>
              <td style="text-align:center;">37.3</td>
            </tr>
            <tr style="border-top:1px solid #2a2a3d;">
              <td style="padding:.5rem 0;color:#a78bfa;font-weight:600;">YOLOv8s</td>
              <td style="text-align:center;">11.2M</td>
              <td style="text-align:center;color:#f59e0b;">🔥 Fast</td>
              <td style="text-align:center;">44.9</td>
            </tr>
            <tr style="border-top:1px solid #2a2a3d;">
              <td style="padding:.5rem 0;color:#a78bfa;font-weight:600;">YOLOv8m</td>
              <td style="text-align:center;">25.9M</td>
              <td style="text-align:center;color:#ef4444;">🐢 Moderate</td>
              <td style="text-align:center;">50.2</td>
            </tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card-title">📦 All 80 COCO Classes</div>', unsafe_allow_html=True)
        coco_classes = [
            "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
            "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
            "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
            "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
            "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
            "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
            "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
            "chair","couch","potted plant","bed","dining table","toilet","tv","laptop",
            "mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
            "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
        ]
        cols_cc = st.columns(2)
        half = len(coco_classes) // 2
        for i, cls in enumerate(coco_classes):
            color = get_color(cls)
            hex_c = "#{:02x}{:02x}{:02x}".format(*color[::-1])
            with cols_cc[0 if i < half else 1]:
                st.markdown(
                    f'<div style="font-size:.78rem;padding:.2rem .4rem;display:flex;align-items:center;gap:.5rem;">'
                    f'<span style="width:8px;height:8px;border-radius:2px;background:{hex_c};'
                    f'flex-shrink:0;display:inline-block;"></span>{cls}</div>',
                    unsafe_allow_html=True
                )