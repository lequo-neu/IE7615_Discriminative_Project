"""
Streamlit Application for CNN Attendance System
Professional UI/UX Design - Optimized for 1920x1080
Author: Kevin (Lead Developer) - Group 8
"""

import streamlit as st
import numpy as np
import json
from pathlib import Path
import pandas as pd
from PIL import Image
import time
import os
import random

# TensorFlow optimizations - MUST be before importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="CNN Attendance System | Group 8",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS - Optimized for 1920x1080
st.markdown("""
<style>
    /* ===== GLOBAL STYLES - 1920x1080 OPTIMIZED ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    .main .block-container {
        max-width: 1800px;
        padding: 0.5rem 2rem;
        margin: 0 auto;
    }
    
    .main {
        overflow: hidden;
    }
    
    /* Hide Streamlit UI */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* ===== TYPOGRAPHY ===== */
    h1, h2, h3, h4, h5, h6, p, span, div {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* ===== COMPACT HEADER ===== */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.8rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white !important;
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.85) !important;
        font-size: 0.85rem !important;
        margin: 0.3rem 0 0 0 !important;
    }
    
    /* ===== COMPACT TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #f8fafc;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        padding: 8px 20px !important;
        border-radius: 6px !important;
        color: #64748b !important;
        background: transparent !important;
        border: none !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* ===== COMPACT SECTION TITLES ===== */
    .section-title {
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748b !important;
        margin-bottom: 0.6rem !important;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* ===== COMPACT RESULT CARDS ===== */
    .result-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(34, 197, 94, 0.2);
    }
    
    .result-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #16a34a;
        margin-bottom: 0.3rem;
    }
    
    .result-value {
        font-size: 1.4rem !important;
        font-weight: 800 !important;
        color: #15803d !important;
        line-height: 1.2;
    }
    
    .confidence-badge {
        display: inline-block;
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.3rem 1rem;
        border-radius: 50px;
        margin-top: 0.3rem;
    }
    
    /* ===== COMPACT METRICS ===== */
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.6rem;
        text-align: center;
    }
    
    .metric-card:hover {
        border-color: #667eea;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15);
    }
    
    .metric-label {
        font-size: 0.6rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94a3b8;
        margin-bottom: 0.15rem;
    }
    
    .metric-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    /* ===== COMPACT TOP 3 ===== */
    .top3-item {
        display: flex;
        align-items: center;
        padding: 8px 12px;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        margin-bottom: 6px;
    }
    
    .top3-item.gold {
        background: linear-gradient(135deg, #fefce8 0%, #fef9c3 100%);
        border-color: #eab308;
    }
    
    .top3-rank {
        font-size: 1.2rem;
        margin-right: 10px;
    }
    
    .top3-name {
        flex: 1;
        font-weight: 600;
        color: #1e293b;
        font-size: 0.85rem;
    }
    
    .top3-score {
        font-weight: 700;
        color: #667eea;
        font-size: 0.95rem;
    }
    
    .top3-bar {
        width: 100%;
        height: 4px;
        background: #e2e8f0;
        border-radius: 2px;
        margin-top: 4px;
        overflow: hidden;
    }
    
    .top3-bar-fill {
        height: 100%;
        border-radius: 2px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* ===== DETECTION ITEMS - LARGER TEXT ===== */
    .detection-item {
        display: flex;
        align-items: center;
        padding: 10px 14px;
        background: white;
        border-radius: 10px;
        margin-bottom: 6px;
        border: 1px solid #e2e8f0;
    }
    
    .detection-item:hover {
        transform: translateX(2px);
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    
    .detection-index {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 0.9rem;
        margin-right: 12px;
        flex-shrink: 0;
    }
    
    .detection-info {
        flex: 1;
    }
    
    .detection-id {
        font-weight: 700;
        color: #1e293b;
        font-size: 1rem;
    }
    
    .detection-name {
        color: #64748b;
        font-size: 0.9rem;
    }
    
    .detection-conf {
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    /* ===== FILE UPLOADER COMPACT ===== */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #cbd5e1 !important;
        border-radius: 10px !important;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
        padding: 0.6rem !important;
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: #667eea !important;
    }
    
    /* ===== COMPACT SELECT BOX ===== */
    div[data-testid="stSelectbox"] label {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        text-transform: uppercase;
    }
    
    /* ===== IMAGE STYLING ===== */
    .stImage > img {
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }
    
    /* ===== FOOTER - LARGER TEXT ===== */
    .custom-footer {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1rem 2rem;
        border-radius: 10px;
        margin-top: 1rem;
        text-align: center;
    }
    
    .custom-footer p {
        color: rgba(255,255,255,0.9) !important;
        margin: 0 !important;
        font-size: 1rem;
    }
    
    /* ===== CENTER PREVIEW IMAGE ===== */
    .preview-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .custom-footer strong {
        color: #a5b4fc;
    }
    
    /* ===== STREAMLIT OVERRIDES ===== */
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.65rem !important;
        text-transform: uppercase;
    }
    
    .stProgress > div > div {
        height: 6px;
        border-radius: 3px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    h1, h2, h3 {
        padding: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Reduce spacing */
    .element-container {
        margin-bottom: 0.3rem;
    }
    
    .stMarkdown p {
        margin-bottom: 0.3rem;
    }
    
    /* ===== RANDOM BUTTON STYLING ===== */
    .random-btn-container {
        display: flex;
        justify-content: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_CLASS_PATH = PROJECT_ROOT / "models" / "classification"
MODELS_DETECT_PATH = PROJECT_ROOT / "models" / "detection"
RESULTS_PATH = PROJECT_ROOT / "results"
DATA_PATH = PROJECT_ROOT / "data"

@st.cache_data
def load_class_mapping():
    with open(PROJECT_ROOT / "data" / "class_mapping.json", 'r') as f:
        return json.load(f)

@st.cache_data
def load_object_names():
    names_file = PROJECT_ROOT / "data" / "object_names.json"
    if names_file.exists():
        with open(names_file, 'r') as f:
            data = json.load(f)
            return {k: v for k, v in data.items() if not k.startswith('_')}
    return {}

class_info = load_class_mapping()
object_names_map = load_object_names()

def get_display_name(obj_id):
    return object_names_map.get(obj_id, obj_id)

class_names = [get_display_name(class_info['idx_to_class'][str(i)]) 
               for i in range(class_info['num_classes'])]

# Lazy model loading
def get_classifier(model_name):
    if 'loaded_classifier' not in st.session_state:
        st.session_state.loaded_classifier = {}
    
    if model_name not in st.session_state.loaded_classifier:
        with st.spinner(f"🔄 Loading {model_name}..."):
            from tensorflow import keras
            model_path = MODELS_CLASS_PATH / f"{model_name}_best.keras"
            st.session_state.loaded_classifier[model_name] = keras.models.load_model(
                model_path, compile=False
            )
    
    return st.session_state.loaded_classifier[model_name]

def get_detector(model_variant):
    if 'loaded_detector' not in st.session_state:
        st.session_state.loaded_detector = {}
    
    if model_variant not in st.session_state.loaded_detector:
        with st.spinner(f"🔄 Loading YOLOv8{model_variant}..."):
            from ultralytics import YOLO
            model_path = MODELS_DETECT_PATH / f"yolov8{model_variant}_best.pt"
            st.session_state.loaded_detector[model_variant] = YOLO(model_path)
    
    return st.session_state.loaded_detector[model_variant]


def get_random_test_images(num_images=5):
    """Get random test images from multi_objects_random dataset"""
    random_images_path = DATA_PATH / "multi_objects_random" / "images" / "all_generated"
    if random_images_path.exists():
        all_images = list(random_images_path.glob("*.jpg"))
        if len(all_images) >= num_images:
            return random.sample(all_images, num_images)
        return all_images
    return []


# ===== HEADER =====
st.markdown("""
<div class="main-header">
    <h1>🎓 CNN ATTENDANCE SYSTEM</h1>
    <p>Discriminative Deep Learning for Object Identification | IE 7615 Project</p>
</div>
""", unsafe_allow_html=True)

# ===== TABS =====
tab1, tab2, tab3, tab4 = st.tabs(["📷 SINGLE OBJECT", "🎯 MULTI-OBJECT", "🌐 SPATIAL MULTI-OBJECT", "📊 PERFORMANCE"])


# ==================== TAB 1: CLASSIFICATION ====================
with tab1:
    col_config, col_results = st.columns([1, 2], gap="medium")
    
    with col_config:
        st.markdown('<p class="section-title">⚙️ Configuration</p>', unsafe_allow_html=True)
        
        classifier_model = st.selectbox(
            "MODEL",
            ['mobilenet', 'advanced_cnn', 'custom_cnn', 'efficientnet', 'resnet50'],
            index=0,
            format_func=lambda x: {
                'custom_cnn': '🧠 Custom CNN',
                'advanced_cnn': '🚀 Advanced CNN',
                'resnet50': '🔷 ResNet50',
                'efficientnet': '⚡ EfficientNet',
                'mobilenet': '📱 MobileNetV2'
            }[x]
        )
        
        st.markdown('<p class="section-title">📤 Upload</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("file", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            display_img = image.resize((200, 200), Image.Resampling.LANCZOS)
            st.markdown('<p class="section-title">🖼️ Preview</p>', unsafe_allow_html=True)
            # Center image using columns
            _, center_col, _ = st.columns([0.1, 0.8, 0.1])
            with center_col:
                st.image(display_img, width=200)
    
    with col_results:
        if uploaded_file is not None:
            import cv2
            
            with st.spinner("🔍 Analyzing..."):
                img_array = np.array(image.resize((224, 224)))
                if len(img_array.shape) == 2:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                img_array = img_array.astype('float32') / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                model = get_classifier(classifier_model)
                start = time.time()
                predictions = model.predict(img_array, verbose=0)
                inference_time = (time.time() - start) * 1000
                
                top_3_idx = np.argsort(predictions[0])[-3:][::-1]
                top_3_probs = predictions[0][top_3_idx]
                top_3_classes = [class_names[i] for i in top_3_idx]
            
            # Result Card
            st.markdown('<p class="section-title">🎯 Result</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">IDENTIFIED OBJECT</div>
                <div class="result-value">{top_3_classes[0]}</div>
                <div class="confidence-badge">{top_3_probs[0]*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            st.markdown('<p class="section-title">📈 Metrics</p>', unsafe_allow_html=True)
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Time</div><div class="metric-value">{inference_time:.1f}ms</div></div>', unsafe_allow_html=True)
            with col_m2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">FPS</div><div class="metric-value">{1000/inference_time:.1f}</div></div>', unsafe_allow_html=True)
            with col_m3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Model</div><div class="metric-value" style="font-size:0.8rem;">{classifier_model.upper()}</div></div>', unsafe_allow_html=True)
            
            # Top 3
            st.markdown('<p class="section-title">🏆 Top 3</p>', unsafe_allow_html=True)
            medals = ['🥇', '🥈', '🥉']
            for i, (cls, prob) in enumerate(zip(top_3_classes, top_3_probs)):
                extra = ' gold' if i == 0 else ''
                st.markdown(f"""
                <div class="top3-item{extra}">
                    <span class="top3-rank">{medals[i]}</span>
                    <div style="flex:1;"><div class="top3-name">{cls}</div>
                    <div class="top3-bar"><div class="top3-bar-fill" style="width:{prob*100}%;"></div></div></div>
                    <span class="top3-score">{prob*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center; padding:3rem; background:#f8fafc; border-radius:12px; border:2px dashed #cbd5e1;">
                <div style="font-size:3rem;">📷</div>
                <div style="font-size:1rem; font-weight:600; color:#475569;">Upload an Image</div>
            </div>
            """, unsafe_allow_html=True)


# ==================== TAB 2: DETECTION (GRID LAYOUT) ====================
with tab2:
    # 2-column layout: Left (Config + Preview) | Right (Image + Results)
    col_left, col_right = st.columns([1, 2.5], gap="medium")
    
    with col_left:
        st.markdown('<p class="section-title">⚙️ Configuration</p>', unsafe_allow_html=True)
        
        yolo_variant = st.selectbox(
            "MODEL",
            ['n', 's', 'm'],
            index=1,
            format_func=lambda x: {'n': '🚀 YOLOv8n', 's': '⚖️ YOLOv8s', 'm': '🎯 YOLOv8m'}[x]
        )
        
        st.markdown('<p class="section-title">📤 Upload</p>', unsafe_allow_html=True)
        uploaded_file_det = st.file_uploader("file", type=['jpg', 'jpeg', 'png'], key='det', label_visibility="collapsed")
        
        if uploaded_file_det is not None:
            image = Image.open(uploaded_file_det)
            display_img = image.resize((200, 200), Image.Resampling.LANCZOS)
            st.markdown('<p class="section-title">🖼️ Preview</p>', unsafe_allow_html=True)
            # Center image using columns
            _, center_col, _ = st.columns([0.1, 0.8, 0.1])
            with center_col:
                st.image(display_img, width=200)
    
    with col_right:
        if uploaded_file_det is not None:
            import cv2
            from PIL import ImageDraw, ImageFont
            
            with st.spinner("🔍 Detecting..."):
                temp_path = Path("temp_detect.jpg")
                image.save(temp_path)
                
                model = get_detector(yolo_variant)
                start = time.time()
                results = model(temp_path, verbose=False)[0]
                inference_time = (time.time() - start) * 1000
                
                # Draw annotations
                img_pil = image.copy().convert('RGB')
                draw = ImageDraw.Draw(img_pil)
                
                box_colors = [
                    (46, 204, 113), (155, 89, 182), (52, 152, 219),
                    (241, 196, 15), (231, 76, 60), (26, 188, 156),
                    (230, 126, 34), (236, 72, 153), (20, 184, 166), (139, 92, 246)
                ]
                hex_colors = [
                    '#2ecc71', '#9b59b6', '#3498db', '#f1c40f', '#e74c3c',
                    '#1abc9c', '#e67e22', '#ec4899', '#14b8a6', '#8b5cf6'
                ]
                
                detections_list = []
                
                for idx, box in enumerate(results.boxes):
                    cls_idx = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    obj_id = class_info['idx_to_class'][str(cls_idx)]
                    full_name = object_names_map.get(obj_id, obj_id)
                    short_name = full_name.split('-', 1)[1] if '-' in full_name else full_name
                    
                    color = box_colors[idx % len(box_colors)]
                    hex_color = hex_colors[idx % len(hex_colors)]
                    
                    box_w, box_h = x2 - x1, y2 - y1
                    thickness = max(3, int(min(box_w, box_h) / 60))
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
                    
                    try:
                        font_size = max(14, int(min(box_w, box_h) / 10))
                        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
                    except:
                        try:
                            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                        except:
                            font = ImageFont.load_default()
                    
                    index_label = str(idx + 1)
                    text_bbox = draw.textbbox((0, 0), index_label, font=font)
                    text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                    
                    circle_r = max(text_w, text_h) // 2 + 6
                    circle_x, circle_y = x1 + 3 + circle_r, y1 + 3 + circle_r
                    draw.ellipse([circle_x - circle_r, circle_y - circle_r,
                                  circle_x + circle_r, circle_y + circle_r], fill=color)
                    draw.text((circle_x - text_w//2, circle_y - text_h//2), index_label, fill=(255, 255, 255), font=font)
                    
                    # Store center position for sorting by grid position
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    
                    detections_list.append({
                        'idx': idx + 1, 'obj_id': obj_id, 'name': short_name,
                        'conf': conf, 'color': hex_color,
                        'cx': cx, 'cy': cy
                    })
                
                img_annotated = np.array(img_pil)
                temp_path.unlink()
            
            if len(results.boxes) > 0:
                # Two columns: Image (large) | Results
                col_img, col_info = st.columns([1.6, 1], gap="medium")
                
                with col_img:
                    st.markdown('<p class="section-title">🎯 Detection Result</p>', unsafe_allow_html=True)
                    result_img = Image.fromarray(img_annotated)
                    st.image(result_img, use_container_width=True)
                
                with col_info:
                    # Metrics
                    st.markdown('<p class="section-title">📈 Performance</p>', unsafe_allow_html=True)
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Objects</div><div class="metric-value" style="color:#2563eb;">{len(results.boxes)}</div></div>', unsafe_allow_html=True)
                    with col_m2:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Time</div><div class="metric-value">{inference_time:.0f}ms</div></div>', unsafe_allow_html=True)
                    with col_m3:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">FPS</div><div class="metric-value">{1000/inference_time:.1f}</div></div>', unsafe_allow_html=True)
                    
                    # Determine grid layout based on number of objects
                    num_objects = len(detections_list)
                    if num_objects == 4:
                        grid_cols = 2  # 2x2
                    elif num_objects == 6:
                        grid_cols = 3  # 2x3
                    elif num_objects == 9:
                        grid_cols = 3  # 3x3
                    else:
                        grid_cols = 3  # Default 3 columns
                    
                    # Sort detections by position (top-left to bottom-right)
                    # Using center y first (row), then center x (column)
                    sorted_detections = sorted(detections_list, 
                                               key=lambda d: (d.get('cy', 0), d.get('cx', 0)))
                    
                    st.markdown('<p class="section-title">📋 Detected Objects (Grid View)</p>', unsafe_allow_html=True)
                    
                    # Create grid display
                    for row_start in range(0, len(sorted_detections), grid_cols):
                        row_items = sorted_detections[row_start:row_start + grid_cols]
                        cols = st.columns(grid_cols)
                        for col_idx, det in enumerate(row_items):
                            with cols[col_idx]:
                                st.markdown(f"""
                                <div style="background:white; border:2px solid {det['color']}; border-radius:10px; padding:8px; text-align:center; margin-bottom:8px;">
                                    <div style="background:{det['color']}; color:white; font-weight:700; font-size:1.2rem; width:32px; height:32px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; margin-bottom:4px;">{det['idx']}</div>
                                    <div style="font-weight:700; font-size:0.9rem; color:#1e293b;">{det['obj_id']}</div>
                                    <div style="font-size:0.8rem; color:#64748b;">{det['name'][:15]}</div>
                                    <div style="font-weight:700; font-size:1.1rem; color:{det['color']};">{det['conf']*100:.0f}%</div>
                                </div>
                                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ No objects detected")
                st.image(image, use_container_width=True)
        else:
            st.markdown("""
            <div style="text-align:center; padding:4rem; background:#f8fafc; border-radius:12px; border:2px dashed #cbd5e1;">
                <div style="font-size:4rem;">🎯</div>
                <div style="font-size:1.2rem; font-weight:600; color:#475569;">Upload an Image to Start Detection</div>
                <div style="color:#94a3b8; margin-top:0.5rem;">Supported: JPG, JPEG, PNG</div>
            </div>
            """, unsafe_allow_html=True)


# ==================== TAB 3: SPATIAL MULTI-OBJECT (RANDOM LAYOUT) ====================
with tab3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 1px solid #f59e0b; border-radius: 10px; padding: 0.8rem 1.2rem; margin-bottom: 1rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.2rem;">🌐</span>
            <span style="font-weight: 600; color: #92400e;">Spatial Multi-Object Detection</span>
        </div>
        <div style="font-size: 0.85rem; color: #78350f; margin-top: 0.3rem;">
            Detects objects with random spatial positions (1-9 objects per image, no grid bias)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_left_spatial, col_right_spatial = st.columns([1, 2.5], gap="medium")
    
    with col_left_spatial:
        st.markdown('<p class="section-title">⚙️ Configuration</p>', unsafe_allow_html=True)
        
        yolo_variant_spatial = st.selectbox(
            "MODEL",
            ['n', 's', 'm'],
            index=1,
            format_func=lambda x: {'n': '🚀 YOLOv8n', 's': '⚖️ YOLOv8s', 'm': '🎯 YOLOv8m'}[x],
            key='spatial_model'
        )
        
        st.markdown('<p class="section-title">📤 Image Source</p>', unsafe_allow_html=True)
        
        # Random image button
        if st.button("🎲 Load Random Test Image", use_container_width=True, key='random_btn'):
            random_images = get_random_test_images(1)
            if random_images:
                st.session_state.spatial_random_image = str(random_images[0])
        
        st.markdown('<div style="text-align:center; color:#94a3b8; font-size:0.8rem; margin:0.5rem 0;">— or —</div>', unsafe_allow_html=True)
        
        uploaded_file_spatial = st.file_uploader("file", type=['jpg', 'jpeg', 'png'], key='spatial', label_visibility="collapsed")
        
        # Determine image source
        spatial_image = None
        image_source = None
        
        if uploaded_file_spatial is not None:
            spatial_image = Image.open(uploaded_file_spatial)
            image_source = "upload"
        elif 'spatial_random_image' in st.session_state:
            spatial_image = Image.open(st.session_state.spatial_random_image)
            image_source = "random"
        
        if spatial_image is not None:
            display_img = spatial_image.resize((200, 200), Image.Resampling.LANCZOS)
            st.markdown('<p class="section-title">🖼️ Preview</p>', unsafe_allow_html=True)
            _, center_col, _ = st.columns([0.1, 0.8, 0.1])
            with center_col:
                st.image(display_img, width=200)
            
            if image_source == "random":
                filename = Path(st.session_state.spatial_random_image).name
                # Parse object count from filename
                obj_count = filename.split('_obj')[1].split('.')[0] if '_obj' in filename else '?'
                st.markdown(f"""
                <div style="text-align:center; margin-top:0.5rem;">
                    <div style="font-size:0.75rem; color:#64748b;">{filename}</div>
                    <div style="font-size:0.8rem; color:#059669; font-weight:600;">Contains {obj_count} objects</div>
                </div>
                """, unsafe_allow_html=True)
    
    with col_right_spatial:
        if spatial_image is not None:
            import cv2
            from PIL import ImageDraw, ImageFont
            
            with st.spinner("🔍 Detecting objects in spatial layout..."):
                temp_path = Path("temp_spatial_detect.jpg")
                spatial_image.save(temp_path)
                
                model = get_detector(yolo_variant_spatial)
                start = time.time()
                results = model(temp_path, verbose=False)[0]
                inference_time = (time.time() - start) * 1000
                
                # Draw annotations
                img_pil = spatial_image.copy().convert('RGB')
                draw = ImageDraw.Draw(img_pil)
                
                box_colors = [
                    (46, 204, 113), (155, 89, 182), (52, 152, 219),
                    (241, 196, 15), (231, 76, 60), (26, 188, 156),
                    (230, 126, 34), (236, 72, 153), (20, 184, 166), (139, 92, 246)
                ]
                hex_colors = [
                    '#2ecc71', '#9b59b6', '#3498db', '#f1c40f', '#e74c3c',
                    '#1abc9c', '#e67e22', '#ec4899', '#14b8a6', '#8b5cf6'
                ]
                
                detections_list = []
                
                for idx, box in enumerate(results.boxes):
                    cls_idx = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    obj_id = class_info['idx_to_class'][str(cls_idx)]
                    full_name = object_names_map.get(obj_id, obj_id)
                    short_name = full_name.split('-', 1)[1] if '-' in full_name else full_name
                    
                    color = box_colors[idx % len(box_colors)]
                    hex_color = hex_colors[idx % len(hex_colors)]
                    
                    box_w, box_h = x2 - x1, y2 - y1
                    thickness = max(3, int(min(box_w, box_h) / 60))
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
                    
                    try:
                        font_size = max(14, int(min(box_w, box_h) / 10))
                        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
                    except:
                        try:
                            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                        except:
                            font = ImageFont.load_default()
                    
                    index_label = str(idx + 1)
                    text_bbox = draw.textbbox((0, 0), index_label, font=font)
                    text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                    
                    circle_r = max(text_w, text_h) // 2 + 6
                    circle_x, circle_y = x1 + 3 + circle_r, y1 + 3 + circle_r
                    draw.ellipse([circle_x - circle_r, circle_y - circle_r,
                                  circle_x + circle_r, circle_y + circle_r], fill=color)
                    draw.text((circle_x - text_w//2, circle_y - text_h//2), index_label, fill=(255, 255, 255), font=font)
                    
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    
                    detections_list.append({
                        'idx': idx + 1, 'obj_id': obj_id, 'name': short_name,
                        'conf': conf, 'color': hex_color,
                        'cx': cx, 'cy': cy, 'area': box_w * box_h
                    })
                
                img_annotated = np.array(img_pil)
                temp_path.unlink()
            
            if len(results.boxes) > 0:
                col_img, col_info = st.columns([1.6, 1], gap="medium")
                
                with col_img:
                    st.markdown('<p class="section-title">🌐 Spatial Detection Result</p>', unsafe_allow_html=True)
                    result_img = Image.fromarray(img_annotated)
                    st.image(result_img, use_container_width=True)
                
                with col_info:
                    # Metrics
                    st.markdown('<p class="section-title">📈 Performance</p>', unsafe_allow_html=True)
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Objects</div><div class="metric-value" style="color:#2563eb;">{len(results.boxes)}</div></div>', unsafe_allow_html=True)
                    with col_m2:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Time</div><div class="metric-value">{inference_time:.0f}ms</div></div>', unsafe_allow_html=True)
                    with col_m3:
                        avg_conf = np.mean([d['conf'] for d in detections_list]) * 100
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Conf</div><div class="metric-value">{avg_conf:.0f}%</div></div>', unsafe_allow_html=True)
                    
                    st.markdown('<p class="section-title">📋 Detected Objects (by Confidence)</p>', unsafe_allow_html=True)
                    
                    # Sort by confidence (highest first)
                    sorted_by_conf = sorted(detections_list, key=lambda d: d['conf'], reverse=True)
                    
                    # Display as scrollable list
                    for det in sorted_by_conf:
                        st.markdown(f"""
                        <div class="detection-item">
                            <div class="detection-index" style="background:{det['color']};">{det['idx']}</div>
                            <div class="detection-info">
                                <div class="detection-id">{det['obj_id']}</div>
                                <div class="detection-name">{det['name'][:20]}</div>
                            </div>
                            <div class="detection-conf" style="color:{det['color']};">{det['conf']*100:.0f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ No objects detected")
                st.image(spatial_image, use_container_width=True)
        else:
            st.markdown("""
            <div style="text-align:center; padding:4rem; background:#f8fafc; border-radius:12px; border:2px dashed #cbd5e1;">
                <div style="font-size:4rem;">🌐</div>
                <div style="font-size:1.2rem; font-weight:600; color:#475569;">Load or Upload an Image</div>
                <div style="color:#94a3b8; margin-top:0.5rem;">Click "Load Random Test Image" or upload your own</div>
            </div>
            """, unsafe_allow_html=True)


# ==================== TAB 4: PERFORMANCE ====================
with tab4:
    import matplotlib.pyplot as plt
    
    # Classification Section
    st.markdown('<p class="section-title">📷 Classification Models</p>', unsafe_allow_html=True)
    
    class_metrics = []
    for model_name in ['mobilenet', 'advanced_cnn', 'custom_cnn', 'efficientnet', 'resnet50']:
        metrics_file = RESULTS_PATH / 'classification' / 'metrics' / f"{model_name}_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                class_metrics.append(json.load(f))
    
    if class_metrics:
        df_class = pd.DataFrame(class_metrics).sort_values('accuracy', ascending=False)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best Acc", f"{df_class['accuracy'].max():.1%}")
        with col2:
            st.metric("Precision", f"{df_class['precision'].mean():.1%}")
        with col3:
            st.metric("Recall", f"{df_class['recall'].mean():.1%}")
        with col4:
            st.metric("F1", f"{df_class['f1_score'].mean():.1%}")
        
        fig, ax = plt.subplots(figsize=(8, 2.5))
        colors = ['#22c55e' if acc > 0.9 else '#f59e0b' if acc > 0.8 else '#ef4444' for acc in df_class['accuracy']]
        ax.barh(df_class['model'], df_class['accuracy'], color=colors, alpha=0.85, height=0.5)
        ax.axvline(x=0.9, color='#22c55e', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Accuracy', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        best = df_class.loc[df_class['accuracy'].idxmax(), 'model']
        st.success(f"🏆 Best Classification Model: **{best}**")
    else:
        st.info("No classification metrics available")
    
    st.markdown("---")
    
    # Detection Section - Two columns for Grid vs Random
    col_grid, col_random = st.columns(2, gap="large")
    
    with col_grid:
        st.markdown('<p class="section-title">🎯 Detection Models (Grid Layout)</p>', unsafe_allow_html=True)
        
        detect_metrics = []
        for variant in ['n', 's', 'm']:
            metrics_file = RESULTS_PATH / 'detection' / 'metrics' / f"yolov8{variant}_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    detect_metrics.append(json.load(f))
        
        if detect_metrics:
            df_detect = pd.DataFrame(detect_metrics).sort_values('mAP50', ascending=False)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best mAP", f"{df_detect['mAP50'].max():.1%}")
            with col2:
                st.metric("Precision", f"{df_detect['precision'].mean():.1%}")
            with col3:
                st.metric("Recall", f"{df_detect['recall'].mean():.1%}")
            with col4:
                st.metric("FPS", f"{df_detect['inference_fps'].mean():.0f}")
            
            fig, ax = plt.subplots(figsize=(5, 2))
            colors = ['#3b82f6', '#8b5cf6', '#06b6d4']
            ax.barh(df_detect['model'], df_detect['mAP50'], color=colors[:len(df_detect)], alpha=0.85, height=0.5)
            ax.set_xlim(0, 1)
            ax.set_xlabel('mAP50', fontsize=9)
            ax.tick_params(labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            best = df_detect.loc[df_detect['mAP50'].idxmax(), 'model']
            st.info(f"🏆 Best Grid Model: **{best}**")
        else:
            st.info("No grid detection metrics available")
    
    with col_random:
        st.markdown('<p class="section-title">🌐 Detection Models (Spatial/Random)</p>', unsafe_allow_html=True)
        
        # Check for random model metrics
        random_metrics = []
        random_metrics_path = RESULTS_PATH / 'detection' / 'metrics_random'
        
        if random_metrics_path.exists():
            for variant in ['n', 's', 'm']:
                metrics_file = random_metrics_path / f"yolov8{variant}_random_metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        random_metrics.append(json.load(f))
        
        if random_metrics:
            df_random = pd.DataFrame(random_metrics).sort_values('mAP50', ascending=False)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best mAP", f"{df_random['mAP50'].max():.1%}")
            with col2:
                st.metric("Precision", f"{df_random['precision'].mean():.1%}")
            with col3:
                st.metric("Recall", f"{df_random['recall'].mean():.1%}")
            with col4:
                st.metric("FPS", f"{df_random['inference_fps'].mean():.0f}")
            
            fig, ax = plt.subplots(figsize=(5, 2))
            colors = ['#f59e0b', '#ea580c', '#dc2626']
            ax.barh(df_random['model'], df_random['mAP50'], color=colors[:len(df_random)], alpha=0.85, height=0.5)
            ax.set_xlim(0, 1)
            ax.set_xlabel('mAP50', fontsize=9)
            ax.tick_params(labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            best = df_random.loc[df_random['mAP50'].idxmax(), 'model']
            st.info(f"🏆 Best Spatial Model: **{best}**")
        else:
            st.markdown("""
            <div style="background:#fef3c7; border:1px solid #f59e0b; border-radius:8px; padding:1rem; text-align:center;">
                <div style="font-size:2rem;">📊</div>
                <div style="font-weight:600; color:#92400e;">No Spatial Metrics Yet</div>
                <div style="font-size:0.85rem; color:#78350f; margin-top:0.3rem;">
                    Train models on the random dataset and save metrics to:<br>
                    <code>results/detection/metrics_random/</code>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ===== FOOTER =====
st.markdown("""
<div class="custom-footer">
    <p><strong>IE 7615</strong> | Group 8: Quoc Hung Le, Hassan Alfareed, Khoa Tran</p>
</div>
""", unsafe_allow_html=True)
