"""
Streamlit Application for CNN Attendance System
Professional UI/UX Design - Optimized Performance
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
import base64
from io import BytesIO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="CNN Attendance System | Group 8",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS with LARGER fonts
st.markdown("""
<style>
    section.stMain .block-container {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
            
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
    
    .main .block-container { max-width: 1900px; padding: 0rem 1rem 0.5rem 1rem !important; }
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .main-header { background: linear-gradient(180deg, #475569 90%, #334155 100%); padding: 0.5rem 1.2rem; border-radius: 8px; margin-top: 0rem; margin-bottom: 0.5rem; }
    .main-header h1 { color: white !important; font-size: 1.4rem !important; font-weight: 700 !important; margin: 0 !important; }
    .main-header p { color: #cbd5e1 !important; font-size: 1rem !important; margin: 0 !important; }
    
    .stTabs [data-baseweb="tab-list"] { 
        gap: 6px; 
        background: linear-gradient(135deg, #f8fafc, #e2e8f0); 
        padding: 8px; 
        border-radius: 12px; 
        border: 2px solid #e2e8f0;
        margin-bottom: 1rem;
    }

    .stTabs [data-baseweb="tab"] { 
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border-radius: 10px !important;
        color: #475569 !important;
        background: transparent !important;
        border: 2px solid transparent !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] { 
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        background: linear-gradient(135deg, #475569 0%, #334155 100%) !important;
        color: #f1f5f9 !important;
        box-shadow: 0 8px 25px rgba(71, 85, 105, 0.4) !important;
        transform: translateY(-3px) scale(1.02) !important;
        border-color: #64748b !important;
    }

    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        font-weight: 700 !important;
        background: rgba(71, 85, 105, 0.1) !important;
        color: #1e293b !important;
        transform: translateY(-1px);
    }

    
    .section-box {
        background: linear-gradient(135deg, rgba(30, 64, 175, 0.9), rgba(15, 23, 42, 0.95));
        color: white !important;
        border: 2px solid #9ca3af;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        overflow: hidden;
        width: 100%;
        box-sizing: border-box;
    }
    .section-box-header,
    .section-box-content {
        width: 100%;
        box-sizing: border-box;
    }

    
    /* LARGER metric boxes */
    .metric-box { border: 2px solid #e5e7eb; border-radius: 6px; background: #f9fafb; padding: 0.1rem; text-align: center; }
    .metric-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #6b7280; }
    .metric-value { font-size: 1.3rem; font-weight: 700; color: #1f2937; }
    .metric-value.green { color: #16a34a; }
    .metric-value.blue { color: #2563eb; }
    
    .stButton > button { background: #475569 !important; color: white !important; border: none !important; border-radius: 5px !important; padding: 0.5rem 1rem !important; font-weight: 600 !important; font-size: 0.9rem !important; }
    .stButton > button:hover { background: #334155 !important; }
    
    div[data-testid="stFileUploader"] { border: 2px dashed #d1d5db !important; border-radius: 6px !important; background: #f9fafb !important; padding: 0.3rem !important; }
    div[data-testid="stFileUploader"]:hover { border-color: #9ca3af !important; }
    div[data-testid="stFileUploader"] small, div[data-testid="stFileUploader"] section > div:first-child { display: none !important; }
    
    .result-card { background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border: 2px solid #86efac; border-radius: 8px; padding: 1rem; text-align: center; }
    .result-label { font-size: 0.8rem; font-weight: 600; text-transform: uppercase; color: #16a34a; }
    .result-value { font-size: 1.4rem; font-weight: 700; color: #15803d; }
    .confidence-badge { display: inline-block; background: #22c55e; color: white; font-size: 1.1rem; font-weight: 600; padding: 0.3rem 0.8rem; border-radius: 12px; margin-top: 0.3rem; }
    
    .top3-item { display: flex; align-items: center; padding: 8px 12px; background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 5px; margin-bottom: 6px; }
    .top3-item.gold { background: #fefce8; border-color: #fde047; }
    .top3-rank { font-size: 1.1rem; margin-right: 8px; }
    .top3-name { flex: 1; font-weight: 600; color: #1f2937; font-size: 1rem; }
    .top3-score { font-weight: 700; color: #475569; font-size: 1.1rem; }
    .top3-bar { width: 100%; height: 3px; background: #e5e7eb; border-radius: 2px; margin-top: 3px; }
    .top3-bar-fill { height: 100%; border-radius: 2px; background: #475569; }
    
    /* LARGER detection items */
    .detection-item { display: flex; align-items: center; padding: 8px 10px; background: #f8fafc; border-radius: 6px; margin-bottom: 4px; border: 1px solid #e5e7eb; }
    .detection-index { width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 0.9rem; margin-right: 8px; flex-shrink: 0; }
    .detection-name { font-weight: 600; color: #1f2937; font-size: 1rem; }
    .detection-id { color: #6b7280; font-size: 0.8rem; }
    .detection-conf { font-weight: 700; font-size: 1.1rem; }
    
    .detection-list { max-height: 250px; overflow-y: auto; overflow-x: hidden; padding-right: 4px; }
    .detection-list::-webkit-scrollbar { width: 5px; }
    .detection-list::-webkit-scrollbar-track { background: #f1f5f9; border-radius: 3px; }
    .detection-list::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
    
    .empty-state { text-align: center; padding: 1.5rem; background: #f8fafc; border-radius: 6px; border: 1px dashed #d1d5db; }
    .empty-state-icon { font-size: 2rem; }
    .empty-state-title { font-size: 1rem; font-weight: 600; color: #475569; }
    
    /* LARGER comparison table */
    .comp-table { width: 100%; border-collapse: collapse; font-size: 0.95rem; }
    .comp-table th { background: #e2e8f0; color: #475569; font-weight: 600; padding: 8px 6px; text-align: center; border-bottom: 2px solid #cbd5e1; font-size: 0.85rem; }
    .comp-table td { padding: 6px 5px; border-bottom: 1px solid #e5e7eb; color: #374151; text-align: center; font-size: 0.95rem; }
    .comp-table tr:hover { background: #f1f5f9; }
    .comp-table .best { color: #16a34a; font-weight: 700; }
    .comp-table .current { background: #fef3c7; }
    .comp-table .realtime { color: #2563eb; font-weight: 700; }
    .comp-table .model-name { text-align: left; font-weight: 600; font-size: 0.9rem; }
    .comp-table .pending { color: #9ca3af; }
    
    .custom-footer { background: #334155; padding: 0.5rem; border-radius: 6px; margin-top: 0.5rem; text-align: center; }
    .custom-footer p { color: #cbd5e1 !important; margin: 0 !important; font-size: 0.8rem; }
    
    .preview-container { display: flex; justify-content: center; align-items: center; }
    .preview-container img { max-width: 140px; max-height: 140px; border-radius: 5px; }
    
    /* LARGER metric boxes for sidebar */
    .metric-box-sm { border: 2px solid #e5e7eb; border-radius: 6px; background: #f9fafb; padding: 0.4rem; text-align: center; }
    .metric-box-sm .metric-label { font-size: 0.7rem; }
    .metric-box-sm .metric-value { font-size: 1.1rem; }
    
    .stImage > img { border-radius: 8px !important; border: 2px solid #e5e7eb !important; }
    .element-container { margin-bottom: 0.15rem; }
    h1, h2, h3 { padding: 0 !important; margin: 0 !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.2rem; }
    
    /* Fix expander text */
    .stExpander { margin-top: 0.5rem; }
    .stExpander summary { font-size: 0.9rem !important; font-weight: 600 !important; }
    
    /* Image frame container - 672x672 square, center top alignment */
    .image-frame {
        width: 672px;
        height: 672px;
        max-width: 100%;
        background: #f8fafc;
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        display: flex;
        justify-content: center;
        align-items: flex-start;
        overflow: hidden;
        margin: 0 auto;
    }
    .image-frame img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
        object-position: center top;
    }
</style>
""", unsafe_allow_html=True)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_CLASS_PATH = PROJECT_ROOT / "models" / "classification"
MODELS_DETECT_PATH = PROJECT_ROOT / "models" / "detection"
MODELS_GRID_M5_PATH = PROJECT_ROOT / "models" / "detection" / "grid_m5"
MODELS_GRID_M8_PATH = PROJECT_ROOT / "models" / "detection" / "grid_m8"
MODELS_GRID_ADV_PATH = PROJECT_ROOT / "models" / "detection" / "grid_adv"
DATA_GRID_ADV_PATH = PROJECT_ROOT / "data" / "multi_objects_grid_adv"
RESULTS_PATH = PROJECT_ROOT / "results"
DATA_PATH = PROJECT_ROOT / "data"

@st.cache_data
def load_class_mapping():
    with open(PROJECT_ROOT / "data" / "class_mapping.json", 'r') as f:
        return json.load(f)

@st.cache_data
def load_object_names():
    nf = PROJECT_ROOT / "data" / "object_names.json"
    if nf.exists():
        with open(nf, 'r') as f:
            data = json.load(f)
            return {k: v for k, v in data.items() if not k.startswith('_')}
    return {}

def calc_detection_stats(detections):
    """Calculate min/max confidence from detections"""
    if not detections:
        return {'min_conf': 0, 'max_conf': 0, 'avg_conf': 0}
    confs = [d['cf'] for d in detections]
    return {
        'min_conf': min(confs),
        'max_conf': max(confs),
        'avg_conf': np.mean(confs)
    }

class_info = load_class_mapping()
object_names_map = load_object_names()
NUM_CLASSES = class_info['num_classes']

def get_display_name(oid): return object_names_map.get(oid, oid)
class_names = [get_display_name(class_info['idx_to_class'][str(i)]) for i in range(NUM_CLASSES)]

ALL_CLS_MODELS = ['mobilenet', 'advanced_cnn', 'custom_cnn', 'efficientnet', 'resnet50']
ALL_YOLO_MODELS = ['n', 's', 'm']
ALL_ADV_GRID_VARIANTS = ['s', 'm', 'l', 'xl']
MODEL_DISPLAY_NAMES = {
    'mobilenet': 'MobileNet', 'advanced_cnn': 'Adv CNN', 'custom_cnn': 'Custom',
    'efficientnet': 'EffNet', 'resnet50': 'ResNet50',
    'n': 'YOLOv8n', 's': 'YOLOv8s', 'm': 'YOLOv8m'
}

# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def get_classifier(mn):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
    
    mp = MODELS_CLASS_PATH / f"{mn}_best.keras"
    if not mp.exists(): return None
    
    # Warmup input for faster first inference
    warmup_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    
    try:
        model = keras.models.load_model(mp, compile=False)
        # No need to compile for inference-only
        # Warmup with direct call (faster than predict)
        _ = model(warmup_input, training=False)
        return model
    except Exception as e:
        print(f"Standard load failed for {mn}: {e}")
    
    try:
        if mn == 'mobilenet':
            base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            base.trainable = True
            for layer in base.layers[:-20]: layer.trainable = False
            model = models.Sequential([base, layers.GlobalAveragePooling2D(), layers.Dense(512, activation='relu'), layers.Dropout(0.5), layers.Dense(NUM_CLASSES, activation='softmax')])
        elif mn == 'resnet50':
            base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            base.trainable = False
            model = models.Sequential([base, layers.GlobalAveragePooling2D(), layers.BatchNormalization(), layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)), layers.Dropout(0.5), layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)), layers.Dropout(0.3), layers.Dense(NUM_CLASSES, activation='softmax')])
        elif mn == 'efficientnet':
            base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            base.trainable = False
            model = models.Sequential([base, layers.GlobalAveragePooling2D(), layers.BatchNormalization(), layers.Dense(512, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.4), layers.Dense(256, activation='relu'), layers.Dropout(0.3), layers.Dense(NUM_CLASSES, activation='softmax')])
        else:
            return None
        model.build(input_shape=(None, 224, 224, 3))
        model.load_weights(mp)
        # Warmup with direct call
        _ = model(warmup_input, training=False)
        return model
    except Exception as we:
        print(f"Could not load {mn}: {we}")
        return None

@st.cache_resource
def get_grid_detector(v, grid_type='m5'):
    from ultralytics import YOLO
    if grid_type == 'm5':
        gp = MODELS_GRID_M5_PATH / f"yolov8{v}_grid_best.pt"
    else:
        gp = MODELS_GRID_M8_PATH / f"yolov8{v}_grid_best.pt"
    gp2 = MODELS_DETECT_PATH / f"yolov8{v}_best.pt"
    mp = gp if gp.exists() else gp2
    if not mp.exists(): return None
    model = YOLO(mp)
    _ = model(np.zeros((224,224,3), dtype=np.uint8), verbose=False)
    return model

@st.cache_resource
def get_advanced_grid_detector(variant):
    from ultralytics import YOLO
    mp = MODELS_GRID_ADV_PATH / f"yolov8s_{variant}_best.pt"
    if not mp.exists(): return None
    print(f"Loading advanced grid model from {mp}")
    model = YOLO(mp)
    _ = model(np.zeros((224,224,3), dtype=np.uint8), verbose=False)
    return model

@st.cache_resource
def get_spatial_detector(v):
    from ultralytics import YOLO
    sp = MODELS_DETECT_PATH / "spatial" / f"yolov8{v}_spatial_best.pt"
    gp = MODELS_DETECT_PATH / f"yolov8{v}_best.pt"
    mp = sp if sp.exists() else gp
    if not mp.exists(): return None
    model = YOLO(mp)
    _ = model(np.zeros((224,224,3), dtype=np.uint8), verbose=False)
    return model

def get_random_images(dt='spatial', n=1):
    if dt == 'spatial': paths = [DATA_PATH / "multi_objects_spatial" / "images" / "test"]
    elif dt == 'grid': paths = [DATA_PATH / "multi_objects_grid" / "images" / "test"]
    elif dt == 'adv_grid': paths = [DATA_GRID_ADV_PATH / "grid_s" / "images" / "test", DATA_GRID_ADV_PATH / "grid_m" / "images" / "test", DATA_GRID_ADV_PATH / "grid_l" / "images" / "test", DATA_GRID_ADV_PATH / "grid_xl" / "images" / "test"]
    else: paths = [DATA_PATH / "processed" / "single_objects" / "test"]
    for p in paths:
        if p.exists():
            if dt == 'single':
                imgs = []
                for sub in p.iterdir():
                    if sub.is_dir(): imgs.extend(list(sub.glob("*.jpg")) + list(sub.glob("*.png")))
            else:
                imgs = list(p.glob("*.jpg")) + list(p.glob("*.png"))
            if imgs: return random.sample(imgs, min(n, len(imgs)))
    return []

# ============================================================================
# INFERENCE - Optimized for speed
# ============================================================================

# Cache preprocessed functions to avoid repeated imports
@st.cache_resource
def get_preprocess_functions():
    """Cache preprocessing functions for faster access"""
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
    return {
        'mobilenet': mobilenet_preprocess,
        'resnet50': resnet_preprocess,
        'efficientnet': efficientnet_preprocess
    }

def fast_classify_single(model, image, model_name):
    """Optimized classification using direct model call instead of predict()"""
    import tensorflow as tf
    
    # Resize using PIL (faster than cv2 for this)
    img = image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
    img = np.array(img, dtype=np.float32)
    
    # Get cached preprocess functions
    preprocess_funcs = get_preprocess_functions()
    
    # Apply preprocessing
    if model_name in preprocess_funcs:
        img = preprocess_funcs[model_name](img)
    else:
        img = img / 255.0
    
    # Expand dims and convert to tensor
    img = np.expand_dims(img, axis=0)
    
    # Use direct model call (MUCH faster than model.predict())
    # model.predict() has ~100-200ms overhead, direct call is ~10-50ms
    start = time.perf_counter()
    pred = model(img, training=False)
    pred = pred.numpy()  # Convert tensor to numpy
    t = (time.perf_counter() - start) * 1000
    
    return pred, t

def run_yolo_detection(image, model, model_variant):
    from PIL import ImageDraw, ImageFont
    
    temp = Path(f"temp_{model_variant}.jpg")
    image.save(temp)
    
    start = time.perf_counter()
    res = model(temp, verbose=False)[0]
    inf_t = (time.perf_counter() - start) * 1000
    
    img_pil = image.copy().convert('RGB')
    draw = ImageDraw.Draw(img_pil)
    colors = [(71,85,105),(34,197,94),(59,130,246),(234,179,8),(239,68,68),(20,184,166),(249,115,22),(168,85,247)]
    hex_c = ['#475569','#22c55e','#3b82f6','#eab308','#ef4444','#14b8a6','#f97316','#a855f7']
    
    dets = []
    for i, box in enumerate(res.boxes):
        ci, cf = int(box.cls[0]), float(box.conf[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        oid = class_info['idx_to_class'][str(ci)]
        nm = object_names_map.get(oid, oid)
        c, hc = colors[i%len(colors)], hex_c[i%len(hex_c)]
        draw.rectangle([x1,y1,x2,y2], outline=c, width=3)
        try: font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=48)
        except: font = ImageFont.load_default()
        lb = str(i+1)
        bb = draw.textbbox((0,0), lb, font=font)
        tw,th = bb[2]-bb[0], bb[3]-bb[1]
        cr = max(tw,th)//2+4
        draw.ellipse([x1+2,y1+2,x1+2+cr*2,y1+2+cr*2], fill=c)
        draw.text((x1+2+cr-tw//2, y1+2+cr-th//2), lb, fill=(255,255,255), font=font)
        dets.append({'idx':i+1,'oid':oid,'nm':nm,'cf':cf,'hc':hc,'cy':(y1+y2)/2,'cx':(x1+x2)/2})
    
    temp.unlink()
    
    return {
        'detections': dets,
        'annotated_image': img_pil,
        'time': inf_t,
        'num_objects': len(dets),
        'avg_conf': np.mean([d['cf'] for d in dets]) if dets else 0,
        'available': True
    }

def create_framed_image(img_pil, frame_size=672):
    """Create a square frame with image centered at top"""
    # Create white background
    frame = Image.new('RGB', (frame_size, frame_size), (248, 250, 252))
    
    # Resize image to fit frame while maintaining aspect ratio
    img_w, img_h = img_pil.size
    ratio = min(frame_size / img_w, frame_size / img_h)
    new_w = int(img_w * ratio)
    new_h = int(img_h * ratio)
    resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Paste at center-top
    x_offset = (frame_size - new_w) // 2
    y_offset = 0  # Top alignment
    frame.paste(resized, (x_offset, y_offset))
    
    return frame

# ============================================================================
# SESSION STATE
# ============================================================================
def get_cls_results():
    if 'cls_results' not in st.session_state: st.session_state.cls_results = {}
    return st.session_state.cls_results

def update_cls_result(mn, res):
    if 'cls_results' not in st.session_state: st.session_state.cls_results = {}
    st.session_state.cls_results[mn] = res

def clear_cls_results():
    st.session_state.cls_results = {}

def get_grid_results():
    if 'grid_results' not in st.session_state: st.session_state.grid_results = {}
    return st.session_state.grid_results

def update_grid_result(v, res):
    if 'grid_results' not in st.session_state: st.session_state.grid_results = {}
    st.session_state.grid_results[v] = res

def clear_grid_results():
    st.session_state.grid_results = {}

def get_spatial_results():
    if 'spatial_results' not in st.session_state: st.session_state.spatial_results = {}
    return st.session_state.spatial_results

def update_spatial_result(v, res):
    if 'spatial_results' not in st.session_state: st.session_state.spatial_results = {}
    st.session_state.spatial_results[v] = res

def clear_spatial_results():
    st.session_state.spatial_results = {}

def get_adv_grid_results():
    if 'adv_grid_results' not in st.session_state: st.session_state.adv_grid_results = {}
    return st.session_state.adv_grid_results

def update_adv_grid_result(v, res):
    if 'adv_grid_results' not in st.session_state: st.session_state.adv_grid_results = {}
    st.session_state.adv_grid_results[v] = res

def clear_adv_grid_results():
    st.session_state.adv_grid_results = {}

def run_single_cls_model(image, model_name):
    model = get_classifier(model_name)
    if model is not None:
        try:
            pred, inf_t = fast_classify_single(model, image, model_name)
            top_idx = np.argmax(pred[0])
            return {'predictions': pred[0], 'top_idx': top_idx, 'top_class': class_names[top_idx], 'top_conf': pred[0][top_idx], 'time': inf_t, 'available': True}
        except Exception as e:
            return {'available': False, 'error': str(e)}
    return {'available': False, 'error': 'Model not found'}

def run_single_grid_model(image, variant, grid_type='m5'):
    model = get_grid_detector(variant, grid_type)
    if model is not None:
        try:
            return run_yolo_detection(image, model, f'grid_{grid_type}_{variant}')
        except Exception as e:
            return {'available': False, 'error': str(e)}
    return {'available': False, 'error': 'Model not found'}

def run_single_spatial_model(image, variant):
    model = get_spatial_detector(variant)
    if model is not None:
        try:
            return run_yolo_detection(image, model, f'spatial_{variant}')
        except Exception as e:
            return {'available': False, 'error': str(e)}
    return {'available': False, 'error': 'Model not found'}

def run_single_adv_grid_model(image, variant):
    model = get_advanced_grid_detector(variant)
    if model is not None:
        try:
            return run_yolo_detection(image, model, f'adv_grid_{variant}')
        except Exception as e:
            return {'available': False, 'error': str(e)}
    return {'available': False, 'error': 'Model not found'}

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="main-header">
    <h1>🎓 CNN ATTENDANCE SYSTEM</h1>
    <p>Discriminative Deep Learning for Object Identification | IE 7615 Project | Goup 8</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab4, tab3 = st.tabs(["📷 SINGLE OBJECT", "🎯 GRID MULTI-OBJECT", "🚀 ADVANCED GRID MULTI-OBJECT", "🌐 SPATIAL MULTI-OBJECT"])

# ============================================================================
# TAB 1: SINGLE OBJECT - With Charts
# ============================================================================
with tab1:
    col_cfg, col_res, col_perf = st.columns([1, 1.5, 1.5], gap="small")
    
    with col_cfg:
        st.markdown('<div class="section-box"><div class="section-box-header">⚙️ CONFIGURATION</div><div class="section-box-content">', unsafe_allow_html=True)
        cls_model = st.selectbox("Model", ALL_CLS_MODELS, format_func=lambda x: {'custom_cnn':'Custom CNN','advanced_cnn':'Advanced CNN','resnet50':'ResNet50','efficientnet':'EfficientNet','mobilenet':'MobileNetV2'}[x], key='cls_m', label_visibility="collapsed")
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-box"><div class="section-box-header">📁 IMAGE SOURCE</div><div class="section-box-content">', unsafe_allow_html=True)
        if st.button("🎲 Random Test Image", use_container_width=True, key='cls_rand'):
            imgs = get_random_images('single', 1)
            if imgs: st.session_state.cls_img = str(imgs[0]); clear_cls_results()
        st.markdown('<div style="text-align:center;color:#9ca3af;font-size:0.8rem;margin:0.3rem 0;">— or —</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload", type=['jpg','jpeg','png'], key='cls_up', label_visibility="collapsed")
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        img = None
        if uploaded:
            img = Image.open(uploaded)
            if 'cls_img' in st.session_state: del st.session_state.cls_img
            if 'last_cls_up' not in st.session_state or st.session_state.last_cls_up != uploaded.name:
                clear_cls_results(); st.session_state.last_cls_up = uploaded.name
        elif 'cls_img' in st.session_state:
            img = Image.open(st.session_state.cls_img)
            if 'last_cls_path' not in st.session_state or st.session_state.last_cls_path != st.session_state.cls_img:
                clear_cls_results(); st.session_state.last_cls_path = st.session_state.cls_img
        
        if img:
            st.markdown('<div class="section-box"><div class="section-box-header">👁️ PREVIEW</div><div class="section-box-content">', unsafe_allow_html=True)
            buffered = BytesIO(); img.resize((140, 140), Image.Resampling.LANCZOS).save(buffered, format="PNG")
            st.markdown(f'<div class="preview-container"><img src="data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}" /></div>', unsafe_allow_html=True)
            st.markdown('</div></div>', unsafe_allow_html=True)
    
    cls_results = get_cls_results()
    if img and cls_model not in cls_results:
        res = run_single_cls_model(img, cls_model); update_cls_result(cls_model, res); cls_results = get_cls_results()
    
    with col_res:
        st.markdown('<div class="section-box"><div class="section-box-header">📊 RESULT</div><div class="section-box-content">', unsafe_allow_html=True)
        if img and cls_model in cls_results and cls_results[cls_model].get('available'):
            res = cls_results[cls_model]; pred = res['predictions']; inf_t = res['time']
            top3_idx = np.argsort(pred)[-3:][::-1]; top3_prob = pred[top3_idx]
            top3_cls = [class_names[i] for i in top3_idx]; top3_oid = [class_info['idx_to_class'][str(i)] for i in top3_idx]
            
            st.markdown(f'<div class="result-card"><div class="result-label">IDENTIFIED OBJECT</div><div class="result-value">{top3_cls[0]}</div><div style="font-size:0.8rem;color:#64748b;">{top3_oid[0]}</div><div class="confidence-badge">{top3_prob[0]*100:.1f}%</div></div>', unsafe_allow_html=True)
            
            st.markdown('<div style="margin-top:0.5rem;"><div class="section-box-header" style="border-radius:6px 6px 0 0;">⚡ INFERENCE METRICS</div></div>', unsafe_allow_html=True)
            cols = st.columns(3)
            cols[0].markdown(f'<div class="metric-box"><div class="metric-label">Time</div><div class="metric-value">{inf_t:.0f}ms</div></div>', unsafe_allow_html=True)
            cols[1].markdown(f'<div class="metric-box"><div class="metric-label">FPS</div><div class="metric-value">{1000/inf_t:.1f}</div></div>', unsafe_allow_html=True)
            cols[2].markdown(f'<div class="metric-box"><div class="metric-label">Confidence</div><div class="metric-value green">{top3_prob[0]*100:.0f}%</div></div>', unsafe_allow_html=True)
            
            st.markdown('<div style="margin-top:0.5rem;"><div class="section-box-header" style="border-radius:6px 6px 0 0;">🏆 TOP 3 PREDICTIONS</div></div>', unsafe_allow_html=True)
            medals = ['🥇', '🥈', '🥉']
            for i, (c, o, p) in enumerate(zip(top3_cls, top3_oid, top3_prob)):
                st.markdown(f'<div class="top3-item{" gold" if i==0 else ""}"><span class="top3-rank">{medals[i]}</span><div style="flex:1;"><div class="top3-name">{c}</div><div style="font-size:0.75rem;color:#94a3b8;">{o}</div><div class="top3-bar"><div class="top3-bar-fill" style="width:{p*100}%;"></div></div></div><span class="top3-score">{p*100:.1f}%</span></div>', unsafe_allow_html=True)
        elif img: st.error(f"Model {cls_model} not available")
        else: st.markdown('<div class="empty-state"><div class="empty-state-icon">📷</div><div class="empty-state-title">No Image</div></div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    with col_perf:
        st.markdown('<div class="section-box"><div class="section-box-header">⚡ SELECTED MODEL STATS</div><div class="section-box-content">', unsafe_allow_html=True)
        if img and cls_model in cls_results and cls_results[cls_model].get('available'):
            res = cls_results[cls_model]; cols = st.columns(4)
            cols[0].markdown(f'<div class="metric-box"><div class="metric-label">Model</div><div class="metric-value" style="font-size:1rem;">{MODEL_DISPLAY_NAMES[cls_model]}</div></div>', unsafe_allow_html=True)
            cols[1].markdown(f'<div class="metric-box"><div class="metric-label">Time</div><div class="metric-value">{res["time"]:.0f}ms</div></div>', unsafe_allow_html=True)
            cols[2].markdown(f'<div class="metric-box"><div class="metric-label">FPS</div><div class="metric-value">{1000/res["time"]:.1f}</div></div>', unsafe_allow_html=True)
            cols[3].markdown(f'<div class="metric-box"><div class="metric-label">Conf</div><div class="metric-value green">{res["top_conf"]*100:.0f}%</div></div>', unsafe_allow_html=True)
        else: st.markdown('<div class="empty-state" style="padding:0.5rem;"><div class="empty-state-icon">⏱️</div><div class="empty-state-title">No Data</div></div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-box"><div class="section-box-header">📈 MODELS COMPARISON</div><div class="section-box-content">', unsafe_allow_html=True)
        if img:
            if st.button("🔄 Compare All Models", use_container_width=True, key='cmp_cls'):
                with st.spinner("Running all models..."):
                    for mn in ALL_CLS_MODELS:
                        if mn not in cls_results: update_cls_result(mn, run_single_cls_model(img, mn))
                    cls_results = get_cls_results()
        
        best_conf = max((cls_results[mn]['top_conf'] for mn in ALL_CLS_MODELS if mn in cls_results and cls_results[mn].get('available')), default=0)
        best_time = min((cls_results[mn]['time'] for mn in ALL_CLS_MODELS if mn in cls_results and cls_results[mn].get('available')), default=float('inf'))
        
        table_html = '<table class="comp-table"><tr><th>Model</th><th>Prediction</th><th>Conf</th><th>Time</th><th>FPS</th></tr>'
        for mn in ALL_CLS_MODELS:
            row_cls = 'current' if mn == cls_model else ''
            if mn in cls_results and cls_results[mn].get('available'):
                r = cls_results[mn]; conf_cls = 'best' if r['top_conf'] == best_conf and len(cls_results) > 1 else ''; time_cls = 'best' if r['time'] == best_time and len(cls_results) > 1 else ''
                pred_short = r['top_class'][:10] + '..' if len(r['top_class']) > 12 else r['top_class']
                table_html += f'<tr class="{row_cls}"><td class="model-name">{MODEL_DISPLAY_NAMES[mn]}</td><td style="font-size:0.8rem;">{pred_short}</td><td class="realtime {conf_cls}">{r["top_conf"]*100:.1f}%</td><td class="realtime {time_cls}">{r["time"]:.0f}ms</td><td class="realtime">{1000/r["time"]:.1f}</td></tr>'
            else: table_html += f'<tr class="{row_cls}"><td class="model-name">{MODEL_DISPLAY_NAMES[mn]}</td><td class="pending" colspan="4">{"—" if not img else "Click Compare All"}</td></tr>'
        table_html += '</table>'
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        # CHARTS - Time & FPS sorted
        available_results = {mn: cls_results[mn] for mn in ALL_CLS_MODELS if mn in cls_results and cls_results[mn].get('available')}
        if len(available_results) > 1:
            import matplotlib.pyplot as plt
            
            # Sort by time (high to low)
            sorted_by_time = sorted(available_results.items(), key=lambda x: x[1]['time'], reverse=True)
            
            st.markdown('<div class="section-box"><div class="section-box-header">📊 PERFORMANCE CHARTS</div><div class="section-box-content">', unsafe_allow_html=True)
            
            fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))
            fig.patch.set_facecolor('#fafafa')
            
            # Time chart (sorted high to low)
            names_t = [MODEL_DISPLAY_NAMES.get(m, m)[:8] for m, _ in sorted_by_time]
            times = [r['time'] for _, r in sorted_by_time]
            colors_t = ['#f59e0b' if m == cls_model else '#64748b' for m, _ in sorted_by_time]
            
            axes[0].barh(names_t, times, color=colors_t, height=0.6)
            axes[0].set_xlabel('Time (ms)', fontsize=9)
            axes[0].set_title('Inference Time', fontsize=10, fontweight='bold')
            axes[0].set_facecolor('#fafafa')
            axes[0].tick_params(labelsize=8)
            for i, v in enumerate(times):
                axes[0].text(v + 2, i, f'{v:.0f}', va='center', fontsize=7)
            
            # FPS chart (sorted high to low)
            sorted_by_fps = sorted(available_results.items(), key=lambda x: 1000/x[1]['time'], reverse=True)
            names_f = [MODEL_DISPLAY_NAMES.get(m, m)[:8] for m, _ in sorted_by_fps]
            fps = [1000/r['time'] for _, r in sorted_by_fps]
            colors_f = ['#f59e0b' if m == cls_model else '#22c55e' for m, _ in sorted_by_fps]
            
            axes[1].barh(names_f, fps, color=colors_f, height=0.6)
            axes[1].set_xlabel('FPS', fontsize=9)
            axes[1].set_title('Frames Per Second', fontsize=10, fontweight='bold')
            axes[1].set_facecolor('#fafafa')
            axes[1].tick_params(labelsize=8)
            for i, v in enumerate(fps):
                axes[1].text(v + 0.5, i, f'{v:.1f}', va='center', fontsize=7)
            
            for ax in axes:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown('</div></div>', unsafe_allow_html=True)


# ============================================================================
# TAB 2: GRID MULTI-OBJECT
# ============================================================================
with tab2:
    col_l, col_r, col_p = st.columns([0.8, 2.2, 1], gap="small")
    
    with col_l:
        st.markdown('<div class="section-box"><div class="section-box-header">⚙️ CONFIG</div><div class="section-box-content">', unsafe_allow_html=True)
        
        # Grid type selection (m5 or m8)
        st.markdown('<div style="font-size:0.75rem;font-weight:600;color:#6b7280;margin-bottom:0.2rem;">GRID TYPE</div>', unsafe_allow_html=True)
        grid_type = st.selectbox(
            "Grid Type", 
            ['m5', 'm8'], 
            index=0, 
            format_func=lambda x: f'Grid {x.upper()} (max {"5" if x=="m5" else "8"} cells)',
            key='grid_type',
            label_visibility="collapsed"
        )
        
        # Clear results when grid type changes
        if 'last_grid_type' not in st.session_state:
            st.session_state.last_grid_type = grid_type
        if st.session_state.last_grid_type != grid_type:
            clear_grid_results()
            st.session_state.last_grid_type = grid_type
        
        st.markdown('<div style="font-size:0.75rem;font-weight:600;color:#6b7280;margin-bottom:0.2rem;margin-top:0.5rem;">MODEL</div>', unsafe_allow_html=True)
        yolo_g = st.selectbox("Model", ALL_YOLO_MODELS, index=1, format_func=lambda x: f'YOLOv8{x}', key='yg', label_visibility="collapsed")
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-box"><div class="section-box-header">📁 SOURCE</div><div class="section-box-content">', unsafe_allow_html=True)
        if st.button("🎲 Random", use_container_width=True, key='g_rand'):
            imgs = get_random_images('grid', 1)
            if imgs: st.session_state.g_img = str(imgs[0]); clear_grid_results()
        st.markdown('<div style="text-align:center;color:#9ca3af;font-size:0.7rem;margin:0.2rem 0;">— or —</div>', unsafe_allow_html=True)
        up_g = st.file_uploader("Upload", type=['jpg','jpeg','png'], key='g_up', label_visibility="collapsed")
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        g_img = None
        if up_g:
            g_img = Image.open(up_g)
            if 'g_img' in st.session_state: del st.session_state.g_img
            if 'last_g_up' not in st.session_state or st.session_state.last_g_up != up_g.name:
                clear_grid_results(); st.session_state.last_g_up = up_g.name
        elif 'g_img' in st.session_state:
            g_img = Image.open(st.session_state.g_img)
            if 'last_g_path' not in st.session_state or st.session_state.last_g_path != st.session_state.g_img:
                clear_grid_results(); st.session_state.last_g_path = st.session_state.g_img
        
        if g_img:
            st.markdown('<div class="section-box"><div class="section-box-header">👁️ PREVIEW</div><div class="section-box-content">', unsafe_allow_html=True)
            # Keep aspect ratio, fit within container
            st.image(g_img, use_container_width=True)
            st.markdown('</div></div>', unsafe_allow_html=True)
    
    grid_results = get_grid_results()
    # Create unique key for grid_type + model combination
    grid_key = f"{grid_type}_{yolo_g}"
    if g_img and grid_key not in grid_results:
        res = run_single_grid_model(g_img, yolo_g, grid_type); update_grid_result(grid_key, res); grid_results = get_grid_results()
    
    with col_r:
        st.markdown('<div class="section-box"><div class="section-box-header">📊 DETECTION RESULT</div><div class="section-box-content">', unsafe_allow_html=True)
        if g_img and grid_key in grid_results and grid_results[grid_key].get('available'):
            res = grid_results[grid_key]; dets = res['detections']; inf_t = res['time']
            
            if dets:
                # Calculate realtime stats
                stats = calc_detection_stats(dets)
                
                # Metrics row - 6 columns with Min/Max Conf
                cols = st.columns(6)
                cols[0].markdown(f'<div class="metric-box"><div class="metric-label">Objects</div><div class="metric-value blue">{len(dets)}</div></div>', unsafe_allow_html=True)
                cols[1].markdown(f'<div class="metric-box"><div class="metric-label">Time</div><div class="metric-value">{inf_t:.0f}ms</div></div>', unsafe_allow_html=True)
                cols[2].markdown(f'<div class="metric-box"><div class="metric-label">FPS</div><div class="metric-value">{1000/inf_t:.1f}</div></div>', unsafe_allow_html=True)
                cols[3].markdown(f'<div class="metric-box"><div class="metric-label">Avg Conf</div><div class="metric-value green">{stats["avg_conf"]*100:.0f}%</div></div>', unsafe_allow_html=True)
                cols[4].markdown(f'<div class="metric-box"><div class="metric-label">Min Conf</div><div class="metric-value">{stats["min_conf"]*100:.0f}%</div></div>', unsafe_allow_html=True)
                cols[5].markdown(f'<div class="metric-box"><div class="metric-label">Max Conf</div><div class="metric-value green">{stats["max_conf"]*100:.0f}%</div></div>', unsafe_allow_html=True)
                
                # Main image in 672x672 frame
                framed_img = create_framed_image(res['annotated_image'], 672)
                st.image(framed_img, use_container_width=True)

            else: st.warning("No objects detected")
        elif g_img: st.error(f"Model YOLOv8{yolo_g} not available")
        else: st.markdown('<div class="empty-state"><div class="empty-state-icon">🎯</div><div class="empty-state-title">No Image</div></div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    with col_p:
        # STATS
        st.markdown('<div class="section-box"><div class="section-box-header">⚡ STATS</div><div class="section-box-content">', unsafe_allow_html=True)
        if g_img and grid_key in grid_results and grid_results[grid_key].get('available'):
            res = grid_results[grid_key]
            cols = st.columns(2)
            cols[0].markdown(f'<div class="metric-box-sm"><div class="metric-label">Model</div><div class="metric-value">v8{yolo_g}</div></div>', unsafe_allow_html=True)
            cols[1].markdown(f'<div class="metric-box-sm"><div class="metric-label">Objects</div><div class="metric-value blue">{res["num_objects"]}</div></div>', unsafe_allow_html=True)
            cols2 = st.columns(2)
            cols2[0].markdown(f'<div class="metric-box-sm"><div class="metric-label">Time</div><div class="metric-value">{res["time"]:.0f}ms</div></div>', unsafe_allow_html=True)
            cols2[1].markdown(f'<div class="metric-box-sm"><div class="metric-label">Conf</div><div class="metric-value green">{res["avg_conf"]*100:.0f}%</div></div>', unsafe_allow_html=True)
        else: st.markdown('<div class="empty-state" style="padding:0.3rem;"><div class="empty-state-title" style="font-size:0.9rem;">No Data</div></div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        # COMPARE
        st.markdown('<div class="section-box"><div class="section-box-header">📈 COMPARE</div><div class="section-box-content">', unsafe_allow_html=True)
        if g_img:
            if st.button("🔄 Compare All", use_container_width=True, key='cmp_grid'):
                with st.spinner("Running..."):
                    for v in ALL_YOLO_MODELS:
                        key_v = f"{grid_type}_{v}"
                        if key_v not in grid_results: 
                            update_grid_result(key_v, run_single_grid_model(g_img, v, grid_type))
                    grid_results = get_grid_results()
        
        # Get keys for current grid_type
        grid_keys = [f"{grid_type}_{v}" for v in ALL_YOLO_MODELS]
        best_conf_g = max((grid_results[k]['avg_conf'] for k in grid_keys if k in grid_results and grid_results[k].get('available')), default=0)
        best_time_g = min((grid_results[k]['time'] for k in grid_keys if k in grid_results and grid_results[k].get('available')), default=float('inf'))
        
        # Table with Min/Max Conf columns
        table_html = '<table class="comp-table"><tr><th>Model</th><th>#</th><th>Avg</th><th>Min</th><th>Max</th><th>ms</th></tr>'
        for v in ALL_YOLO_MODELS:
            key_v = f"{grid_type}_{v}"
            row_cls = 'current' if v == yolo_g else ''
            if key_v in grid_results and grid_results[key_v].get('available'):
                r = grid_results[key_v]
                stats_v = calc_detection_stats(r['detections'])
                conf_cls = 'best' if r['avg_conf'] == best_conf_g and len([k for k in grid_keys if k in grid_results]) > 1 else ''
                time_cls = 'best' if r['time'] == best_time_g and len([k for k in grid_keys if k in grid_results]) > 1 else ''
                table_html += f'<tr class="{row_cls}"><td class="model-name">v8{v}</td><td class="realtime">{r["num_objects"]}</td><td class="realtime {conf_cls}">{stats_v["avg_conf"]*100:.0f}%</td><td class="realtime">{stats_v["min_conf"]*100:.0f}%</td><td class="realtime">{stats_v["max_conf"]*100:.0f}%</td><td class="realtime {time_cls}">{r["time"]:.0f}</td></tr>'
            else: table_html += f'<tr class="{row_cls}"><td class="model-name">v8{v}</td><td class="pending" colspan="5">—</td></tr>'
        table_html += '</table>'
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        # OBJECTS LIST
        if g_img and grid_key in grid_results and grid_results[grid_key].get('available'):
            dets = grid_results[grid_key]['detections']
            if dets:
                st.markdown('<div class="section-box"><div class="section-box-header">📋 OBJECTS</div><div class="section-box-content">', unsafe_allow_html=True)
                det_html = '<div class="detection-list">'
                for d in sorted(dets, key=lambda x:(x['cy'],x['cx'])):
                    det_html += f'<div class="detection-item"><div class="detection-index" style="background:{d["hc"]};">{d["idx"]}</div><div style="flex:1;"><div class="detection-name">{d["nm"]}</div><div class="detection-id">{d["oid"]}</div></div><div class="detection-conf" style="color:{d["hc"]};">{d["cf"]*100:.0f}%</div></div>'
                det_html += '</div>'
                st.markdown(det_html, unsafe_allow_html=True)
                st.markdown('</div></div>', unsafe_allow_html=True)


# ============================================================================
# TAB 3: ADVANCED GRID MULTI-OBJECT
# ============================================================================
with tab4:
    col_l, col_r, col_p = st.columns([0.8, 2.2, 1], gap="small")
    
    with col_l:
        st.markdown('<div class="section-box"><div class="section-box-header">⚙️ CONFIG</div><div class="section-box-content">', unsafe_allow_html=True)
        
        st.markdown('<div style="font-size:0.75rem;font-weight:600;color:#6b7280;margin-bottom:0.2rem;">MODEL VARIANT</div>', unsafe_allow_html=True)
        adv_variant = st.selectbox(
            "Model Variant", 
            ['Any (Compare All)', 's', 'm', 'l', 'xl'], 
            index=0,
            format_func=lambda x: x if x == 'Any (Compare All)' else f'YOLOv8s_{x.upper()}',
            key='adv_variant',
            label_visibility="collapsed"
        )
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-box"><div class="section-box-header">📁 SOURCE</div><div class="section-box-content">', unsafe_allow_html=True)
        if st.button("🎲 Random", use_container_width=True, key='adv_rand'):
            imgs = get_random_images('adv_grid', 1)
            if imgs: st.session_state.adv_img = str(imgs[0]); clear_adv_grid_results()
        st.markdown('<div style="text-align:center;color:#9ca3af;font-size:0.7rem;margin:0.2rem 0;">— or —</div>', unsafe_allow_html=True)
        up_adv = st.file_uploader("Upload", type=['jpg','jpeg','png'], key='adv_up', label_visibility="collapsed")
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        adv_img = None
        if up_adv:
            adv_img = Image.open(up_adv)
            if 'adv_img' in st.session_state: del st.session_state.adv_img
            if 'last_adv_up' not in st.session_state or st.session_state.last_adv_up != up_adv.name:
                clear_adv_grid_results(); st.session_state.last_adv_up = up_adv.name
        elif 'adv_img' in st.session_state:
            adv_img = Image.open(st.session_state.adv_img)
            if 'last_adv_path' not in st.session_state or st.session_state.last_adv_path != st.session_state.adv_img:
                clear_adv_grid_results(); st.session_state.last_adv_path = st.session_state.adv_img
        
        if adv_img:
            st.markdown('<div class="section-box"><div class="section-box-header">👁️ PREVIEW</div><div class="section-box-content">', unsafe_allow_html=True)
            st.image(adv_img, use_container_width=True)
            st.markdown('</div></div>', unsafe_allow_html=True)
    
    adv_grid_results = get_adv_grid_results()
    
    # Handle "Any" mode - run all variants
    if adv_img and adv_variant == 'Any (Compare All)':
        # Check if we need to run all models
        missing = [v for v in ALL_ADV_GRID_VARIANTS if v not in adv_grid_results]
        if missing:
            with st.spinner(f"Running {len(missing)} variants..."):
                for v in missing:
                    res = run_single_adv_grid_model(adv_img, v)
                    update_adv_grid_result(v, res)
                adv_grid_results = get_adv_grid_results()
    
    # Handle single variant mode
    elif adv_img and adv_variant not in adv_grid_results:
        res = run_single_adv_grid_model(adv_img, adv_variant)
        update_adv_grid_result(adv_variant, res)
        adv_grid_results = get_adv_grid_results()
    
    with col_r:
        st.markdown('<div class="section-box"><div class="section-box-header">📊 DETECTION RESULT</div><div class="section-box-content">', unsafe_allow_html=True)
        
        if adv_img:
            # Determine which result to show
            if adv_variant == 'Any (Compare All)':
                # Find best variant by avg_conf
                available = {v: adv_grid_results[v] for v in ALL_ADV_GRID_VARIANTS if v in adv_grid_results and adv_grid_results[v].get('available')}
                if available:
                    best_variant = max(available.keys(), key=lambda v: available[v]['avg_conf'])
                    res = available[best_variant]
                    
                    st.markdown(f'<div style="background:#fef3c7;border:2px solid #fbbf24;border-radius:6px;padding:0.5rem;margin-bottom:0.5rem;text-align:center;"><span style="font-size:0.9rem;font-weight:600;color:#92400e;">🏆 BEST VARIANT: YOLOv8s_{best_variant.upper()}</span></div>', unsafe_allow_html=True)
                else:
                    st.error("No models available")
                    res = None
            else:
                # Single variant mode
                if adv_variant in adv_grid_results and adv_grid_results[adv_variant].get('available'):
                    res = adv_grid_results[adv_variant]
                else:
                    st.error(f"Model YOLOv8s_{adv_variant} not available")
                    res = None
            
            if res:
                dets = res['detections']
                inf_t = res['time']
                
                if dets:
                    # Calculate stats
                    stats = calc_detection_stats(dets)
                    
                    # Metrics row
                    cols = st.columns(6)
                    cols[0].markdown(f'<div class="metric-box"><div class="metric-label">Objects</div><div class="metric-value blue">{len(dets)}</div></div>', unsafe_allow_html=True)
                    cols[1].markdown(f'<div class="metric-box"><div class="metric-label">Time</div><div class="metric-value">{inf_t:.0f}ms</div></div>', unsafe_allow_html=True)
                    cols[2].markdown(f'<div class="metric-box"><div class="metric-label">FPS</div><div class="metric-value">{1000/inf_t:.1f}</div></div>', unsafe_allow_html=True)
                    cols[3].markdown(f'<div class="metric-box"><div class="metric-label">Avg Conf</div><div class="metric-value green">{stats["avg_conf"]*100:.0f}%</div></div>', unsafe_allow_html=True)
                    cols[4].markdown(f'<div class="metric-box"><div class="metric-label">Min Conf</div><div class="metric-value">{stats["min_conf"]*100:.0f}%</div></div>', unsafe_allow_html=True)
                    cols[5].markdown(f'<div class="metric-box"><div class="metric-label">Max Conf</div><div class="metric-value green">{stats["max_conf"]*100:.0f}%</div></div>', unsafe_allow_html=True)
                    
                    # Display image
                    framed_img = create_framed_image(res['annotated_image'], 672)
                    st.image(framed_img, use_container_width=True)
                else:
                    st.warning("No objects detected")
        else:
            st.markdown('<div class="empty-state"><div class="empty-state-icon">🚀</div><div class="empty-state-title">No Image</div></div>', unsafe_allow_html=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    with col_p:
        # STATS
        st.markdown('<div class="section-box"><div class="section-box-header">⚡ STATS</div><div class="section-box-content">', unsafe_allow_html=True)
        
        if adv_img:
            # Show stats for selected or best variant
            display_variant = adv_variant
            if adv_variant == 'Any (Compare All)':
                available = {v: adv_grid_results[v] for v in ALL_ADV_GRID_VARIANTS if v in adv_grid_results and adv_grid_results[v].get('available')}
                if available:
                    display_variant = max(available.keys(), key=lambda v: available[v]['avg_conf'])
            
            if display_variant in adv_grid_results and adv_grid_results[display_variant].get('available'):
                res = adv_grid_results[display_variant]
                cols = st.columns(2)
                variant_label = 's_' + display_variant.upper() if display_variant != 'Any (Compare All)' else 'BEST'
                cols[0].markdown(f'<div class="metric-box-sm"><div class="metric-label">Model</div><div class="metric-value" style="font-size:0.9rem;">v8{variant_label}</div></div>', unsafe_allow_html=True)
                cols[1].markdown(f'<div class="metric-box-sm"><div class="metric-label">Objects</div><div class="metric-value blue">{res["num_objects"]}</div></div>', unsafe_allow_html=True)
                cols2 = st.columns(2)
                cols2[0].markdown(f'<div class="metric-box-sm"><div class="metric-label">Time</div><div class="metric-value">{res["time"]:.0f}ms</div></div>', unsafe_allow_html=True)
                cols2[1].markdown(f'<div class="metric-box-sm"><div class="metric-label">Avg Conf</div><div class="metric-value green">{res["avg_conf"]*100:.0f}%</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="empty-state" style="padding:0.3rem;"><div class="empty-state-title" style="font-size:0.9rem;">No Data</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty-state" style="padding:0.3rem;"><div class="empty-state-title" style="font-size:0.9rem;">No Data</div></div>', unsafe_allow_html=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        # COMPARE ALL VARIANTS
        st.markdown('<div class="section-box"><div class="section-box-header">📈 VARIANTS COMPARISON</div><div class="section-box-content">', unsafe_allow_html=True)
        
        if adv_img:
            if st.button("🔄 Compare All Variants", use_container_width=True, key='cmp_adv_grid'):
                with st.spinner("Running all variants..."):
                    for v in ALL_ADV_GRID_VARIANTS:
                        if v not in adv_grid_results:
                            update_adv_grid_result(v, run_single_adv_grid_model(adv_img, v))
                    adv_grid_results = get_adv_grid_results()
            
            # Find best values
            best_conf = max((adv_grid_results[v]['avg_conf'] for v in ALL_ADV_GRID_VARIANTS if v in adv_grid_results and adv_grid_results[v].get('available')), default=0)
            best_time = min((adv_grid_results[v]['time'] for v in ALL_ADV_GRID_VARIANTS if v in adv_grid_results and adv_grid_results[v].get('available')), default=float('inf'))
            
            # Comparison table
            current_var = adv_variant if adv_variant != 'Any (Compare All)' else None
            table_html = '<table class="comp-table"><tr><th>Variant</th><th>#</th><th>Avg</th><th>Min</th><th>Max</th><th>ms</th></tr>'
            
            for v in ALL_ADV_GRID_VARIANTS:
                row_cls = 'current' if v == current_var else ''
                if v in adv_grid_results and adv_grid_results[v].get('available'):
                    r = adv_grid_results[v]
                    stats_v = calc_detection_stats(r['detections'])
                    conf_cls = 'best' if r['avg_conf'] == best_conf and len([x for x in ALL_ADV_GRID_VARIANTS if x in adv_grid_results]) > 1 else ''
                    time_cls = 'best' if r['time'] == best_time and len([x for x in ALL_ADV_GRID_VARIANTS if x in adv_grid_results]) > 1 else ''
                    table_html += f'<tr class="{row_cls}"><td class="model-name">v8s_{v.upper()}</td><td class="realtime">{r["num_objects"]}</td><td class="realtime {conf_cls}">{stats_v["avg_conf"]*100:.0f}%</td><td class="realtime">{stats_v["min_conf"]*100:.0f}%</td><td class="realtime">{stats_v["max_conf"]*100:.0f}%</td><td class="realtime {time_cls}">{r["time"]:.0f}</td></tr>'
                else:
                    table_html += f'<tr class="{row_cls}"><td class="model-name">v8s_{v.upper()}</td><td class="pending" colspan="5">—</td></tr>'
            
            table_html += '</table>'
            st.markdown(table_html, unsafe_allow_html=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        # OBJECTS LIST
        if adv_img:
            # Determine which result to show
            display_variant = adv_variant
            if adv_variant == 'Any (Compare All)':
                available = {v: adv_grid_results[v] for v in ALL_ADV_GRID_VARIANTS if v in adv_grid_results and adv_grid_results[v].get('available')}
                if available:
                    display_variant = max(available.keys(), key=lambda v: available[v]['avg_conf'])
            
            if display_variant in adv_grid_results and adv_grid_results[display_variant].get('available'):
                dets = adv_grid_results[display_variant]['detections']
                if dets:
                    st.markdown('<div class="section-box"><div class="section-box-header">📋 OBJECTS</div><div class="section-box-content">', unsafe_allow_html=True)
                    det_html = '<div class="detection-list">'
                    for d in sorted(dets, key=lambda x:(x['cy'],x['cx'])):
                        det_html += f'<div class="detection-item"><div class="detection-index" style="background:{d["hc"]};">{d["idx"]}</div><div style="flex:1;"><div class="detection-name">{d["nm"]}</div><div class="detection-id">{d["oid"]}</div></div><div class="detection-conf" style="color:{d["hc"]};">{d["cf"]*100:.0f}%</div></div>'
                    det_html += '</div>'
                    st.markdown(det_html, unsafe_allow_html=True)
                    st.markdown('</div></div>', unsafe_allow_html=True)



# ============================================================================
# TAB 4: SPATIAL MULTI-OBJECT
# ============================================================================
with tab3:
    col_l, col_r, col_p = st.columns([0.8, 2.2, 1], gap="small")
    
    with col_l:
        st.markdown('<div class="section-box"><div class="section-box-header">⚙️ CONFIG</div><div class="section-box-content">', unsafe_allow_html=True)
        yolo_s = st.selectbox("Model", ALL_YOLO_MODELS, index=1, format_func=lambda x: f'YOLOv8{x}', key='ys', label_visibility="collapsed")
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-box"><div class="section-box-header">📁 SOURCE</div><div class="section-box-content">', unsafe_allow_html=True)
        if st.button("🎲 Random", use_container_width=True, key='s_rand'):
            imgs = get_random_images('spatial', 1)
            if imgs: st.session_state.s_img = str(imgs[0]); clear_spatial_results()
        st.markdown('<div style="text-align:center;color:#9ca3af;font-size:0.7rem;margin:0.2rem 0;">— or —</div>', unsafe_allow_html=True)
        up_s = st.file_uploader("Upload", type=['jpg','jpeg','png'], key='s_up', label_visibility="collapsed")
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        s_img = None
        if up_s:
            s_img = Image.open(up_s)
            if 's_img' in st.session_state: del st.session_state.s_img
            if 'last_s_up' not in st.session_state or st.session_state.last_s_up != up_s.name:
                clear_spatial_results(); st.session_state.last_s_up = up_s.name
        elif 's_img' in st.session_state:
            s_img = Image.open(st.session_state.s_img)
            if 'last_s_path' not in st.session_state or st.session_state.last_s_path != st.session_state.s_img:
                clear_spatial_results(); st.session_state.last_s_path = st.session_state.s_img
        
        if s_img:
            st.markdown('<div class="section-box"><div class="section-box-header">👁️ PREVIEW</div><div class="section-box-content">', unsafe_allow_html=True)
            # Keep aspect ratio, fit within container
            st.image(s_img, use_container_width=True)
            st.markdown('</div></div>', unsafe_allow_html=True)
    
    spatial_results = get_spatial_results()
    if s_img and yolo_s not in spatial_results:
        res = run_single_spatial_model(s_img, yolo_s); update_spatial_result(yolo_s, res); spatial_results = get_spatial_results()
    
    with col_r:
        st.markdown('<div class="section-box"><div class="section-box-header">📊 DETECTION RESULT</div><div class="section-box-content">', unsafe_allow_html=True)
        if s_img and yolo_s in spatial_results and spatial_results[yolo_s].get('available'):
            res = spatial_results[yolo_s]; dets = res['detections']; inf_t = res['time']
            
            if dets:
                # Calculate realtime stats
                stats = calc_detection_stats(dets)
                
                # Metrics row - 6 columns with Min/Max Conf
                cols = st.columns(6)
                cols[0].markdown(f'<div class="metric-box"><div class="metric-label">Objects</div><div class="metric-value blue">{len(dets)}</div></div>', unsafe_allow_html=True)
                cols[1].markdown(f'<div class="metric-box"><div class="metric-label">Time</div><div class="metric-value">{inf_t:.0f}ms</div></div>', unsafe_allow_html=True)
                cols[2].markdown(f'<div class="metric-box"><div class="metric-label">FPS</div><div class="metric-value">{1000/inf_t:.1f}</div></div>', unsafe_allow_html=True)
                cols[3].markdown(f'<div class="metric-box"><div class="metric-label">Avg Conf</div><div class="metric-value green">{stats["avg_conf"]*100:.0f}%</div></div>', unsafe_allow_html=True)
                cols[4].markdown(f'<div class="metric-box"><div class="metric-label">Min Conf</div><div class="metric-value">{stats["min_conf"]*100:.0f}%</div></div>', unsafe_allow_html=True)
                cols[5].markdown(f'<div class="metric-box"><div class="metric-label">Max Conf</div><div class="metric-value green">{stats["max_conf"]*100:.0f}%</div></div>', unsafe_allow_html=True)
                
                # Main image in 672x672 frame
                framed_img = create_framed_image(res['annotated_image'], 672)
                st.image(framed_img, use_container_width=True)
            else: st.warning("No objects detected")
        elif s_img: st.error(f"Model YOLOv8{yolo_s} not available")
        else: st.markdown('<div class="empty-state"><div class="empty-state-icon">🌐</div><div class="empty-state-title">No Image</div></div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    with col_p:
        # STATS
        st.markdown('<div class="section-box"><div class="section-box-header">⚡ STATS</div><div class="section-box-content">', unsafe_allow_html=True)
        if s_img and yolo_s in spatial_results and spatial_results[yolo_s].get('available'):
            res = spatial_results[yolo_s]
            cols = st.columns(2)
            cols[0].markdown(f'<div class="metric-box-sm"><div class="metric-label">Model</div><div class="metric-value">v8{yolo_s}</div></div>', unsafe_allow_html=True)
            cols[1].markdown(f'<div class="metric-box-sm"><div class="metric-label">Objects</div><div class="metric-value blue">{res["num_objects"]}</div></div>', unsafe_allow_html=True)
            cols2 = st.columns(2)
            cols2[0].markdown(f'<div class="metric-box-sm"><div class="metric-label">Time</div><div class="metric-value">{res["time"]:.0f}ms</div></div>', unsafe_allow_html=True)
            cols2[1].markdown(f'<div class="metric-box-sm"><div class="metric-label">Avg Conf</div><div class="metric-value green">{res["avg_conf"]*100:.0f}%</div></div>', unsafe_allow_html=True)
        else: st.markdown('<div class="empty-state" style="padding:0.3rem;"><div class="empty-state-title" style="font-size:0.9rem;">No Data</div></div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        # OBJECTS LIST
        if s_img and yolo_s in spatial_results and spatial_results[yolo_s].get('available'):
            dets = spatial_results[yolo_s]['detections']
            if dets:
                st.markdown('<div class="section-box"><div class="section-box-header">📋 DETECTED OBJECTS</div><div class="section-box-content">', unsafe_allow_html=True)
                det_html = '<div class="detection-list">'
                for d in sorted(dets, key=lambda x: x['cf'], reverse=True):
                    det_html += f'<div class="detection-item"><div class="detection-index" style="background:{d["hc"]};">{d["idx"]}</div><div style="flex:1;"><div class="detection-name">{d["nm"]}</div><div class="detection-id">{d["oid"]}</div></div><div class="detection-conf" style="color:{d["hc"]};">{d["cf"]*100:.0f}%</div></div>'
                det_html += '</div>'
                st.markdown(det_html, unsafe_allow_html=True)
                st.markdown('</div></div>', unsafe_allow_html=True)

        # COMPARE with mAP columns
        st.markdown('<div class="section-box"><div class="section-box-header">📈 COMPARE</div><div class="section-box-content">', unsafe_allow_html=True)
        if s_img:
            if st.button("🔄 Compare All", use_container_width=True, key='cmp_spatial'):
                with st.spinner("Running..."):
                    for v in ALL_YOLO_MODELS:
                        if v not in spatial_results: update_spatial_result(v, run_single_spatial_model(s_img, v))
                    spatial_results = get_spatial_results()
        
        best_conf_s = max((spatial_results[v]['avg_conf'] for v in ALL_YOLO_MODELS if v in spatial_results and spatial_results[v].get('available')), default=0)
        best_time_s = min((spatial_results[v]['time'] for v in ALL_YOLO_MODELS if v in spatial_results and spatial_results[v].get('available')), default=float('inf'))
        
        # Table with Min/Max Conf columns
        table_html = '<table class="comp-table"><tr><th>Model</th><th>#</th><th>Avg</th><th>Min</th><th>Max</th><th>ms</th></tr>'
        for v in ALL_YOLO_MODELS:
            row_cls = 'current' if v == yolo_s else ''
            if v in spatial_results and spatial_results[v].get('available'):
                r = spatial_results[v]
                stats_v = calc_detection_stats(r['detections'])
                conf_cls = 'best' if r['avg_conf'] == best_conf_s and len(spatial_results) > 1 else ''
                time_cls = 'best' if r['time'] == best_time_s and len(spatial_results) > 1 else ''
                table_html += f'<tr class="{row_cls}"><td class="model-name">v8{v}</td><td class="realtime">{r["num_objects"]}</td><td class="realtime {conf_cls}">{stats_v["avg_conf"]*100:.0f}%</td><td class="realtime">{stats_v["min_conf"]*100:.0f}%</td><td class="realtime">{stats_v["max_conf"]*100:.0f}%</td><td class="realtime {time_cls}">{r["time"]:.0f}</td></tr>'
            else: table_html += f'<tr class="{row_cls}"><td class="model-name">v8{v}</td><td class="pending" colspan="5">—</td></tr>'
        table_html += '</table>'
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

# FOOTER
st.markdown('<div class="custom-footer"><p><b>Version: 1.28 2026.02.06</b> | Group 8: Quoc Hung, Hassan Alfareed, Khoa Tran</p></div>', unsafe_allow_html=True)
