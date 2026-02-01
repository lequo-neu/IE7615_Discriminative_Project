"""
Streamlit Application for CNN Attendance System
Optimized for 1400x800 display - no scrolling
Author: Kevin (Lead Developer)
"""

import streamlit as st
import numpy as np
import json
from pathlib import Path
import pandas as pd
from PIL import Image
import time
import os

# TensorFlow optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="G8_CNN",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ultra compact CSS for 1400x800 - fit everything
st.markdown("""
<style>
    /* Main container optimized for 1600x800 */
    .main .block-container {
        max-width: 1600px;
        padding: 0.3rem 2rem;
        margin: 0 auto;
    }
    
    .main {
        padding-top: 0.2rem;
    }
    
    /* Hide all Streamlit UI */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Ultra compact typography */
    h1 {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        margin: 0.2rem 0 !important;
        padding: 0 !important;
    }
    
    h2 {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        margin: 0.2rem 0 !important;
    }
    
    h3 {
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        margin: 0.1rem 0 !important;
    }
    
    /* Minimal spacing - tighter lines */
    .element-container {
        margin-bottom: 0.1rem;
    }
    
    .stMarkdown p {
        margin-bottom: 0.2rem;
        line-height: 1.3;
    }
    
    /* Compact image with frame - centered */
    .stImage {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0.3rem auto;
    }
    
    .stImage > img {
        width: 200px !important;
        height: 200px !important;
        border: 2px solid #4a5568;
        border-radius: 8px;
        padding: 6px;
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        box-shadow: 0 3px 5px rgba(0,0,0,0.3);
        display: block;
        margin: 0 auto;
    }
    
    .stImage figcaption {
        text-align: center !important;
        font-size: 0.7rem;
        margin-top: 0.2rem;
    }
    
    /* Compact cards */
    .stSuccess, .stInfo, .stWarning {
        padding: 0.5rem !important;
        margin: 0.2rem 0 !important;
        border-radius: 6px !important;
    }
    
    .stSuccess {
        background: rgba(16, 185, 129, 0.15) !important;
        border: 2px solid rgba(16, 185, 129, 0.4) !important;
    }
    
    .stInfo {
        background: rgba(59, 130, 246, 0.15) !important;
        border: 2px solid rgba(59, 130, 246, 0.4) !important;
    }
    
    /* Compact metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.7rem !important;
        text-transform: lowercase;
    }
    
    div[data-testid="metric-container"] {
        padding: 0.3rem;
    }
    
    /* Thin progress bars */
    .stProgress > div > div {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Compact file uploader */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #4a5568 !important;
        border-radius: 6px !important;
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%) !important;
        padding: 0.6rem !important;
        margin: 0.2rem 0 !important;
    }
    
    div[data-testid="stFileUploader"] section {
        padding: 0.3rem !important;
    }
    
    /* Compact selectbox */
    div[data-testid="stSelectbox"] {
        margin-bottom: 0.3rem;
    }
    
    div[data-testid="stSelectbox"] label {
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        text-transform: lowercase;
        margin-bottom: 0.2rem !important;
    }
    
    /* Thin dividers */
    hr {
        margin: 0.3rem 0;
        border-top: 1px solid #4a5568;
    }
    
    /* Compact tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.8rem;
        padding: 0.2rem 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85rem;
        font-weight: 500;
        padding: 0.4rem 1rem;
    }
    
    /* Compact dataframe */
    .stDataFrame {
        font-size: 0.8rem;
    }
    
    /* Column optimization */
    .stColumn {
        min-width: 0;
        padding: 0 0.4rem;
    }
    
    /* Spinner compact */
    .stSpinner > div {
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_CLASS_PATH = PROJECT_ROOT / "models" / "classification"
MODELS_DETECT_PATH = PROJECT_ROOT / "models" / "detection"
RESULTS_PATH = PROJECT_ROOT / "results"

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
        with st.spinner(f"Loading {model_name}..."):
            import tensorflow as tf
            from tensorflow import keras
            model_path = MODELS_CLASS_PATH / f"{model_name}_best.h5"
            st.session_state.loaded_classifier[model_name] = keras.models.load_model(
                model_path, compile=False
            )
    
    return st.session_state.loaded_classifier[model_name]

def get_detector(model_variant):
    if 'loaded_detector' not in st.session_state:
        st.session_state.loaded_detector = {}
    
    if model_variant not in st.session_state.loaded_detector:
        with st.spinner(f"Loading YOLOv8{model_variant}..."):
            from ultralytics import YOLO
            model_path = MODELS_DETECT_PATH / f"yolov8{model_variant}_best.pt"
            st.session_state.loaded_detector[model_variant] = YOLO(model_path)
    
    return st.session_state.loaded_detector[model_variant]


# Ultra compact header
st.title("🎓 CNN Attendance System")
st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["📷 Classification", "🎯 Detection", "📊 Performance"])


# TAB 1
with tab1:
    st.header("Single Object Classification")
    
    col1, col2 = st.columns([0.9, 1.6], gap="medium")
    
    with col1:
        st.subheader("config")
        
        classifier_model = st.selectbox(
            "model",
            ['mobilenet', 'custom_cnn', 'efficientnet', 'resnet50'],
            index=0,
            format_func=lambda x: {
                'custom_cnn': 'Custom CNN',
                'resnet50': 'ResNet50',
                'efficientnet': 'EfficientNet',
                'mobilenet': 'MobileNetV2'
            }[x]
        )
        
        st.markdown("#### upload")
        uploaded_file = st.file_uploader("file", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            display_img = image.resize((224, 224), Image.Resampling.LANCZOS)

            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.image(display_img, caption="preview (224×224)", width=224)
    
    with col2:
        if uploaded_file is not None:
            import cv2
            
            with st.spinner("Processing..."):
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
                
                st.subheader("results")
                st.success(f"**{top_3_classes[0]}**")
                st.metric("confidence", f"{top_3_probs[0]*100:.1f}%")
                
                st.markdown("**top 3**")
                for i, (cls, prob) in enumerate(zip(top_3_classes, top_3_probs)):
                    medal = ['🥇', '🥈', '🥉'][i]
                    col_a, col_b = st.columns([2.5, 1])
                    with col_a:
                        st.markdown(f"{medal} {cls}")
                        st.progress(float(prob))
                    with col_b:
                        st.metric("", f"{prob*100:.1f}%")
                
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("inference", f"{inference_time:.1f} ms")
                with col_m2:
                    st.metric("fps", f"{1000/inference_time:.1f}")
        else:
            st.info("👈 Upload image")


# TAB 2
with tab2:
    st.header("Multi Object Detection")
    
    col1, col2 = st.columns([0.9, 1.6], gap="medium")
    
    with col1:
        st.subheader("config")
        
        yolo_variant = st.selectbox(
            "model",
            ['n', 's', 'm'],
            format_func=lambda x: {'n': 'YOLOv8n', 's': 'YOLOv8s', 'm': 'YOLOv8m'}[x]
        )
        
        st.markdown("#### upload")
        uploaded_file_det = st.file_uploader("file", type=['jpg', 'jpeg', 'png'], key='det', label_visibility="collapsed")
        
        if uploaded_file_det is not None:
            image = Image.open(uploaded_file_det)
            display_img = image.resize((224, 224), Image.Resampling.LANCZOS)

            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.image(display_img, caption="preview (224×224)", width=224)
    
    with col2:
        if uploaded_file_det is not None:
            import cv2
            
            with st.spinner("Detecting..."):
                temp_path = Path("temp_detect.jpg")
                image.save(temp_path)
                
                model = get_detector(yolo_variant)
                start = time.time()
                results = model(temp_path, verbose=False)[0]
                inference_time = (time.time() - start) * 1000
                
                img_with_boxes = results.plot()
                img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
                
                st.subheader("results")
                
                # Limit detection image height
                result_img = Image.fromarray(img_with_boxes)
                result_img.thumbnail((600, 400), Image.Resampling.LANCZOS)
                st.image(result_img, use_container_width=True)
                
                if len(results.boxes) > 0:
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("objects", len(results.boxes))
                    with col_m2:
                        st.metric("time", f"{inference_time:.1f} ms")
                    with col_m3:
                        st.metric("fps", f"{1000/inference_time:.1f}")
                    
                    # Compact table
                    detections = []
                    for idx, box in enumerate(results.boxes):
                        cls_idx = int(box.cls[0])
                        conf = float(box.conf[0])
                        detections.append({
                            '#': idx + 1,
                            'Object': class_names[cls_idx],
                            'Conf': f"{conf*100:.0f}%"
                        })
                    
                    df_det = pd.DataFrame(detections)
                    st.dataframe(df_det, use_container_width=True, hide_index=True, height=150)
                else:
                    st.warning("No objects")
                
                temp_path.unlink()
        else:
            st.info("👈 Upload image")


# TAB 3
with tab3:
    st.header("Performance")
    
    import matplotlib.pyplot as plt
    
    # Split into two columns for side-by-side view on 1600x800
    col_left, col_right = st.columns(2, gap="large")
    
    with col_left:
        st.subheader("classification")
        
        class_metrics = []
        for model_name in ['mobilenet', 'custom_cnn', 'efficientnet', 'resnet50']:
            metrics_file = RESULTS_PATH / 'classification' / 'metrics' / f"{model_name}_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    class_metrics.append(json.load(f))
        
        if class_metrics:
            df_class = pd.DataFrame(class_metrics)
            df_class = df_class.sort_values('accuracy', ascending=False)
            
            # Compact metrics in 2x2 grid
            col1, col2 = st.columns(2)
            with col1:
                st.metric("accuracy", f"{df_class['accuracy'].max():.3f}")
                st.metric("precision", f"{df_class['precision'].mean():.3f}")
            with col2:
                st.metric("recall", f"{df_class['recall'].mean():.3f}")
                st.metric("f1", f"{df_class['f1_score'].mean():.3f}")
            
            # Compact table
            st.dataframe(
                df_class[['model', 'accuracy', 'inference_time_ms']].style.format({
                    'accuracy': '{:.4f}',
                    'inference_time_ms': '{:.1f}'
                }),
                use_container_width=True,
                hide_index=True,
                height=130
            )
            
            # Chart for 1600x800
            fig, ax = plt.subplots(figsize=(7, 3.2))
            colors = ['#10b981' if acc > 0.9 else '#ef4444' for acc in df_class['accuracy']]
            ax.barh(df_class['model'], df_class['accuracy'], color=colors, alpha=0.8)
            ax.axvline(x=0.9, color='#6b7280', linestyle='--', linewidth=1.5)
            ax.grid(True, alpha=0.2, axis='x')
            ax.tick_params(labelsize=8)
            ax.set_xlabel('Accuracy', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Recommendations
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.success(f"**Best:** {df_class.loc[df_class['accuracy'].idxmax(), 'model']}")
            with col_r2:
                st.info(f"**Fastest:** {df_class.loc[df_class['inference_time_ms'].idxmin(), 'model']}")
        else:
            st.info("No metrics")
    
    with col_right:
        st.subheader("detection")
        
        detect_metrics = []
        for variant in ['n', 's', 'm']:
            metrics_file = RESULTS_PATH / 'detection' / 'metrics' / f"yolov8{variant}_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    detect_metrics.append(json.load(f))
        
        if detect_metrics:
            df_detect = pd.DataFrame(detect_metrics)
            df_detect = df_detect.sort_values('mAP50', ascending=False)
            
            # Compact metrics in 2x2 grid
            col1, col2 = st.columns(2)
            with col1:
                st.metric("mAP50", f"{df_detect['mAP50'].max():.3f}")
                st.metric("precision", f"{df_detect['precision'].mean():.3f}")
            with col2:
                st.metric("recall", f"{df_detect['recall'].mean():.3f}")
                st.metric("avg fps", f"{df_detect['fps'].mean():.0f}")
            
            # Compact table
            st.dataframe(
                df_detect[['model', 'mAP50', 'fps']].style.format({
                    'mAP50': '{:.3f}',
                    'fps': '{:.1f}'
                }),
                use_container_width=True,
                hide_index=True,
                height=100
            )
            
            # Chart for 1600x800
            fig, ax = plt.subplots(figsize=(7, 3.2))
            ax.barh(df_detect['model'], df_detect['mAP50'], color='#3b82f6', alpha=0.8)
            ax.grid(True, alpha=0.2, axis='x')
            ax.tick_params(labelsize=8)
            ax.set_xlabel('mAP50', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Recommendation
            st.info(f"**Best:** {df_detect.loc[df_detect['mAP50'].idxmax(), 'model']}")
        else:
            st.info("No detection metrics")

# Minimal footer
st.markdown("---")
st.markdown("**IE 7615** | Group 8: Quoc Hung Le, Hassan Alfareed, Khoa Tran")
