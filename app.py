import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from PIL import Image as PILImage
warnings.filterwarnings('ignore')

# PAGE CONFIG
st.set_page_config(
    page_title="Stroke Risk Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    [data-testid="stSidebar"] { background-color: #111827 !important; }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stRadio label { color: white !important; }
    .hero-section {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 60px 40px; border-radius: 20px;
        text-align: center; margin-bottom: 30px; color: white;
    }
    .hero-title { font-size: 48px; font-weight: 800; color: white; margin-bottom: 10px; }
    .hero-subtitle { font-size: 18px; color: #a8b2d8; margin-bottom: 30px; }
    .stat-card {
        background: white; padding: 25px; border-radius: 15px;
        text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #0f3460;
    }
    .stat-number { font-size: 36px; font-weight: 800; color: #0f3460; }
    .stat-label { font-size: 14px; color: #666; margin-top: 5px; }
    .feature-card {
        background: white; padding: 25px; border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 15px; border-top: 4px solid #0f3460;
    }
    .feature-icon { font-size: 40px; margin-bottom: 10px; }
    .feature-title { font-size: 18px; font-weight: 700; color: #1a1a2e; margin-bottom: 8px; }
    .feature-desc { font-size: 14px; color: #666; line-height: 1.6; }
    .result-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        padding: 30px; border-radius: 15px; text-align: center;
        color: white; margin: 20px 0;
    }
    .result-low {
        background: linear-gradient(135deg, #26de81, #20bf6b);
        padding: 30px; border-radius: 15px; text-align: center;
        color: white; margin: 20px 0;
    }
    .result-title { font-size: 28px; font-weight: 800; margin-bottom: 10px; }
    .result-percentage { font-size: 56px; font-weight: 900; margin: 10px 0; }
    .result-subtitle { font-size: 16px; opacity: 0.9; }
    .metric-card {
        background: white; padding: 20px; border-radius: 12px;
        text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    .metric-value { font-size: 24px; font-weight: 700; color: #0f3460; }
    .metric-label { font-size: 12px; color: #888; margin-top: 5px; }
    .section-header {
        font-size: 28px; font-weight: 700; color: #1a1a2e;
        margin-bottom: 20px; padding-bottom: 10px;
        border-bottom: 3px solid #0f3460;
    }
    .input-section-title {
        font-size: 18px; font-weight: 700; color: #0f3460;
        margin-bottom: 20px; padding-bottom: 10px;
        border-bottom: 2px solid #f0f0f0;
    }
    .disclaimer {
        background: #fff3cd; border: 1px solid #ffc107;
        border-radius: 10px; padding: 15px 20px;
        margin-top: 20px; font-size: 13px; color: #856404;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stButton"] > button {
        background-color: #0f3460 !important; color: white !important;
        border: none !important; padding: 15px 40px !important;
        border-radius: 50px !important; font-size: 18px !important;
        font-weight: 700 !important; width: 100% !important;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #16213e !important; border: none !important;
    }
    div[data-testid="stButton"] > button p {
        color: white !important; font-size: 18px !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# LOAD MODELS
@st.cache_resource
def load_models():
    ml_model = joblib.load(r'C:\Users\Ar\Desktop\multi-model-strok-prediction\models\best_ml_model.pkl')
    ml_scaler = joblib.load(r'C:\Users\Ar\Desktop\multi-model-strok-prediction\models\scaler.pkl')
    dl_model = joblib.load(r'C:\Users\Ar\Desktop\multi-model-strok-prediction\models\dl_model.pkl')
    return ml_model, ml_scaler, dl_model

@st.cache_resource
def load_cv_model():
    cv_model = joblib.load(r'C:\Users\Ar\Desktop\multi-model-strok-prediction\models\cv_model.pkl')
    img_scaler = joblib.load(r'C:\Users\Ar\Desktop\multi-model-strok-prediction\models\img_scaler.pkl')
    return cv_model, img_scaler

ml_model, ml_scaler, dl_model = load_models()
cv_model, img_scaler = load_cv_model()

# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0;'>
        <div style='font-size:50px;'>🧠</div>
        <div style='font-size:20px; font-weight:700; color:white;'>StrokeAI</div>
        <div style='font-size:12px; color:#aaa;'>Prediction System</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔍 Prediction", "🧠 Brain MRI", "📊 Model Info"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px; color:#aaa; padding:10px;'>
        <b style='color:white;'>Models Used:</b><br>
        ✅ XGBoost (ML)<br>
        ✅ Neural Network (DL)<br>
        ✅ Computer Vision (CV)<br><br>
        <b style='color:white;'>Dataset:</b><br>
        📊 5,110 patients<br>
        📋 12 clinical features<br>
        🧠 2,501 brain CT scans
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════
if page == "🏠 Home":

    st.markdown("""
    <div class='hero-section'>
        <div style='font-size:60px; margin-bottom:10px;'>🧠</div>
        <div class='hero-title'>Stroke Risk Prediction System</div>
        <div class='hero-subtitle'>
            Advanced Multi-Modal AI System combining Machine Learning,
            Deep Learning and Computer Vision for early stroke detection
        </div>
        <div style='margin-top:20px;'>
            <span style='background:rgba(255,255,255,0.2); padding:8px 20px;
                        border-radius:50px; margin:5px; font-size:14px;'>
                🤖 XGBoost ML
            </span>
            <span style='background:rgba(255,255,255,0.2); padding:8px 20px;
                        border-radius:50px; margin:5px; font-size:14px;'>
                🧬 Neural Network
            </span>
            <span style='background:rgba(255,255,255,0.2); padding:8px 20px;
                        border-radius:50px; margin:5px; font-size:14px;'>
                👁️ Computer Vision
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-number'>5,110</div>
            <div class='stat-label'>Patients Analyzed</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-number'>98.2%</div>
            <div class='stat-label'>CV Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-number'>0.994</div>
            <div class='stat-label'>CV AUC Score</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-number'>3</div>
            <div class='stat-label'>AI Models</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>About This System</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:white; padding:30px; border-radius:15px;
                box-shadow:0 4px 15px rgba(0,0,0,0.08); margin-bottom:30px;
                font-size:15px; color:#444; line-height:1.8;'>
        Stroke is one of the leading causes of death and disability worldwide.
        Early detection of stroke risk factors can save lives and prevent permanent disability.
        This system uses three complementary AI approaches to provide a comprehensive
        stroke risk assessment combining clinical data analysis and brain imaging.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>How It Works</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>🤖</div>
            <div class='feature-title'>Machine Learning</div>
            <div class='feature-desc'>
                XGBoost model trained on 5,110 patient records with advanced
                preprocessing including SMOTETomek balancing and feature encoding.
                Achieves 93% accuracy on clinical stroke prediction.
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>🧬</div>
            <div class='feature-title'>Deep Learning</div>
            <div class='feature-desc'>
                Neural Network with multiple hidden layers (64-32-16 neurons)
                captures complex non-linear patterns in patient data that
                traditional ML models may miss.
            </div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>👁️</div>
            <div class='feature-title'>Computer Vision</div>
            <div class='feature-desc'>
                Random Forest classifier trained on 2,501 brain CT scan images
                achieving 98.2% accuracy and 0.994 AUC for visual stroke detection.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Key Stroke Risk Factors</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='background:white; padding:25px; border-radius:15px;
                    box-shadow:0 4px 15px rgba(0,0,0,0.08);'>
            <div style='font-weight:700; font-size:16px; color:#0f3460; margin-bottom:15px;'>
                ⚠️ High Risk Factors</div>
            <div style='font-size:14px; color:#444; line-height:2;'>
                🔴 &nbsp; High Blood Pressure (Hypertension)<br>
                🔴 &nbsp; High Average Glucose Level<br>
                🔴 &nbsp; High BMI (Obesity)<br>
                🔴 &nbsp; Advanced Age (60+)<br>
                🔴 &nbsp; Heart Disease History<br>
                🔴 &nbsp; Smoking (Current or Former)
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:white; padding:25px; border-radius:15px;
                    box-shadow:0 4px 15px rgba(0,0,0,0.08);'>
            <div style='font-weight:700; font-size:16px; color:#20bf6b; margin-bottom:15px;'>
                ✅ Protective Factors</div>
            <div style='font-size:14px; color:#444; line-height:2;'>
                🟢 &nbsp; Normal Blood Pressure<br>
                🟢 &nbsp; Healthy Glucose Levels<br>
                🟢 &nbsp; Healthy BMI (18.5 - 24.9)<br>
                🟢 &nbsp; Young Age<br>
                🟢 &nbsp; No Heart Disease<br>
                🟢 &nbsp; Never Smoked
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; padding:30px; background:white;
                border-radius:15px; box-shadow:0 4px 15px rgba(0,0,0,0.08);'>
        <div style='font-size:22px; font-weight:700; color:#1a1a2e; margin-bottom:10px;'>
            Ready to check your stroke risk?</div>
        <div style='font-size:14px; color:#666;'>
            👈 Click on <b>"🔍 Prediction"</b> or <b>"🧠 Brain MRI"</b> in the sidebar
        </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════
# PAGE 2 — PREDICTION
# ══════════════════════════════
elif page == "🔍 Prediction":

    st.markdown("<div class='section-header'>🔍 Stroke Risk Prediction</div>",
                unsafe_allow_html=True)
    st.write("Fill in the patient details below and click Predict to get instant AI-powered stroke risk assessment.")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='input-section-title'>👤 Patient Information</div>",
                    unsafe_allow_html=True)
        age = st.slider("Age", 1, 100, 50)
        avg_glucose_level = st.slider("Average Glucose Level (mg/dL)", 50.0, 300.0, 100.0, 0.1)
        bmi = st.slider("BMI", 10.0, 60.0, 25.0, 0.1)
        hypertension = st.radio("Hypertension", ["No", "Yes"], horizontal=True)
        heart_disease = st.radio("Heart Disease", ["No", "Yes"], horizontal=True)

    with col2:
        st.markdown("<div class='input-section-title'>🏥 Medical & Lifestyle</div>",
                    unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        ever_married = st.radio("Ever Married", ["No", "Yes"], horizontal=True)
        work_type = st.selectbox("Work Type",
                                  ["Private", "Self-employed", "Govt_job",
                                   "children", "Never_worked"])
        residence_type = st.radio("Residence Type", ["Urban", "Rural"], horizontal=True)
        smoking_status = st.selectbox("Smoking Status",
                                       ["never smoked", "formerly smoked",
                                        "smokes", "Unknown"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Predict Stroke Risk", use_container_width=True)

    if predict_btn:
        hypertension_val = 1 if hypertension == "Yes" else 0
        heart_disease_val = 1 if heart_disease == "Yes" else 0
        gender_male = 1 if gender == "Male" else 0
        gender_other = 1 if gender == "Other" else 0
        ever_married_yes = 1 if ever_married == "Yes" else 0
        work_never = 1 if work_type == "Never_worked" else 0
        work_private = 1 if work_type == "Private" else 0
        work_self = 1 if work_type == "Self-employed" else 0
        work_children = 1 if work_type == "children" else 0
        residence_urban = 1 if residence_type == "Urban" else 0
        smoking_formerly = 1 if smoking_status == "formerly smoked" else 0
        smoking_never = 1 if smoking_status == "never smoked" else 0
        smoking_smokes = 1 if smoking_status == "smokes" else 0

        input_data = pd.DataFrame([[
            age, hypertension_val, heart_disease_val,
            avg_glucose_level, bmi,
            gender_male, gender_other, ever_married_yes,
            work_never, work_private, work_self, work_children,
            residence_urban,
            smoking_formerly, smoking_never, smoking_smokes
        ]], columns=[
            'age', 'hypertension', 'heart_disease',
            'avg_glucose_level', 'bmi',
            'gender_Male', 'gender_Other', 'ever_married_Yes',
            'work_type_Never_worked', 'work_type_Private',
            'work_type_Self-employed', 'work_type_children',
            'Residence_type_Urban',
            'smoking_status_formerly smoked',
            'smoking_status_never smoked',
            'smoking_status_smokes'
        ])

        input_scaled = ml_scaler.transform(input_data)
        ml_prob = ml_model.predict_proba(input_scaled)[0][1]
        dl_prob = dl_model.predict_proba(input_scaled)[0][1]
        combined_prob = (ml_prob + dl_prob) / 2
        risk_pct = round(combined_prob * 100, 2)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Prediction Results")

        if combined_prob >= 0.5:
            st.markdown(f"""
            <div class='result-high'>
                <div class='result-title'>⚠️ HIGH STROKE RISK</div>
                <div class='result-percentage'>{risk_pct}%</div>
                <div class='result-subtitle'>
                    This patient shows significant stroke risk factors.<br>
                    Immediate medical consultation is strongly recommended.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-low'>
                <div class='result-title'>✅ LOW STROKE RISK</div>
                <div class='result-percentage'>{risk_pct}%</div>
                <div class='result-subtitle'>
                    This patient shows low stroke risk based on entered details.<br>
                    Regular health checkups are still recommended.
                </div>
            </div>""", unsafe_allow_html=True)

        bar_color = "#ee5a24" if combined_prob >= 0.5 else "#20bf6b"
        st.markdown(f"""
        <div style='background:white; padding:20px; border-radius:12px;
                    box-shadow:0 2px 10px rgba(0,0,0,0.08); margin:15px 0;'>
            <div style='font-size:14px; color:#666; margin-bottom:8px;'>Risk Level</div>
            <div style='background:#f0f0f0; border-radius:50px; height:20px;'>
                <div style='background:{bar_color}; width:{risk_pct}%;
                            height:20px; border-radius:50px;'></div>
            </div>
            <div style='display:flex; justify-content:space-between;
                        font-size:12px; color:#888; margin-top:5px;'>
                <span>0% (No Risk)</span>
                <span>50% (Moderate)</span>
                <span>100% (High Risk)</span>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### Individual Model Predictions")
        col1, col2, col3 = st.columns(3)

        with col1:
            ml_color = "#ee5a24" if ml_prob >= 0.5 else "#20bf6b"
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size:24px; margin-bottom:5px;'>🤖</div>
                <div class='metric-value'>{round(ml_prob*100, 2)}%</div>
                <div class='metric-label'>XGBoost ML</div>
                <div style='font-size:11px; color:{ml_color}; font-weight:600; margin-top:5px;'>
                    {"HIGH RISK" if ml_prob >= 0.5 else "LOW RISK"}</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            dl_color = "#ee5a24" if dl_prob >= 0.5 else "#20bf6b"
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size:24px; margin-bottom:5px;'>🧬</div>
                <div class='metric-value'>{round(dl_prob*100, 2)}%</div>
                <div class='metric-label'>Neural Network DL</div>
                <div style='font-size:11px; color:{dl_color}; font-weight:600; margin-top:5px;'>
                    {"HIGH RISK" if dl_prob >= 0.5 else "LOW RISK"}</div>
            </div>""", unsafe_allow_html=True)

        with col3:
            comb_color = "#ee5a24" if combined_prob >= 0.5 else "#20bf6b"
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size:24px; margin-bottom:5px;'>🔮</div>
                <div class='metric-value'>{risk_pct}%</div>
                <div class='metric-label'>Combined Score</div>
                <div style='font-size:11px; color:{comb_color}; font-weight:600; margin-top:5px;'>
                    {"HIGH RISK" if combined_prob >= 0.5 else "LOW RISK"}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Patient Summary")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{age}</div>
                <div class='metric-label'>Age (years)</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{bmi}</div>
                <div class='metric-label'>BMI</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{avg_glucose_level}</div>
                <div class='metric-label'>Glucose (mg/dL)</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{hypertension}</div>
                <div class='metric-label'>Hypertension</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class='disclaimer'>
            ⚠️ <b>Medical Disclaimer:</b> This tool is for educational and research
            purposes only. Always consult a qualified healthcare provider.
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════
# PAGE 3 — BRAIN MRI
# ══════════════════════════════
elif page == "🧠 Brain MRI":

    st.markdown("<div class='section-header'>🧠 Brain CT Scan Analysis</div>",
                unsafe_allow_html=True)
    st.write("Upload a brain CT scan image to detect stroke using Computer Vision model.")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div style='background:white; padding:25px; border-radius:15px;
                    box-shadow:0 4px 15px rgba(0,0,0,0.08);'>
            <div style='font-size:18px; font-weight:700; color:#0f3460;
                        margin-bottom:15px;'>📤 Upload Brain CT Scan</div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose a brain CT scan image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a brain CT scan image for stroke detection"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#e8f4fd; padding:20px; border-radius:12px;
                    border-left:4px solid #0f3460;'>
            <div style='font-size:15px; font-weight:700; color:#0f3460;
                        margin-bottom:10px;'>ℹ️ About CV Model</div>
            <div style='font-size:13px; color:#444; line-height:1.8;'>
                ✅ Trained on 2,501 brain CT scans<br>
                ✅ Normal: 1,551 images<br>
                ✅ Stroke: 950 images<br>
                ✅ Accuracy: 98.2%<br>
                ✅ Precision: 100%<br>
                ✅ Recall: 95.26%<br>
                ✅ AUC: 0.994
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        if uploaded_file is not None:
            image = PILImage.open(uploaded_file).convert('L')

            st.markdown("""
            <div style='background:white; padding:20px; border-radius:15px;
                        box-shadow:0 4px 15px rgba(0,0,0,0.08);'>
                <div style='font-size:16px; font-weight:700; color:#0f3460;
                            margin-bottom:10px;'>Uploaded Image</div>
            """, unsafe_allow_html=True)
            st.image(image, caption='Brain CT Scan', use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            img_resized = image.resize((64, 64))
            img_array = np.array(img_resized).flatten().reshape(1, -1)
            img_scaled = img_scaler.transform(img_array)

            pred = cv_model.predict(img_scaled)[0]
            prob = cv_model.predict_proba(img_scaled)[0]
            class_names = ['Normal', 'Stroke']
            pred_label = class_names[pred]
            confidence = round(prob[pred] * 100, 2)

            st.markdown("<br>", unsafe_allow_html=True)

            if pred == 1:
                st.markdown(f"""
                <div class='result-high'>
                    <div class='result-title'>⚠️ STROKE DETECTED</div>
                    <div class='result-percentage'>{confidence}%</div>
                    <div class='result-subtitle'>
                        Brain CT scan shows signs of stroke.<br>
                        Immediate medical attention recommended.
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-low'>
                    <div class='result-title'>✅ NORMAL BRAIN SCAN</div>
                    <div class='result-percentage'>{confidence}%</div>
                    <div class='result-subtitle'>
                        No stroke detected in brain CT scan.<br>
                        Regular checkups still recommended.
                    </div>
                </div>""", unsafe_allow_html=True)

            col3, col4 = st.columns(2)
            with col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{pred_label}</div>
                    <div class='metric-label'>Prediction</div>
                </div>""", unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{confidence}%</div>
                    <div class='metric-label'>Confidence</div>
                </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style='background:white; padding:60px 30px; border-radius:15px;
                        box-shadow:0 4px 15px rgba(0,0,0,0.08); text-align:center;'>
                <div style='font-size:60px; margin-bottom:15px;'>🧠</div>
                <div style='font-size:18px; font-weight:700; color:#0f3460;
                            margin-bottom:10px;'>Upload a Brain CT Scan</div>
                <div style='font-size:14px; color:#666;'>
                    Upload an image on the left to get instant<br>
                    AI-powered stroke detection results
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='disclaimer'>
        ⚠️ <b>Medical Disclaimer:</b> This CV analysis is for educational purposes only.
        Always consult a qualified radiologist for brain scan interpretation.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════
# PAGE 4 — MODEL INFO
# ══════════════════════════════
elif page == "📊 Model Info":

    st.markdown("<div class='section-header'>📊 Model Performance</div>",
                unsafe_allow_html=True)

    comparison_data = {
        'Model': ['Logistic Regression', 'Random Forest',
                  'XGBoost', 'Neural Network (MLP)', 'CV Model (Brain CT)'],
        'Type': ['ML', 'ML', 'ML', 'Deep Learning', 'Computer Vision'],
        'Accuracy': ['~78%', '~87%', '93.64%', '86.89%', '98.2%'],
        'Precision': ['N/A', 'N/A', '25.81%', '12.50%', '100.0%'],
        'Recall': ['N/A', 'N/A', '16.0%', '28.0%', '95.26%'],
        'F1 Score': ['N/A', 'N/A', 'N/A', 'N/A', '97.57%'],
        'AUC': ['~0.82', '~0.88', '0.786', '0.766', '0.994']
    }

    st.dataframe(
        pd.DataFrame(comparison_data),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='background:white; padding:25px; border-radius:15px;
                    box-shadow:0 4px 15px rgba(0,0,0,0.08);'>
            <div style='font-size:18px; font-weight:700; color:#0f3460;
                        margin-bottom:15px;'>🔑 Key Findings</div>
            <div style='font-size:14px; color:#444; line-height:2;'>
                ✅ CV Model achieved best accuracy (98.2%) on brain CT scans<br>
                ✅ XGBoost best on tabular clinical data (93.64%)<br>
                ✅ Neural Network captured complex patterns (86.89%)<br>
                ✅ Low recall on tabular data due to class imbalance (5% stroke)<br>
                ✅ SMOTETomek helped balance training data<br>
                ✅ Combined model provides most reliable predictions
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background:white; padding:25px; border-radius:15px;
                    box-shadow:0 4px 15px rgba(0,0,0,0.08);'>
            <div style='font-size:18px; font-weight:700; color:#0f3460;
                        margin-bottom:15px;'>🛠️ Technical Stack</div>
            <div style='font-size:14px; color:#444; line-height:2;'>
                🐍 Python 3.12<br>
                📊 Scikit-learn, XGBoost<br>
                🧬 MLPClassifier (Neural Network)<br>
                👁️ Random Forest (Computer Vision)<br>
                ⚖️ SMOTETomek (Class Balancing)<br>
                🌐 Streamlit (Web Dashboard)<br>
                📈 Matplotlib, Seaborn
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='background:white; padding:25px; border-radius:15px;
                    box-shadow:0 4px 15px rgba(0,0,0,0.08);'>
            <div style='font-size:18px; font-weight:700; color:#0f3460;
                        margin-bottom:15px;'>📋 Tabular Dataset</div>
            <div style='font-size:14px; color:#444; line-height:2;'>
                📊 Total Patients: 5,110<br>
                📋 Features: 12 clinical<br>
                🎯 Target: Stroke (0/1)<br>
                ⚖️ Balance: 95% No Stroke / 5% Stroke<br>
                🔧 Missing: BMI (201 values)<br>
                📌 Source: Kaggle (fedesoriano)
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background:white; padding:25px; border-radius:15px;
                    box-shadow:0 4px 15px rgba(0,0,0,0.08);'>
            <div style='font-size:18px; font-weight:700; color:#0f3460;
                        margin-bottom:15px;'>🧠 Brain CT Dataset</div>
            <div style='font-size:14px; color:#444; line-height:2;'>
                🖼️ Total Images: 2,501<br>
                ✅ Normal: 1,551 images<br>
                ⚠️ Stroke: 950 images<br>
                📐 Image Size: 64x64 pixels<br>
                🎨 Format: Grayscale<br>
                📌 Source: Kaggle (afridirahman)
            </div>
        </div>""", unsafe_allow_html=True)