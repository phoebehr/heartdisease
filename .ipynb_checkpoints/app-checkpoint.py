import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------------------------
# 1. Page Config & CSS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    .main-title {
        color: #e63946;
        text-align: center;
        font-weight: 800;
        font-size: 2.3rem;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #4a5568;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #e63946;
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.6rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #d62828;
        box-shadow: 0 4px 12px rgba(230, 57, 70, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Model & Data Loaders
# ---------------------------------------------------------
@st.cache_resource
def load_model(filename):
    try:
        return joblib.load(filename)
    except Exception:
        with open(filename, "rb") as f:
            return pickle.load(f)

@st.cache_data
def load_test_data():
    try:
        X_test = joblib.load("X_test.pkl")
        y_test = joblib.load("y_test.pkl")
        scaler = joblib.load('scaler.pkl')
        scaled_columns = joblib.load('scaled_columns.pkl')
        return X_test, y_test, scaler, scaled_columns
    except Exception:
        return None, None, None, None

model_files = {
    "KNN (K-Nearest Neighbors)": "knn_model.pkl",
    "SVM (Support Vector Machine)": "svm_model.pkl",
    "ANN (Artificial Neural Network)": "ann_model.pkl"
}

# Unpack global variables
X_test, y_test, scaler, scaled_columns = load_test_data()

# ---------------------------------------------------------
# 3. Automatic Feature Engineering & Converter
# ---------------------------------------------------------
def transform_user_input(user_dict, expected_features, scaler=None, scaled_columns=None):
    """
    Transforms raw user questionnaire inputs into the preprocessed 
    engineered feature format expected by the pre-trained model.
    """
    df = pd.DataFrame([user_dict])
    
    # 1. Base numeric variables
    df_transformed = pd.DataFrame()
    df_transformed['age'] = df['age']
    df_transformed['trestbps'] = df['trestbps']
    df_transformed['chol'] = df['chol']
    df_transformed['thalch'] = df['thalch']
    df_transformed['oldpeak'] = df['oldpeak']
    df_transformed['ca'] = df['ca']
    
    # 2. Derived engineered features
    df_transformed['bp_age_ratio'] = df['trestbps'] / (df['age'] + 1e-5)
    df_transformed['predicted_max_hr'] = 220.0 - df['age']
    df_transformed['hr_reserve'] = df_transformed['predicted_max_hr'] - df['thalch']
    df_transformed['oldpeak_x_flat_slope'] = np.where(df['slope'] == 'flat', df['oldpeak'], 0.0)
    df_transformed['vessel_thal_severity'] = df['ca'] + np.where(df['thal'] != 'normal', 1.0, 0.0)
    
    df_transformed['risk_flag_count'] = (
        (df['chol'] > 200).astype(int) + 
        (df['trestbps'] > 130).astype(int) + 
        (df['exang'] == True).astype(int)
    )

    # Safely perform scaling if scaler and scaled_columns are loaded
    if scaler is not None and scaled_columns is not None:
        cols_to_scale = [c for c in scaled_columns if c in df_transformed.columns]
        if cols_to_scale:
            df_transformed[cols_to_scale] = scaler.transform(df_transformed[cols_to_scale])
    
    # Missing value indicators
    df_transformed['data_was_missing'] = 0
    df_transformed['ca_was_missing'] = 0
    df_transformed['slope_was_missing'] = 0
    df_transformed['chol_category_missing'] = 0
    
    # Age category dummies
    age_val = user_dict['age']
    df_transformed['age_group_40-50'] = 1 if 40 <= age_val < 50 else 0
    df_transformed['age_group_60-70'] = 1 if 60 <= age_val < 70 else 0
    
    # Categorical dummies
    df_transformed['sex_Male'] = 1 if user_dict['sex'] == 'Male' else 0
    df_transformed['cp_atypical angina'] = 1 if user_dict['cp'] == 'atypical angina' else 0
    df_transformed['cp_non-anginal'] = 1 if user_dict['cp'] == 'non-anginal' else 0
    df_transformed['cp_typical angina'] = 1 if user_dict['cp'] == 'typical angina' else 0
    df_transformed['cp_asymptomatic'] = 1 if user_dict['cp'] == 'asymptomatic' else 0
    
    df_transformed['slope_flat'] = 1 if user_dict['slope'] == 'flat' else 0
    df_transformed['slope_upsloping'] = 1 if user_dict['slope'] == 'upsloping' else 0
    df_transformed['slope_downsloping'] = 1 if user_dict['slope'] == 'downsloping' else 0

    df_transformed['thal_fixed defect'] = 1 if user_dict['thal'] == 'fixed defect' else 0
    df_transformed['thal_reversable defect'] = 1 if user_dict['thal'] == 'reversable defect' else 0
    df_transformed['thal_normal'] = 1 if user_dict['thal'] == 'normal' else 0

    # Align exact columns expected by the pre-trained model
    if expected_features is not None:
        df_transformed = df_transformed.reindex(columns=expected_features, fill_value=0)
        
    return df_transformed

# ---------------------------------------------------------
# 4. Header & Main Navigation Tabs
# ---------------------------------------------------------
st.markdown("<h1 class='main-title'>🫀 Heart Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Interactive clinical questionnaire with real-time machine learning prediction.</p>", unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2 = st.tabs(["📝 Heart Disease Quiz", "📊 Model Evaluation"])

# ---------------------------------------------------------
# TAB 1: User Input Questionnaire & Classification Prediction
# ---------------------------------------------------------
with tab1:
    st.subheader("🫀 Heart Disease Quiz")
    # 1. Model Selection above questionnaire
    st.markdown("##### ⚙️ Select Prediction Model")
    selected_user_model = st.selectbox(
        "Choose Model for Assessment", 
        list(model_files.keys()), 
        key="user_input_model_select"
    )
    
    # Load selected model and display status indication
    user_model = None
    try:
        user_model = load_model(model_files[selected_user_model])
        st.info(f"🤖 **Active Model:** `{selected_user_model}` is loaded and ready for predictions.")
    except Exception:
        st.error(f"⚠️ Unable to load `{selected_user_model}`. Please ensure `.pkl` files exist in working directory.")

    st.markdown("---")

    # Detect expected features for selected model
    expected_features = None
    if user_model is not None and hasattr(user_model, "feature_names_in_"):
        expected_features = list(user_model.feature_names_in_)
    elif X_test is not None and isinstance(X_test, pd.DataFrame):
        expected_features = list(X_test.columns)

    # 2. Clinical Questionnaire
    with st.form("patient_questionnaire"):
        st.markdown("### 📋 Clinical Questionnaire")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**👤 Demographics & Basic Vitals**")
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=55, step=1)
            sex = st.selectbox("Sex", ("Male", "Female"))
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0.0, max_value=250.0, value=130.0)
            chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0.0, max_value=600.0, value=240.0)

        with col2:
            st.markdown("**🩸 Symptoms & ECG**")
            cp = st.selectbox("Chest Pain Type (CP)", ("asymptomatic", "typical angina", "atypical angina", "non-anginal"))
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", (False, True))
            restecg = st.selectbox("Resting ECG Results", ("normal", "st-t abnormality", "lv hypertrophy"))
            thalch = st.number_input("Max Heart Rate Achieved (bpm)", min_value=50.0, max_value=230.0, value=150.0)

        with col3:
            st.markdown("**🏃 Stress Test & Advanced Metrics**")
            exang = st.selectbox("Exercise Induced Angina", (False, True))
            oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("ST Slope", ("flat", "upsloping", "downsloping"))
            ca = st.number_input("Major Vessels Colored (0-3)", min_value=0.0, max_value=3.0, value=0.0, step=1.0)
            thal = st.selectbox("Thalassemia Status", ("normal", "fixed defect", "reversable defect"))

        submit_button = st.form_submit_button(label="🚀 Predict Heart Disease")

    with st.expander("📖 Need Help? Click to view Variable Descriptions & Meanings"):
        st.markdown("""
        | Variable Name | Full Meaning & Clinical Explanation |
        | :--- | :--- |
        | **Age** | Patient's age in years. |
        | **Sex** | Biological sex (`Male` / `Female`). |
        | **Resting Blood Pressure (`trestbps`)** | Resting blood pressure in mm Hg upon admission to the hospital (Normal is typically < 120 mm Hg). |
        | **Serum Cholesterol (`chol`)** | Total cholesterol level in mg/dl (Desirable level is < 200 mg/dl). |
        | **Chest Pain Type (`cp`)** | Chest pain sensation during activity:<br>• **Typical Angina**: Heart-related pain triggered by exertion.<br>• **Atypical Angina**: Chest discomfort not classic for heart pain.<br>• **Non-anginal Pain**: Non-cardiac chest pain.<br>• **Asymptomatic**: No pain felt. |
        | **Fasting Blood Sugar (`fbs`)** | `True` if fasting blood sugar is greater than 120 mg/dl (indicates potential diabetes risk). |
        | **Resting ECG (`restecg`)** | Resting electrocardiogram result (`normal`, `st-t abnormality`, or `lv hypertrophy` / enlarged heart). |
        | **Max Heart Rate Achieved (`thalch`)** | Maximum heart rate reached during strenuous exercise stress testing. |
        | **Exercise Induced Angina (`exang`)** | Whether chest pain occurs specifically during physical exertion (`True` / `False`). |
        | **ST Depression (`oldpeak`)** | ST depression induced by exercise relative to rest (indicates cardiac stress/ischemia). |
        | **ST Slope (`slope`)** | The slope of the peak exercise ST segment (`upsloping` is normal, `flat` or `downsloping` suggests heart distress). |
        | **Major Vessels (`ca`)** | Number of major coronary blood vessels (0–3) visible under fluoroscopy. |
        | **Thalassemia (`thal`)** | Blood flow & heart muscle imaging result (`normal`, `fixed defect` [past damage], `reversable defect` [active blood blockage]). |
        """)

    if submit_button:
        user_dict = {
            'age': float(age),
            'sex': sex,
            'cp': cp,
            'trestbps': float(trestbps),
            'chol': float(chol),
            'fbs': fbs,
            'restecg': restecg,
            'thalch': float(thalch),
            'exang': exang,
            'oldpeak': float(oldpeak),
            'slope': slope,
            'ca': float(ca),
            'thal': thal
        }
        
        processed_input = transform_user_input(
            user_dict, 
            expected_features, 
            scaler=scaler, 
            scaled_columns=scaled_columns
        )
        
        st.markdown("---")
        st.subheader("🔍 Prediction Results")

        if user_model is not None:
            try:
                # Binary classification prediction
                raw_pred = user_model.predict(processed_input)
                
                # Handle 2D output arrays (e.g., Keras/TensorFlow multi-class or probabilities)
                if len(raw_pred.shape) > 1 and raw_pred.shape[1] > 1:
                    pred_class = int(np.argmax(raw_pred, axis=1)[0])
                elif len(raw_pred.shape) > 1:
                    pred_class = int(raw_pred[0][0] > 0.5)
                else:
                    pred_class = int(raw_pred[0])

                # Binary decision output
                if pred_class == 1:
                    st.error("### 🚨 Heart Disease Detected (YES)")
                    st.write(f"The model **`{selected_user_model}`** predicts **Presence of Heart Disease** based on the entered clinical parameters.")
                    st.markdown("• **Recommended Action**: Consult with a qualified cardiologist for further clinical testing and diagnosis.")
                else:
                    st.success("### 🎉 No Heart Disease Detected (NO)")
                    st.write(f"The model **`{selected_user_model}`** predicts **No Presence of Heart Disease** for this clinical sample.")
                    st.markdown("• **Recommended Action**: Continue maintaining healthy lifestyle habits, balanced nutrition, and regular checkups.")

                with st.expander("🔬 View Processed Feature Vector Sent to Model"):
                    st.dataframe(processed_input)

            except Exception as e:
                st.error(f"Prediction Execution Error: {e}")
        else:
            st.error("Model is not loaded. Please select a valid model above.")

# ---------------------------------------------------------
# TAB 2: Model Evaluation Dashboard
# ---------------------------------------------------------
with tab2:
    # 1. Model Selection within evaluation tab
    st.subheader("⚙️ Model Evaluation")
    selected_eval_model = st.selectbox(
        "Select Model to Evaluate", 
        list(model_files.keys()), 
        key="eval_model_select"
    )

    # Indication of selected model
    eval_model = None
    try:
        eval_model = load_model(model_files[selected_eval_model])
        st.success(f"🎯 **Currently Evaluating:** `{selected_eval_model}`")
    except Exception:
        st.error(f"⚠️ Unable to load `{selected_eval_model}`. Please ensure `.pkl` files exist in working directory.")

    st.markdown("---")
    st.subheader(f"📊 Performance Evaluation: {selected_eval_model}")

    if eval_model is not None and X_test is not None and y_test is not None:
        try:
            y_pred = eval_model.predict(X_test)

            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                y_pred = np.argmax(y_pred, axis=1)
            elif len(y_pred.shape) > 1:
                y_pred = (y_pred > 0.5).astype(int).flatten()

            acc = accuracy_score(y_test, y_pred)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.metric(label=f"{selected_eval_model} Test Accuracy", value=f"{acc * 100:.2f}%")
                st.write("### Test Features Sample")
                st.dataframe(X_test.head(5) if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test).head(5))

            with col2:
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    color_continuous_scale="Reds",
                    labels=dict(x="Predicted Class", y="Actual Class"),
                    title=f"Confusion Matrix ({selected_eval_model})"
                )
                st.plotly_chart(fig_cm, use_container_width=True)

        except Exception as e:
            st.error(f"Error calculating model evaluation metrics: {e}")
    else:
        st.warning("Ensure model files and `X_test.pkl` / `y_test.pkl` are available to render performance evaluation.")

    if X_test is not None:
        st.markdown("---")
        with st.expander("📊 View Reference Test Dataset (X_test.pkl)"):
            st.dataframe(X_test.head(10))