import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

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
# NOTE: each model .pkl is now a full sklearn Pipeline (feature selection +
# scaling + classifier combined) — it expects the RAW, unselected, unscaled
# feature set as input, and handles selection/scaling internally. Do NOT
# apply any manual scaling or column-trimming before calling .predict().

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
        X_test_raw = joblib.load("X_test_raw.pkl")
        y_test = joblib.load("y_test.pkl")
        return X_test_raw, y_test
    except Exception:
        return None, None

model_files = {
    "KNN (K-Nearest Neighbors)": "knn_model.pkl",
    "SVM (Support Vector Machine)": "svm_model.pkl",
    "ANN (Artificial Neural Network)": "ann_model.pkl"
}

# Unpack global variables
X_test_raw, y_test = load_test_data()

# ---------------------------------------------------------
# 3. Automatic Feature Engineering & Converter
# ---------------------------------------------------------
def transform_user_input(user_dict, expected_features):
    """
    Transforms raw user questionnaire inputs into the RAW engineered feature
    format expected by the pre-trained pipeline (feature selection and
    scaling happen automatically inside the pipeline's .predict() call —
    this function must NOT scale or trim columns itself).
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

    # 2. Missing value indicators — always 0, live user input is never "missing"
    df_transformed['data_was_missing'] = 0
    df_transformed['ca_was_missing'] = 0
    df_transformed['thal_was_missing'] = 0
    df_transformed['slope_was_missing'] = 0

    # 3. Derived engineered features — matching training formulas exactly
    df_transformed['risk_flag_count'] = (
        (df['fbs'] == True).astype(int) +
        (df['exang'] == True).astype(int) +
        (df['oldpeak'] > 1).astype(int) +
        (df['ca'] > 0).astype(int)
    )
    df_transformed['predicted_max_hr'] = 220.0 - df['age']
    df_transformed['oldpeak_x_flat_slope'] = np.where(df['slope'] == 'flat', df['oldpeak'], 0.0)
    df_transformed['hr_reserve'] = df_transformed['predicted_max_hr'] - df['thalch']
    df_transformed['bp_age_ratio'] = df['trestbps'] / (df['age'] + 1e-5)
    # vessel_thal_severity = ca + 2 ONLY for 'reversable defect' (not any non-normal thal)
    df_transformed['vessel_thal_severity'] = df['ca'] + np.where(
        df['thal'] == 'reversable defect', 2.0, 0.0
    )

    # 4. sex — reference category dropped during training: 'Female'
    df_transformed['sex_Male'] = 1 if user_dict['sex'] == 'Male' else 0
    df_transformed['sex_missing'] = 0

    # 5. cp — reference category dropped during training: 'asymptomatic'
    df_transformed['cp_atypical angina'] = 1 if user_dict['cp'] == 'atypical angina' else 0
    df_transformed['cp_missing'] = 0
    df_transformed['cp_non-anginal'] = 1 if user_dict['cp'] == 'non-anginal' else 0
    df_transformed['cp_typical angina'] = 1 if user_dict['cp'] == 'typical angina' else 0

    # 6. fbs — reference category dropped during training: False
    df_transformed['fbs_True'] = 1 if user_dict['fbs'] == True else 0
    df_transformed['fbs_missing'] = 0

    # 7. restecg — reference category dropped during training: 'lv hypertrophy'
    df_transformed['restecg_missing'] = 0
    df_transformed['restecg_normal'] = 1 if user_dict['restecg'] == 'normal' else 0
    df_transformed['restecg_st-t abnormality'] = 1 if user_dict['restecg'] == 'st-t abnormality' else 0

    # 8. exang — reference category dropped during training: False
    df_transformed['exang_True'] = 1 if user_dict['exang'] == True else 0
    df_transformed['exang_missing'] = 0

    # 9. slope — reference category dropped during training: 'downsloping'
    df_transformed['slope_flat'] = 1 if user_dict['slope'] == 'flat' else 0
    df_transformed['slope_missing'] = 0
    df_transformed['slope_upsloping'] = 1 if user_dict['slope'] == 'upsloping' else 0

    # 10. thal — reference category dropped during training: 'fixed defect'
    df_transformed['thal_missing'] = 0
    df_transformed['thal_normal'] = 1 if user_dict['thal'] == 'normal' else 0
    df_transformed['thal_reversable defect'] = 1 if user_dict['thal'] == 'reversable defect' else 0

    # 11. chol_category — reference category dropped during training: 'borderline'
    chol_val = user_dict['chol']
    if chol_val < 200:
        chol_cat = 'normal'
    elif chol_val < 240:
        chol_cat = 'borderline'
    else:
        chol_cat = 'high'
    df_transformed['chol_category_high'] = 1 if chol_cat == 'high' else 0
    df_transformed['chol_category_missing'] = 0
    df_transformed['chol_category_normal'] = 1 if chol_cat == 'normal' else 0

    # 12. age_group — reference category dropped during training: '<40'
    age_val = user_dict['age']
    df_transformed['age_group_40-50'] = 1 if 40 <= age_val < 50 else 0
    df_transformed['age_group_50-60'] = 1 if 50 <= age_val < 60 else 0
    df_transformed['age_group_60-70'] = 1 if 60 <= age_val < 70 else 0
    df_transformed['age_group_70+'] = 1 if age_val >= 70 else 0

    # Align exact columns (and order) expected by the pre-trained pipeline.
    # This also safely drops anything accidentally extra and zero-fills
    # anything unexpectedly missing, rather than erroring outright.
    if expected_features is not None:
        df_transformed = df_transformed.reindex(columns=expected_features, fill_value=0)

    return df_transformed

# ---------------------------------------------------------
# 4. Header & Main Navigation Tabs
# ---------------------------------------------------------
st.markdown("<h1 class='main-title'>🫀 Heart Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Interactive clinical questionnaire with real-time machine learning prediction.</p>", unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["📝 Heart Disease Quiz", "📊 Model Evaluation", "📈 Model Comparison"])

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

    # Detect expected features for selected model.
    # Since each model is now a full Pipeline, `feature_names_in_` reflects
    # the RAW columns it was fit on (before internal selection/scaling) —
    # this is exactly what transform_user_input() needs to match.
    expected_features = None
    if user_model is not None and hasattr(user_model, "feature_names_in_"):
        expected_features = list(user_model.feature_names_in_)
    elif X_test_raw is not None and isinstance(X_test_raw, pd.DataFrame):
        expected_features = list(X_test_raw.columns)

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

        processed_input = transform_user_input(user_dict, expected_features)

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

    if eval_model is not None and X_test_raw is not None and y_test is not None:
        try:
            # eval_model is a full Pipeline — pass the RAW test features directly,
            # it selects/scales internally, exactly as it did during training.
            y_pred = eval_model.predict(X_test_raw)

            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                y_pred = np.argmax(y_pred, axis=1)
            elif len(y_pred.shape) > 1:
                y_pred = (y_pred > 0.5).astype(int).flatten()

            acc = accuracy_score(y_test, y_pred)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.metric(label=f"{selected_eval_model} Test Accuracy", value=f"{acc * 100:.2f}%")
                st.write("### Test Features Sample")
                st.dataframe(X_test_raw.head(5) if isinstance(X_test_raw, pd.DataFrame) else pd.DataFrame(X_test_raw).head(5))

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
        st.warning("Ensure model files and `X_test_raw.pkl` / `y_test.pkl` are available to render performance evaluation.")

    if X_test_raw is not None:
        st.markdown("---")
        with st.expander("📊 View Reference Test Dataset (X_test_raw.pkl)"):
            st.dataframe(X_test_raw.head(10))

# ---------------------------------------------------------
# TAB 3: Model Comparison
# ---------------------------------------------------------
with tab3:
    st.subheader("📈 Model Comparison")
    st.write("Side-by-side comparison of Accuracy, Precision, Recall, and F1-score for all three models, evaluated on the same test set.")

    st.markdown("---")

    if X_test_raw is not None and y_test is not None:
        comparison_rows = []
        load_errors = []

        for model_label, filename in model_files.items():
            try:
                comp_model = load_model(filename)
                y_pred = comp_model.predict(X_test_raw)

                # Handle 2D output arrays (e.g., Keras/TensorFlow multi-class or probabilities)
                if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                    y_pred = np.argmax(y_pred, axis=1)
                elif len(y_pred.shape) > 1:
                    y_pred = (y_pred > 0.5).astype(int).flatten()

                comparison_rows.append({
                    "Model": model_label,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "F1-Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                })
            except Exception as e:
                load_errors.append(f"⚠️ Could not evaluate `{model_label}`: {e}")

        for err in load_errors:
            st.error(err)

        if comparison_rows:
            comp_df = pd.DataFrame(comparison_rows).set_index("Model")

            # 1. Summary table — best value per metric highlighted
            st.markdown("#### 📋 Metric Summary")
            styled_df = comp_df.style.format("{:.2%}").highlight_max(
                axis=0, color="#c6f6d5"
            )
            st.dataframe(styled_df, use_container_width=True)

            # 2. Best model per metric, as quick-glance cards
            st.markdown("#### 🏆 Best Model per Metric")
            metric_cols = st.columns(4)
            for col, metric in zip(metric_cols, ["Accuracy", "Precision", "Recall", "F1-Score"]):
                best_model_name = comp_df[metric].idxmax()
                best_value = comp_df[metric].max()
                with col:
                    st.metric(label=metric, value=f"{best_value * 100:.2f}%", delta=best_model_name)

            st.markdown("---")

            # 3. Grouped bar chart across all metrics
            st.markdown("#### 📊 Visual Comparison")
            fig = go.Figure()
            for model_label in comp_df.index:
                fig.add_trace(go.Bar(
                    name=model_label,
                    x=comp_df.columns,
                    y=comp_df.loc[model_label],
                    text=[f"{v:.1%}" for v in comp_df.loc[model_label]],
                    textposition="auto"
                ))
            fig.update_layout(
                barmode="group",
                yaxis_title="Score",
                yaxis_tickformat=".0%",
                legend_title="Model",
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("ℹ️ How these metrics are calculated"):
                st.markdown("""
                - **Accuracy**: overall proportion of correct predictions.
                - **Precision**: of all patients predicted to have heart disease, how many actually did.
                - **Recall**: of all patients who actually have heart disease, how many were correctly identified.
                - **F1-Score**: the harmonic mean of Precision and Recall, balancing both.

                Precision, Recall, and F1-Score use **weighted averaging** across classes, and all four metrics are calculated on the same held-out test set (`X_test_raw.pkl` / `y_test.pkl`) for a fair, apples-to-apples comparison.
                """)
        else:
            st.warning("No models could be evaluated. Please check that all `.pkl` model files exist in the working directory.")
    else:
        st.warning("Ensure `X_test_raw.pkl` and `y_test.pkl` are available to render the model comparison.")