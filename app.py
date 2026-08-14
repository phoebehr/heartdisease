import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
import joblib

# ---------------------------------------------------------
# 1. Page Config & Setup
# ---------------------------------------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.title("🫀 Heart Disease Prediction Dashboard")
st.write("Compare pre-trained ML models on the Heart Disease dataset.")

# Helper function to load saved pickle files
@st.cache_resource
def load_model(filename):
    # Try joblib first, as it's the standard for scikit-learn
    try:
        return joblib.load(filename)
    except Exception:
        # Fallback to standard pickle just in case
        with open(filename, "rb") as f:
            return pickle.load(f)

@st.cache_data
def load_data():
    X_test = joblib.load("X_test.pkl")
    y_test = joblib.load("y_test.pkl")
    return X_test, y_test

# Load test datasets
X_test, y_test = load_data()

# ---------------------------------------------------------
# 2. Sidebar Model Selector
# ---------------------------------------------------------
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ("KNN", "SVM", "ANN")
)

# Map selection to corresponding pickle file
model_files = {
    "KNN": "knn_model.pkl",
    "SVM": "svm_model.pkl",
    "ANN": "ann_model.pkl"
}

# Load the selected model
current_model = load_model(model_files[model_choice])

# ---------------------------------------------------------
# 3. Model Performance Display
# ---------------------------------------------------------
st.subheader(f"📊 Performance Evaluation: {model_choice}")

# Generate predictions on test set
y_pred = current_model.predict(X_test)

# If the model returns probability or 2D array (like TensorFlow/Keras ANN), flatten or argmax it
if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
    y_pred = np.argmax(y_pred, axis=1)
elif len(y_pred.shape) > 1:
    y_pred = (y_pred > 0.5).astype(int).flatten()

acc = accuracy_score(y_test, y_pred)

col1, col2 = st.columns([1, 1])

with col1:
    st.metric(label=f"{model_choice} Test Accuracy", value=f"{acc * 100:.2f}%")
    st.write("### Test Features Sample")
    st.dataframe(X_test.head(5) if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test).head(5))

with col2:
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Reds",
        labels=dict(x="Predicted Class", y="Actual Class"),
        title=f"Confusion Matrix ({model_choice})"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 4. Single Sample Inference / Prediction
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔮 Run Single Sample Test")

sample_idx = st.number_input("Select Test Sample Index", min_value=0, max_value=len(X_test)-1, value=0)

if isinstance(X_test, pd.DataFrame):
    sample_data = X_test.iloc[[sample_idx]]
else:
    sample_data = X_test[sample_idx:sample_idx+1]

pred_val = current_model.predict(sample_data)

if len(pred_val.shape) > 1 and pred_val.shape[1] > 1:
    pred_class = np.argmax(pred_val, axis=1)[0]
elif len(pred_val.shape) > 1:
    pred_class = int(pred_val[0][0] > 0.5)
else:
    pred_class = pred_val[0]

actual_class = y_test.iloc[sample_idx] if isinstance(y_test, (pd.Series, pd.DataFrame)) else y_test[sample_idx]

st.info(f"**Predicted Class:** {pred_class} | **Actual Class:** {actual_class}")