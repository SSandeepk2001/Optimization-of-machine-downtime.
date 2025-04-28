import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import seaborn as sns
import requests
from streamlit_lottie import st_lottie
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =================== Helper Functions ===================

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

@st.cache_resource
def load_model():
    try:
        model = joblib.load("downtime_predictor.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Dataset (Optimization machine of downtime).csv")
        df = df.drop(columns=[col for col in ['Year', 'Month', 'Day'] if col in df.columns])
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

def calculate_kpis(df, model):
    try:
        feature_cols = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Feature mismatch during KPI calculation: Missing columns: {missing_cols}")
            return {}

        X = df[feature_cols]
        if 'Downtime' not in df.columns:
            st.warning("⚠️ 'Downtime' column missing.")
            return {}

        y_true = df['Downtime']
        y_pred = model.predict(X)
        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1 Score': f1_score(y_true, y_pred, zero_division=0)
        }
    except Exception as e:
        st.error(f"Feature mismatch during KPI calculation: {e}")
        return {}

# =================== Streamlit App ===================

st.set_page_config(page_title="⚙️ Machine Downtime Dashboard", layout="wide", page_icon="🔧")

# ---- Custom CSS ----
st.markdown("""
    <style>
    body {background-color: #f5f7fa;}
    .main {background-color: #f0f2f6; border-radius: 20px; padding: 20px;}
    h1, h2, h3, h4 {color: #27374D;}
    </style>
    """, unsafe_allow_html=True)

# Load assets
lottie_gear = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_tll0j4bb.json")
model = load_model()
df = load_data()

# Sidebar
with st.sidebar:
    st.title("🔧 Navigation Panel")
    app_mode = st.radio("🚀 Go to Section", ["📊 Data Visualization", "🔮 Predict Downtime"])
    st.divider()
    st.caption("Made with ❤️ by Sandeep Kumar")

# Header Section
col1, col2 = st.columns([1,4])
with col1:
    st_lottie(lottie_gear, speed=1, height=150)
with col2:
    st.title("⚙️ Machine Downtime Optimization Dashboard")
    st.subheader("Predict, Visualize, and Celebrate Efficiency 🚀✨")

st.divider()

# =================== Data Visualization ===================
if app_mode == "📊 Data Visualization":
    st.header("📊 Data Insights")

    if not df.empty:
        st.subheader("📂 Sample Dataset")
        st.dataframe(df.head(10), use_container_width=True)

        # 🆕 Data Summary
        st.subheader("📜 Data Summary")
        st.dataframe(df.describe(), use_container_width=True)

        st.subheader("🚀 KPI Metrics")
        kpis = calculate_kpis(df, model)
        if kpis:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy 📈", f"{kpis['Accuracy']*100:.2f}%")
            col2.metric("Precision 🎯", f"{kpis['Precision']*100:.2f}%")
            col3.metric("Recall 🔎", f"{kpis['Recall']*100:.2f}%")
            col4.metric("F1 Score 🧠", f"{kpis['F1 Score']*100:.2f}%")

        st.subheader("🛠️ Missing Data Overview")
        fig = px.imshow(df.isnull(), title="Missing Value Heatmap", color_continuous_scale='reds')
        st.plotly_chart(fig)

        st.subheader("⚡ Downtime Distribution")
        if 'Downtime' in df.columns:
            fig = px.histogram(df, x='Downtime', color='Downtime', color_discrete_sequence=['#FFA07A', '#20B2AA'])
            st.plotly_chart(fig)

        st.subheader("🔗 Feature Correlation Matrix")
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='teal')
        st.plotly_chart(fig)

        # 🆕 Feature Importance
        if hasattr(model, 'feature_importances_'):
            st.subheader("🌟 Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': model.feature_names_in_,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='viridis')
            st.plotly_chart(fig)

        st.success("✅ Visualization Complete! 🎉")

    else:
        st.warning("⚠️ Dataset not found or empty.")

# =================== Prediction Section ===================
elif app_mode == "🔮 Predict Downtime":
    st.header("🔮 Make Predictions")

    if model is not None:
        with st.form(key='prediction_form'):
            st.subheader("📄 Enter Machine Parameters")
            feature_inputs = {}
            if hasattr(model, 'feature_names_in_'):
                for feature in model.feature_names_in_:
                    feature_inputs[feature] = st.text_input(f"{feature}", value="0")
            submit_button = st.form_submit_button(label='Predict Now 🚀')

        if submit_button:
            try:
                input_array = np.array([[float(feature_inputs[f]) for f in model.feature_names_in_]])
                prediction = model.predict(input_array)
                prediction_proba = model.predict_proba(input_array)
                confidence = np.max(prediction_proba) * 100

                if prediction[0] == 1:
                    st.error(f'⚠️ Machine Likely to Downtime! (Confidence: {confidence:.2f}%)')
                else:
                    st.success(f'✅ Machine Operating Smoothly! (Confidence: {confidence:.2f}%)')
                    st.balloons()

            except Exception as e:
                st.error(f"Prediction error: {e}")

        st.divider()

        st.subheader("📥 Bulk Upload for Multiple Predictions")
        uploaded_file = st.file_uploader("📄 Upload your CSV file", type=['csv'])

        if uploaded_file:
            try:
                bulk_data = pd.read_csv(uploaded_file)
                feature_cols = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []
                bulk_data = bulk_data.drop(columns=[col for col in ['Year', 'Month', 'Day'] if col in bulk_data.columns])
                if all(col in bulk_data.columns for col in feature_cols):
                    X_bulk = bulk_data[feature_cols]
                    preds = model.predict(X_bulk)
                    pred_probs = model.predict_proba(X_bulk)
                    bulk_data['Downtime Prediction'] = preds
                    bulk_data['Confidence (%)'] = np.max(pred_probs, axis=1) * 100

                    st.success("✅ Bulk Prediction Successful!")
                    st.dataframe(bulk_data)

                    csv = bulk_data.to_csv(index=False)
                    st.download_button("📥 Download Predictions", data=csv, file_name='bulk_predictions.csv', mime='text/csv')
                else:
                    st.error(f"Uploaded CSV must contain columns: {feature_cols}")
            except Exception as e:
                st.error(f"Bulk prediction error: {e}")

    else:
        st.warning("⚠️ Model not loaded properly. Check your file.")

# =================== Optional PowerBI Integration ===================

st.divider()

st.subheader("📊 PowerBI Dashboard")
placeholder_text = "Update the embed link to view PowerBI dashboard here."
st.info(placeholder_text)

# If you have PowerBI public link ready, uncomment below:
# st.markdown("""
#     <iframe title="PowerBI Dashboard" width="100%" height="600" 
#     src="https://app.powerbi.com/view?r=YOUR_PBI_EMBED_LINK_HERE" frameborder="0" allowFullScreen="true"></iframe>
# """, unsafe_allow_html=True)
