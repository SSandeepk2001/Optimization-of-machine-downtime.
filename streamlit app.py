import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
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
        df = pd.read_csv("OMD-Dataset.csv")
        if 'Day' not in df.columns or 'Month' not in df.columns or 'Year' not in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Day'] = df['Date'].dt.day
            df['Month'] = df['Date'].dt.month
            df['Year'] = df['Date'].dt.year
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

def preprocess_labels(y):
    if y.dtype == 'O':
        return y.map({'No_Machine_Failure': 0, 'Machine_Failure': 1})
    return y

def prepare_input_for_model(input_df, model):
    model_features = model.feature_names_in_
    for feature in model_features:
        if feature not in input_df.columns:
            input_df[feature] = 0
    input_df = input_df[model_features]
    return input_df

def calculate_kpis(df, model):
    if 'Downtime' in df.columns:
        try:
            y_true = preprocess_labels(df['Downtime'])
            X = prepare_input_for_model(df.copy(), model)
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
    else:
        st.warning("⚠️ 'Downtime' column missing.")
        return {}

# =================== Streamlit App ===================

st.set_page_config(page_title="⚙️ Machine Downtime Dashboard", layout="wide", page_icon="🔧")

st.markdown("""
    <style>
    body {background-color: #f5f7fa;}
    .main {background-color: #f0f2f6; border-radius: 20px; padding: 20px;}
    h1, h2, h3, h4 {color: #27374D;}
    </style>
    """, unsafe_allow_html=True)

lottie_gear = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_tll0j4bb.json")
model = load_model()
df = load_data()

with st.sidebar:
    st.title("🔧 Navigation Panel")
    app_mode = st.radio("🚀 Go to Section", ["📊 Data Visualization", "🔮 Predict Downtime", "📈 PowerBI Dashboard"])
    st.divider()
    st.caption("Made with ❤️ by Sandeep Kumar")

col1, col2 = st.columns([1,4])
with col1:
    st_lottie(lottie_gear, speed=1, height=150)
with col2:
    st.title("⚙️ Machine Downtime Optimization Dashboard")
    st.subheader("Predict, Visualize, and Celebrate Efficiency 🚀✨")

st.divider()

if app_mode == "📊 Data Visualization":
    st.header("📊 Data Insights")

    if not df.empty:
        st.subheader("📂 Sample Dataset")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("🚀 KPI Metrics")
        kpis = calculate_kpis(df, model)
        if kpis:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy 📈", f"{kpis['Accuracy']*100:.2f}%")
            col2.metric("Precision 🎯", f"{kpis['Precision']*100:.2f}%")
            col3.metric("Recall 🔎", f"{kpis['Recall']*100:.2f}%")
            col4.metric("F1 Score 🧜‍♀️", f"{kpis['F1 Score']*100:.2f}%")

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

        st.success("✅ Visualization Complete! 🎉")

    else:
        st.warning("⚠️ Dataset not found or empty.")

elif app_mode == "🔮 Predict Downtime":
    st.header("🔮 Make Predictions")

    if model is not None:
        with st.form(key='prediction_form'):
            st.subheader("📄 Enter Machine Parameters")
            day = st.slider('🗕️ Day of Month', 1, 31, 15)
            month = st.slider('🗓️ Month', 1, 12, 6)
            year = st.slider('🗖️ Year', 2020, 2030, 2025)
            submit_button = st.form_submit_button(label='Predict Now 🚀')

        if submit_button:
            try:
                input_data_dict = {
                    'Day': day,
                    'Month': month,
                    'Year': year,
                }
                input_df = pd.DataFrame([input_data_dict])
                input_prepared = prepare_input_for_model(input_df, model)

                prediction = model.predict(input_prepared)
                prediction_proba = model.predict_proba(input_prepared)
                confidence = np.max(prediction_proba) * 100

                if prediction[0] == 1:
                    st.error(f'⚠️ Machine Likely to Downtime! (Confidence: {confidence:.2f}%)')
                else:
                    st.success(f'✅ Machine Operating Smoothly! (Confidence: {confidence:.2f}%)')
                    st.balloons()

            except Exception as e:
                st.error(f"Prediction error: {e}")

        st.divider()

        st.subheader("📅 Bulk Upload for Multiple Predictions")
        uploaded_file = st.file_uploader("📄 Upload your CSV file", type=['csv'])

        if uploaded_file:
            try:
                bulk_data = pd.read_csv(uploaded_file)
                input_prepared = prepare_input_for_model(bulk_data.copy(), model)

                preds = model.predict(input_prepared)
                pred_probs = model.predict_proba(input_prepared)

                bulk_data['Downtime Prediction'] = preds
                bulk_data['Confidence (%)'] = np.max(pred_probs, axis=1) * 100

                st.success("✅ Bulk Prediction Successful!")
                st.dataframe(bulk_data)

                csv = bulk_data.to_csv(index=False, encoding='utf-8-sig')
                st.download_button("📅 Download Predictions", data=csv, file_name='bulk_predictions.csv', mime='text/csv')

            except Exception as e:
                st.error(f"Bulk prediction error: {e}")

    else:
        st.warning("⚠️ Model not loaded properly. Check your file.")

elif app_mode == "📈 PowerBI Dashboard":
    st.header("📈 PowerBI Integrated Dashboard")

    powerbi_report_url = "https://app.powerbi.com/view?r=YOUR_REPORT_ID"  # Replace with your report link

    st.components.v1.html(f"""
        <iframe title="PowerBI Report" width="100%" height="600px"
        src="{powerbi_report_url}" frameborder="0" allowFullScreen="true"></iframe>
    """, height=650)

    st.success("✅ PowerBI Dashboard Loaded Successfully!")
