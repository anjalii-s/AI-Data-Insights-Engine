import pandas as pd
import streamlit as st
from google import genai

# Page setup
st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

st.title("📊 AI Data Analyst & Executive Insights Engine")
st.write("Upload any business dataset (CSV) to generate C-suite strategic insights powered by Gemini.")

# Sidebar for user API Key
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Gemini API Key:", type="password")
    st.markdown("[Get a free Gemini API Key](https://aistudio.google.com/)")

# Web file uploader (Replaces Colab's files.upload)
uploaded_file = st.file_uploader("Upload your Kaggle CSV file", type=["csv"])

if uploaded_file and api_key:
    # Read the dataset safely
    df = pd.read_csv(uploaded_file, encoding="latin1")
    
    st.subheader("Data Preview")
    st.dataframe(df.head(5))

    if st.button("🚀 Generate AI Executive Report"):
        with st.spinner("Analyzing dataset with Gemini 2.5 Flash..."):
            # Initialize client with user's key
            client = genai.Client(api_key=api_key)

            # Statistical processing
            columns_list = list(df.columns)
            summary_stats = df.describe(include='all').to_string()

            prompt = f"""
            You are an expert Data Analyst. Analyze this dataset summary and provide:
            1. A high-level executive summary of what this dataset represents.
            2. 3 key patterns or trends based on the statistics.
            3. 3 actionable business recommendations.

            Dataset Columns: {columns_list}
            Statistical Summary:
            {summary_stats[:2000]}
            """

            # Call Gemini API
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )

            st.markdown("---")
            st.markdown("### 📈 Executive AI Summary")
            st.markdown(response.text)

elif not api_key and uploaded_file:
    st.warning("⚠️ Please enter your Gemini API key in the sidebar to proceed.")
