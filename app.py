import pandas as pd
import streamlit as st
from google import genai

# Page Configuration
st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

st.title("📊 AI Data Analyst & Executive Insights Engine")
st.write("Upload any business dataset (CSV) to generate C-suite strategic insights powered by Gemini.")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Gemini API Key:", type="password")
    st.markdown("[Get a free Gemini API Key](https://aistudio.google.com/)")

# File Uploader
uploaded_file = st.file_uploader("Upload your Kaggle CSV file", type=["csv"])

if uploaded_file:
    # Read the dataset safely
    try:
        df = pd.read_csv(uploaded_file, encoding="latin1")
        st.subheader("Data Preview")
        st.dataframe(df.head(5))
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")

    # Generate Report Button
    if st.button("🚀 Generate AI Executive Report"):
        if not api_key or not api_key.strip():
            st.warning("⚠️ Please enter your Gemini API key in the sidebar on the left.")
        else:
            with st.spinner("Analyzing dataset with Gemini..."):
                try:
                    # Initialize client with user's clean API Key
                    client = genai.Client(api_key=api_key.strip())

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

                    # Using 'gemini-1.5-flash' - guaranteed stable model for free keys
                    response = client.models.generate_content(
                        model="gemini-1.5-flash",
                        contents=prompt,
                    )

                    st.markdown("---")
                    st.markdown("### 📈 Executive AI Summary")
                    st.markdown(response.text)

                except Exception as err:
                    st.error(f"❌ API Call Failed: {err}")
                    st.info("Tip: Double-check that your Gemini API Key is copied correctly with no extra spaces.")
