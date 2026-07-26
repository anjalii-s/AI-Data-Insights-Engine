import pandas as pd
import streamlit as st

from groq import Groq
from google import genai
from google.genai import types

# --------------------------------------------------------
# Page Configuration
# --------------------------------------------------------

st.set_page_config(
    page_title="AI Data Analyst & Executive Insights Engine",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Data Analyst & Executive Insights Engine")

st.write(
    """
Upload any business dataset and generate executive-level insights using
either **Google Gemini** or **Groq Llama**.
"""
)

# --------------------------------------------------------
# Sidebar
# --------------------------------------------------------

with st.sidebar:

    st.header("⚙ Configuration")

    provider = st.radio(
        "Choose AI Provider",
        ["Gemini", "Groq"]
    )

    api_key = st.text_input(
        f"Enter {provider} API Key",
        type="password"
    )

    if provider == "Gemini":
        st.markdown(
            "Get a Gemini API Key:\nhttps://aistudio.google.com/"
        )

    else:
        st.markdown(
            "Get a Groq API Key:\nhttps://console.groq.com/"
        )

# --------------------------------------------------------
# Upload Dataset
# --------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file, encoding="latin1")

    except:

        df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.divider()

    if st.button("🚀 Generate Executive Report"):

        if api_key.strip() == "":

            st.warning("Please enter your API key.")

            st.stop()

        with st.spinner("Analyzing dataset..."):

            try:

                # --------------------------------------
                # Prepare dataset summary
                # --------------------------------------

                dataset_info = f"""

Dataset Shape:
Rows = {df.shape[0]}
Columns = {df.shape[1]}

Column Names:
{list(df.columns)}

Data Types:
{df.dtypes.astype(str).to_string()}

Missing Values:
{df.isnull().sum().to_string()}

Duplicate Rows:
{df.duplicated().sum()}

Statistical Summary:

{df.describe(include="all").fillna("").to_string()[:5000]}

"""

                prompt = f"""
You are a Senior Data Scientist and Business Intelligence Consultant.

Analyze the following dataset summary.

{dataset_info}

Generate a detailed executive report.

The report must contain:

# Executive Summary

Explain what this dataset likely represents.

# Data Quality Assessment

Discuss:
- Missing values
- Duplicate rows
- Potential issues
- Data completeness

# Key Insights

Provide at least FIVE important findings.

# Business Trends

Explain observable trends.

# Risks

Mention possible risks.

# Opportunities

Mention business opportunities.

# Recommended KPIs

Suggest KPIs executives should monitor.

# Actionable Recommendations

Provide at least FIVE recommendations.

# Conclusion

Summarize the business value.

Use professional language suitable for CEOs and business executives.
"""

                # =====================================================
                # GROQ
                # =====================================================

                if provider == "Groq":

                    client = Groq(api_key=api_key)

                    response = client.chat.completions.create(

                        model="llama-3.3-70b-versatile",

                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert Data Scientist."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],

                        temperature=0.3,

                        max_tokens=1800,
                    )

                    report = response.choices[0].message.content

                # =====================================================
                # GEMINI
                # =====================================================

                else:

                    client = genai.Client(
                        api_key=api_key,
                        http_options=types.HttpOptions(
                            api_version="v1"
                        )
                    )

                    response = client.models.generate_content(

                        model="gemini-2.5-flash",

                        contents=prompt,
                    )

                    report = response.text

                st.success("Analysis Completed!")

                st.markdown("---")

                st.markdown("# 📈 Executive AI Report")

                st.markdown(report)

            except Exception as e:

                st.error(f"API Error:\n\n{e}")
