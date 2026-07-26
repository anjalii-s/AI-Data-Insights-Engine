import pandas as pd
import streamlit as st
from groq import Groq

# -------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------

st.set_page_config(
    page_title="AI Business Intelligence & Executive Reporting Engine",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Business Intelligence & Executive Reporting Engine")

st.markdown("""
Generate executive-level business insights from any CSV dataset using **Groq Llama 3.3 70B**.

Simply:

1. Enter your Groq API Key
2. Upload a CSV
3. Click **Generate Executive Report**
""")

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

with st.sidebar:

    st.header("⚙ Configuration")

    api_key = st.text_input(
        "Groq API Key",
        type="password"
    )

    st.info(
        "Get your free API key from:\n\nhttps://console.groq.com/"
    )

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------

@st.cache_data
def load_data(file):

    try:
        return pd.read_csv(file, encoding="latin1")

    except:
        return pd.read_csv(file)


uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file:

    df = load_data(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(df.head(), use_container_width=True)

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.divider()

    # -----------------------------
    # DATA SUMMARY
    # -----------------------------

    dataset_summary = f"""

Dataset Shape
--------------
Rows: {df.shape[0]}
Columns: {df.shape[1]}

Column Names
------------
{list(df.columns)}

Data Types
----------
{df.dtypes.astype(str).to_string()}

Missing Values
--------------
{df.isnull().sum().to_string()}

Duplicate Rows
--------------
{df.duplicated().sum()}

Statistical Summary
-------------------
{df.describe(include='all').fillna('').to_string()[:7000]}

"""

    if st.button("🚀 Generate Executive Report"):

        if api_key.strip() == "":
            st.warning("Please enter your Groq API Key.")
            st.stop()

        prompt = f"""
You are a Senior Data Scientist, Business Intelligence Consultant,
and Strategy Advisor.

Below is a dataset profile.

{dataset_summary}

Generate a professional executive report.

The report should contain:

# Executive Summary
Explain what this dataset likely represents.

# Dataset Overview
Summarize dataset characteristics.

# Data Quality Assessment

Discuss:
- Missing values
- Duplicate rows
- Data completeness
- Data reliability
- Potential preprocessing required

# Key Insights

Provide at least FIVE meaningful insights.

# Business Trends

Explain important patterns.

# Risks

Identify operational or business risks.

# Opportunities

Identify areas where value can be created.

# Recommended KPIs

Suggest measurable KPIs.

# Executive Recommendations

Provide at least FIVE actionable recommendations.

# Conclusion

Summarize the overall business value.

Use professional language suitable for CEOs, senior managers,
and business stakeholders.

Avoid simply repeating statistics.

Instead, interpret the data from a business perspective.
"""

        try:

            with st.spinner("Analyzing dataset using Groq AI..."):

                client = Groq(api_key=api_key)

                response = client.chat.completions.create(

                    model="llama-3.3-70b-versatile",

                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert Business Intelligence Consultant."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    temperature=0.3,
                    max_tokens=2000
                )

                report = response.choices[0].message.content

            st.success("Executive Report Generated Successfully!")

            st.markdown("---")

            st.markdown(report)

            st.download_button(
                "📥 Download Report",
                data=report,
                file_name="Executive_Report.md",
                mime="text/markdown"
            )

        except Exception as e:

            st.error(f"Error:\n\n{e}")
