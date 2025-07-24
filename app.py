import streamlit as st
import pandas as pd
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- Configuration ---
st.set_page_config(
    page_title="Agentic AI Data Analyst",
    page_icon="📊",
    layout="wide"
)

# --- Google Gemini API Configuration ---
try:
    # Try to get the key from Streamlit's secrets manager
    api_key = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    # If not found, prompt the user for their key
    st.warning("Google API Key not found. Please enter it below to proceed.")
    api_key = st.text_input("Enter your Google API Key:", type="password")

if api_key:
    genai.configure(api_key=api_key)
else:
    st.stop() # Stop the app if no API key is available

# --- Helper Functions (The Agent's "Tools") ---

def get_data_summary(df):
    """Generates a text summary and samples of the dataframe."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    summary_text = "### 1. Data Overview & Random Samples\n\n"
    summary_text += f"**Shape of the dataset:** {df.shape[0]} rows and {df.shape[1]} columns.\n\n"
    summary_text += "**Column Information & Data Types:**\n```\n" + info_str + "\n```\n\n"
    summary_text += "**Random Samples (up to 5 per column):**\n"
    
    for col in df.columns:
        # Ensure we don't sample more than the available unique values
        sample_count = min(5, len(df[col].dropna().unique()))
        if sample_count > 0:
            samples = df[col].dropna().sample(sample_count).tolist()
            summary_text += f"- **{col}:** `{samples}`\n"
        else:
            summary_text += f"- **{col}:** `(No data to sample)`\n"
            
    return summary_text

def run_data_quality_check(df):
    """Performs a data quality check on the dataframe."""
    dq_report = "### 2. Data Quality Report\n\n"
    
    # Missing Values
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing_values, 'Percentage (%)': missing_percentage})
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)
    
    if not missing_df.empty:
        dq_report += "**Missing Values:**\n"
        dq_report += missing_df.to_markdown() + "\n\n"
    else:
        dq_report += "**Missing Values:**\nNo missing values found. Great!\n\n"
        
    # Outlier Detection (simple IQR method for numeric columns)
    dq_report += "**Potential Outliers (using IQR method):**\n"
    numeric_cols = df.select_dtypes(include=['number']).columns
    outlier_found = False
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if not outliers.empty:
                dq_report += f"- Column **'{col}'**: Found {len(outliers)} potential outliers.\n"
                outlier_found = True
        if not outlier_found:
            dq_report += "No significant outliers detected in numeric columns.\n\n"
    else:
        dq_report += "No numeric columns available for outlier detection.\n\n"


    # Data Format & Type Check
    dq_report += "**Data Format & Type Check:**\n"
    dq_report += "General data types seem appropriate based on initial load. Further validation might be needed for columns with 'object' type if they are expected to be numeric or dates.\n"
    
    return dq_report

def get_interesting_questions(df_head):
    """Generates interesting questions using the LLM."""
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    
    prompt_template = PromptTemplate(
        input_variables=['df_head'],
        template="""
        You are an expert data analyst. Based on the following data snippet (first 5 rows), generate 10 interesting and actionable business questions we could answer with this dataset.
        For each question, briefly explain why it is valuable to investigate.

        Data Snippet:
        {df_head}

        Your response should be formatted clearly in markdown.
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run(df_head=df_head.to_string())
    return "### 3. Interesting Questions to Explore\n\n" + response

def generate_visualizations(df):
    """Generates basic visualizations for numeric and categorical columns."""
    st.write("### 4. Basic Data Visualizations")
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    if len(numeric_cols) > 0:
        st.write("#### Histograms for Numeric Columns")
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f'Distribution of {col}')
            st.pyplot(fig)
    else:
        st.info("No numeric columns found for histograms.")

    if len(categorical_cols) > 0:
        st.write("#### Bar Charts for Categorical Columns (Top 10 categories)")
        for col in categorical_cols:
            if df[col].nunique() > 1:
                fig, ax = plt.subplots()
                top_10 = df[col].value_counts().nlargest(10)
                sns.barplot(x=top_10.index, y=top_10.values, ax=ax)
                ax.set_title(f'Frequency of Top 10 in {col}')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
    else:
        st.info("No categorical columns found for bar charts.")


# --- Streamlit App UI ---
st.title("📊 Agentic AI Data Analyst")
st.markdown("""
Welcome! I'm an AI agent designed to help you with initial data analysis. 
Upload your Excel or CSV file, and I'll perform a preliminary analysis covering:
1.  **Data Summary & Samples:** Understand the structure and content.
2.  **Data Quality Check:** Find missing values and potential outliers.
3.  **Insightful Questions:** Suggest key business questions the data can answer.
4.  **Visualizations:** Generate basic plots to see the distributions.
""")

# FIX 1: Accept both .csv and .xlsx files
uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file is not None:
    try:
        # FIX 2: Check the file extension to use the correct pandas function
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success("File uploaded and loaded successfully!")
        
        st.write("### Raw Data Preview")
        st.dataframe(df.head())
        
        if st.button("🚀 Start Analysis", key="start_analysis_button"):
            # --- Agentic Chain Execution ---
            with st.spinner("Step 1/4: Generating data summary and samples..."):
                summary_report = get_data_summary(df)
                st.markdown(summary_report)

            with st.spinner("Step 2/4: Running data quality checks..."):
                quality_report = run_data_quality_check(df)
                st.markdown(quality_report)
                
            with st.spinner("Step 3/4: Brainstorming insightful questions with Gemini..."):
                questions_report = get_interesting_questions(df.head())
                st.markdown(questions_report)

            with st.spinner("Step 4/4: Creating visualizations..."):
                generate_visualizations(df)
            
            st.balloons()
            st.success("Analysis Complete!")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

