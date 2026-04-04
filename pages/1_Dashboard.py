import streamlit as st
import pandas as pd
import os
import sys

# Import the evaluation function from your root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ragas_evaluation import run_evaluation

st.set_page_config(page_title="Evaluation Dashboard", layout="wide")
st.title("📊 RAG Pipeline Evaluation Metrics")

# --- NEW: Run Evaluation directly from UI ---
st.markdown("### Run New Evaluation")
st.caption("Clicking this will run your test dataset through the LLM and grade the answers. This takes about 30-60 seconds.")
if st.button("🚀 Run Ragas Evaluation Now", type="primary"):
    with st.spinner("Running Evaluation Pipeline... Please wait, do not refresh the page."):
        try:
            run_evaluation()
            st.success("Evaluation complete! The dashboard has been updated.")
            # Clear cache and rerun to show new data
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred during evaluation: {e}")

st.divider()

# --- EXISTING: Display the Dashboard ---
csv_path = "ragas_evaluation_results.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    
    st.markdown("### System Averages")
    avg_faithfulness = df['faithfulness'].mean() if 'faithfulness' in df.columns else 0
    avg_relevancy = df['answer_relevancy'].mean() if 'answer_relevancy' in df.columns else 0
    avg_clarity = df['clarity_score_out_of_5'].mean() if 'clarity_score_out_of_5' in df.columns else 0
    avg_practicality = df['practicality_score_out_of_5'].mean() if 'practicality_score_out_of_5' in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Factual Faithfulness", f"{avg_faithfulness:.2f} / 1.0")
    col2.metric("Answer Relevancy", f"{avg_relevancy:.2f} / 1.0")
    col3.metric("Explanation Clarity", f"{avg_clarity:.1f} / 5.0")
    col4.metric("Practicality", f"{avg_practicality:.1f} / 5.0")
    
    st.divider()
    
    st.markdown("### Question Breakdown")
    st.dataframe(
        df,
        column_config={
            "user_input": st.column_config.TextColumn("Question", width="medium"),
            "judge_reasoning": st.column_config.TextColumn("Judge's Reasoning", width="large"),
            "faithfulness": st.column_config.NumberColumn("Faithfulness", format="%.2f"),
            "answer_relevancy": st.column_config.NumberColumn("Relevancy", format="%.2f")
        },
        hide_index=True,
        use_container_width=True
    )
    
else:
    st.warning("⚠️ No evaluation data found. Click the button above to run the first evaluation.")