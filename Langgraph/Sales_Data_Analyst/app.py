import streamlit as st
import pandas as pd
import plotly.express as px

from graph import run_query_with_dataframe

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Sales Data Analyst",
    page_icon="📊",
    layout="wide"
)

# ----------------------------
# CSS (Fix white-on-white)
# ----------------------------
st.markdown("""
<style>
[data-testid="stDataFrame"] {
    background-color: white;
    color: black;
}
thead tr th {
    background-color: #f0f2f6;
    color: black;
}
tbody tr td {
    color: black;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Talk to Your Sales Data")
st.caption("LangGraph • DuckDB • Local LLaMA (llama.cpp)")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("📂 Upload CSV")

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ----------------------------
# Load CSV
# ----------------------------
loaded_dfs = {}

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() for c in df.columns]
    loaded_dfs["sales"] = df

# ----------------------------
# Layout
# ----------------------------
col1, col2 = st.columns([2, 1])

# ----------------------------
# Data Preview
# ----------------------------
with col2:
    st.subheader("📋 Data Preview")

    if loaded_dfs:
        df = loaded_dfs["sales"]
        st.metric("Rows", df.shape[0], f"{df.shape[1]} columns")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.info("Upload a CSV file to start.")

# ----------------------------
# Chat + Results
# ----------------------------
with col1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your data…"):
        if not loaded_dfs:
            st.warning("Please upload a CSV file first.")
            st.stop()

        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing…"):

                result = run_query_with_dataframe(
                    question=prompt,
                    dataframes=loaded_dfs
                )

                # ----------------------------
                # Show SQL
                # ----------------------------
                if result.get("sql"):
                    with st.expander("🔍 Show SQL Query"):
                        st.code(result["sql"], language="sql")

                df_result = result.get("data")

                # ----------------------------
                # Show Data
                # ----------------------------
                if df_result is not None and not df_result.empty:
                    st.dataframe(df_result, use_container_width=True)

                    # ----------------------------
                    # Visualization (Plotly ONLY)
                    # ----------------------------
                    chart_type = result.get("chart_type", "bar")
                    numeric_cols = df_result.select_dtypes(include="number").columns.tolist()

                    if len(df_result.columns) >= 2 and numeric_cols:
                        label_col = df_result.columns[0]
                        metric_col = numeric_cols[0]

                        st.subheader("📊 Visualization")

                        if chart_type == "pie":
                            fig = px.pie(
                                df_result,
                                names=label_col,
                                values=metric_col
                            )
                        elif chart_type == "line":
                            fig = px.line(
                                df_result,
                                x=label_col,
                                y=metric_col
                            )
                        else:
                            fig = px.bar(
                                df_result,
                                x=label_col,
                                y=metric_col
                            )

                        st.plotly_chart(fig, use_container_width=True)

                # ----------------------------
                # Text Summary
                # ----------------------------
                if result.get("answer"):
                    st.markdown(result["answer"])
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result["answer"]}
                    )
