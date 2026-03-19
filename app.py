import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import os
from groq import Groq
from sqlalchemy import create_engine

def get_db_engine():
    url = "postgresql://postgres.tmmixkokqqohbgbzfabj:23188722mr22@aws-1-ap-south-1.pooler.supabase.com:5432/postgres"
    return create_engine(url)

def ask(question):
    import pandas as pd
    engine = get_db_engine()
    segments = pd.read_sql("SELECT segment, COUNT(*) as count FROM rfm_scores GROUP BY segment", engine)
    churn = pd.read_sql("SELECT COUNT(*) as total, ROUND(AVG(churn_probability)::numeric * 100, 2) as avg_churn FROM churn_predictions", engine)
    ltv = pd.read_sql("SELECT ROUND(AVG(predicted_ltv)::numeric, 2) as avg_ltv FROM ltv_predictions", engine)
    db_summary = f"SEGMENTS:\n{segments.to_string(index=False)}\n\nCHURN:\n{churn.to_string(index=False)}\n\nLTV:\n{ltv.to_string(index=False)}"
    client = Groq(api_key="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")
    system_prompt = f"You are a senior data analyst for an e-commerce business with 93357 customers. Answer with specific numbers.\n\nDATA:\n{db_summary}"
    response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": question}], temperature=0.3)
    return response.choices[0].message.content

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="🧠",
    layout="wide"
)

def get_db_engine():
    from sqlalchemy import create_engine
    url = "postgresql://postgres.tmmixkokqqohbgbzfabj:23188722mr22@aws-1-ap-south-1.pooler.supabase.com:5432/postgres"
    return create_engine(url)

@st.cache_data
def load_data():
    engine = get_db_engine()
    rfm = pd.read_sql("SELECT * FROM rfm_scores", engine)
    churn = pd.read_sql("SELECT * FROM churn_predictions", engine)
    ltv = pd.read_sql("SELECT * FROM ltv_predictions", engine)
    return rfm, churn, ltv

rfm, churn, ltv = load_data()

# ── Sidebar ────────────────────────────────────────────
st.sidebar.title("🧠 Customer Intelligence")
st.sidebar.markdown("Built on **93,357** real customers")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🤖 AI Analyst"])

# ── Dashboard Page ─────────────────────────────────────
if page == "📊 Dashboard":
    st.title("Customer Intelligence Dashboard")
    st.markdown("Real-time insights from Olist e-commerce data")

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(rfm):,}")
    col2.metric("Avg Spend", f"R$ {rfm['monetary'].mean():.0f}")
    col3.metric("Churn Risk", f"{churn['churn_probability'].mean()*100:.1f}%")
    col4.metric("Avg LTV", f"R$ {ltv['predicted_ltv'].mean():.0f}")

    st.markdown("---")

    # Segment distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Segments")
        segment_counts = rfm["segment"].value_counts().reset_index()
        segment_counts.columns = ["Segment", "Count"]
        fig = px.bar(
            segment_counts, x="Segment", y="Count",
            color="Segment",
            color_discrete_map={
                "Champions": "#2ecc71",
                "Loyal Customers": "#27ae60",
                "Potential Loyalists": "#f39c12",
                "At Risk": "#e67e22",
                "Lost": "#e74c3c"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn Probability Distribution")
        fig = px.histogram(
            churn, x="churn_probability",
            nbins=50, color_discrete_sequence=["#e74c3c"]
        )
        st.plotly_chart(fig, use_container_width=True)

    # LTV distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("LTV by Segment")
        merged = rfm.merge(ltv, on="customer_unique_id", how="left")
        fig = px.box(
            merged, x="segment", y="predicted_ltv",
            color="segment",
            color_discrete_map={
                "Champions": "#2ecc71",
                "Loyal Customers": "#27ae60",
                "Potential Loyalists": "#f39c12",
                "At Risk": "#e67e22",
                "Lost": "#e74c3c"
            }
        )
        fig.update_layout(yaxis_range=[0, 500])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Segment Summary")
        summary = rfm.groupby("segment").agg(
            Customers=("customer_unique_id", "count"),
            Avg_Spend=("monetary", "mean"),
            Avg_Recency=("recency", "mean")
        ).round(2).reset_index()
        st.dataframe(summary, use_container_width=True)

# ── AI Analyst Page ────────────────────────────────────
elif page == "🤖 AI Analyst":
    st.title("🤖 AI Analyst")
    st.markdown("Ask anything about your customers in plain English")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your customers..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analysing..."):
                response = ask(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})