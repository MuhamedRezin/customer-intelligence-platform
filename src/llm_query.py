import pandas as pd
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def get_db_engine():
    from sqlalchemy import create_engine
    url = "postgresql://postgres.tmmixkokqqohbgbzfabj:23188722mr22@aws-1-ap-south-1.pooler.supabase.com:5432/postgres"
    return create_engine(url)

def get_client():
    api_key = "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"
    return Groq(api_key=api_key)

def get_db_summary():
    engine = get_db_engine()
    segments = pd.read_sql("SELECT segment, COUNT(*) as count FROM rfm_scores GROUP BY segment", engine)
    churn = pd.read_sql("SELECT COUNT(*) as total, ROUND(AVG(churn_probability)::numeric * 100, 2) as avg_churn FROM churn_predictions", engine)
    ltv = pd.read_sql("SELECT ROUND(AVG(predicted_ltv)::numeric, 2) as avg_ltv FROM ltv_predictions", engine)
    return f"SEGMENTS:\n{segments.to_string(index=False)}\n\nCHURN:\n{churn.to_string(index=False)}\n\nLTV:\n{ltv.to_string(index=False)}"

def ask(question):
    client = get_client()
    db_summary = get_db_summary()
    system_prompt = f"You are a senior data analyst for an e-commerce business with 93357 customers. Answer with specific numbers.\n\nDATA:\n{db_summary}"
    response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": question}], temperature=0.3)
    return response.choices[0].message.content
