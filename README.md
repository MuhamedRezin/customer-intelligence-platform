# 🧠 Customer Intelligence Platform
> An end-to-end data science system that transforms raw e-commerce transactions into actionable business intelligence — powered by ML models, a cloud database, and an LLM-driven AI analyst.

**🔗 Live Demo:** [customer-intelligence-platform-nsqhuf8mbpvxq5dwtuwje4.streamlit.app](https://customer-intelligence-platform-nsqhuf8mbpvxq5dwtuwje4.streamlit.app)

---

## What This Does

Most data science portfolios stop at a Jupyter notebook. This project goes further — it ingests real data, runs production-grade ML models, stores results in a cloud database, and exposes everything through a live web application with a natural language interface.

Ask it *"Which customer segment has the highest churn risk?"* and it pulls from real data and answers like a senior analyst.

---

## Architecture

```
Raw Data (Olist CSV) → SQLite → Feature Engineering → ML Models → Supabase (PostgreSQL)
                                                                          ↓
                                                          Streamlit Dashboard + LLM Query Layer
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data & SQL | Python, Pandas, SQLite, Supabase (PostgreSQL) |
| Machine Learning | Scikit-learn, XGBoost, SMOTE (imbalanced-learn) |
| LLM Integration | Groq API (Llama 3.3 70B) |
| Visualization | Plotly, Matplotlib, Seaborn |
| Deployment | Streamlit Cloud |
| Version Control | Git, GitHub |

---

## Models & Results

### Churn Prediction (XGBoost Classifier)
- **ROC-AUC: 0.75** on held-out test set
- Applied SMOTE to handle class imbalance (25% churn rate)
- Features: purchase frequency, monetary value, review scores, seller diversity, freight patterns

### Customer Lifetime Value (XGBoost Regressor)
- **R² Score: 0.8775** — explains 87.75% of variance in customer spend
- **Mean Absolute Error: R$20.82** on unseen customers
- Predictions saved per customer for dashboard consumption

### RFM Segmentation
- 93,357 customers scored on Recency, Frequency, Monetary value
- 5 actionable segments: Champions, Loyal Customers, Potential Loyalists, At Risk, Lost

| Segment | Customers | Avg Spend |
|---|---|---|
| Potential Loyalists | 38,395 | R$163 |
| Loyal Customers | 31,506 | R$215 |
| At Risk | 12,266 | R$142 |
| Champions | 8,029 | R$287 |
| Lost | 3,161 | R$98 |

---

## Features

**📊 Dashboard**
- Live KPI cards (total customers, avg spend, churn risk, avg LTV)
- Interactive segment distribution charts
- Churn probability histogram
- LTV by segment boxplot
- Segment summary table

**🤖 AI Analyst**
- Natural language interface powered by Llama 3.3 70B via Groq
- Queries live Supabase database on every question
- Answers like a senior analyst with specific numbers and recommendations

---

## Project Structure

```
customer-intelligence-platform/
├── app.py                    ← Streamlit web app
├── src/
│   ├── database_setup.py     ← Load CSVs into SQLite
│   ├── rfm_analysis.py       ← RFM scoring engine
│   ├── churn_model.py        ← XGBoost churn model
│   ├── ltv_model.py          ← XGBoost LTV regression
│   ├── llm_query.py          ← Groq LLM integration
│   ├── eda.py                ← Exploratory data analysis
│   └── upload_to_supabase.py ← Cloud DB migration
├── Notebooks/
│   └── eda_outputs/          ← EDA visualizations
└── requirements.txt
```

---

## Running Locally

**1. Clone and install dependencies**
```bash
git clone https://github.com/MuhamedRezin/customer-intelligence-platform.git
cd customer-intelligence-platform
pip install -r requirements.txt
```

**2. Download the dataset**

Download from [Kaggle — Brazilian E-Commerce by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place all 9 CSV files in `Data/raw/`

**3. Set up environment variables**
```bash
# Create .env file
GROQ_API_KEY=your_groq_key
SUPABASE_URL=your_supabase_connection_string
```

**4. Run the pipeline**
```bash
python src/database_setup.py      # Load data
python src/rfm_analysis.py        # RFM scoring
python src/churn_model.py         # Train churn model
python src/ltv_model.py           # Train LTV model
streamlit run app.py              # Launch app
```

---

## Dataset

**Brazilian E-Commerce Public Dataset by Olist** — 100,000+ real orders from 2016–2018 across 9 relational tables covering orders, customers, products, sellers, payments, and reviews.

Source: [kaggle.com/datasets/olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

---

## Author

**Muhamed Rezin** — Final year BCA (AI & Data Science), Yenepoya University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-muhamedrezin-blue)](https://linkedin.com/in/muhamedrezin)
[![GitHub](https://img.shields.io/badge/GitHub-MuhamedRezin-black)](https://github.com/MuhamedRezin)