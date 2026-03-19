import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pickle
import os

DB_PATH = "data/olist.db"
MODEL_PATH = "src/models/"
os.makedirs(MODEL_PATH, exist_ok=True)

def load_features():
    conn = sqlite3.connect(DB_PATH)
    
    # Load rfm scores
    rfm = pd.read_sql("SELECT * FROM rfm_scores", conn)
    
    # Load average review score per customer
    reviews_query = """
    SELECT c.customer_unique_id, AVG(r.review_score) AS avg_review_score
    FROM order_reviews r
    JOIN orders o ON r.order_id = o.order_id
    JOIN customers c ON o.customer_id = c.customer_id
    GROUP BY c.customer_unique_id
    """
    reviews = pd.read_sql(reviews_query, conn)

    # Load order items stats per customer
    items_query = """
    SELECT c.customer_unique_id,
           COUNT(DISTINCT oi.seller_id) AS unique_sellers,
           AVG(oi.freight_value) AS avg_freight,
           AVG(oi.price) AS avg_item_price
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN customers c ON o.customer_id = c.customer_id
    GROUP BY c.customer_unique_id
    """
    items = pd.read_sql(items_query, conn)
    conn.close()

    # Merge everything
    df = rfm.merge(reviews, on="customer_unique_id", how="left")
    df = df.merge(items, on="customer_unique_id", how="left")
    df = df.fillna(df.median(numeric_only=True))
    
    print(f"✅ Features loaded: {df.shape[0]:,} customers, {df.shape[1]} columns")
    return df

def create_churn_label(df):
    # Use median recency as threshold instead of hardcoded rule
    recency_threshold = df["recency"].quantile(0.75)
    df["churned"] = (df["recency"] > recency_threshold).astype(int)
    print(f"✅ Recency threshold: {recency_threshold:.0f} days")
    print(f"✅ Churn rate: {df['churned'].mean():.2%}")
    return df

def train_churn_model(df):
    features = ["frequency", "monetary", "avg_review_score", "unique_sellers", "avg_freight", "avg_item_price"]
    X = df[features]
    y = df["churned"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE for class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"✅ SMOTE applied — training samples: {len(X_train_resampled):,}")

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train_resampled, y_train_resampled)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"🎯 ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    # Save model and scaler
    pickle.dump(model, open(f"{MODEL_PATH}churn_model.pkl", "wb"))
    pickle.dump(scaler, open(f"{MODEL_PATH}churn_scaler.pkl", "wb"))
    print(f"\n💾 Model saved to {MODEL_PATH}")

    return model, scaler, X_test_scaled, y_test

def save_predictions(df, model, scaler):
    features = ["frequency", "monetary", "avg_review_score", "unique_sellers", "avg_freight", "avg_item_price"]
    X = scaler.transform(df[features])
    df["churn_probability"] = model.predict_proba(X)[:, 1]
    df["churn_prediction"] = model.predict(X)

    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        df[["customer_unique_id", "churn_probability", "churn_prediction", "churned"]].to_sql(
            "churn_predictions", conn, if_exists="replace", index=False
        )
        conn.commit()
        print("✅ Predictions saved to database")
    finally:
        conn.close()
if __name__ == "__main__":
    print("🚀 Loading features...\n")
    df = load_features()
    df = create_churn_label(df)
    print(f"\n🤖 Training XGBoost model...\n")
    model, scaler, X_test, y_test = train_churn_model(df)
    print("\n💡 Saving predictions for all customers...\n")
    save_predictions(df, model, scaler)
    print("\n🎉 Churn model complete!")