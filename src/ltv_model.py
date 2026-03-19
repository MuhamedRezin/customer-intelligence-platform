import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import pickle
import os

DB_PATH = "data/olist.db"
MODEL_PATH = "src/models/"
os.makedirs(MODEL_PATH, exist_ok=True)

def load_features():
    conn = sqlite3.connect(DB_PATH)

    rfm = pd.read_sql("SELECT * FROM rfm_scores", conn)

    reviews = pd.read_sql("""
        SELECT c.customer_unique_id, AVG(r.review_score) AS avg_review_score
        FROM order_reviews r
        JOIN orders o ON r.order_id = o.order_id
        JOIN customers c ON o.customer_id = c.customer_id
        GROUP BY c.customer_unique_id
    """, conn)

    items = pd.read_sql("""
        SELECT c.customer_unique_id,
               COUNT(DISTINCT oi.seller_id) AS unique_sellers,
               AVG(oi.freight_value) AS avg_freight,
               AVG(oi.price) AS avg_item_price,
               COUNT(DISTINCT o.order_id) AS total_orders
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.order_id
        JOIN customers c ON o.customer_id = c.customer_id
        GROUP BY c.customer_unique_id
    """, conn)

    conn.close()

    df = rfm.merge(reviews, on="customer_unique_id", how="left")
    df = df.merge(items, on="customer_unique_id", how="left")
    df = df.fillna(df.median(numeric_only=True))

    print(f"✅ Features loaded: {df.shape[0]:,} customers")
    return df

def train_ltv_model(df):
    # Target: monetary value (total spend = LTV proxy)
    features = ["recency", "frequency", "avg_review_score",
                "unique_sellers", "avg_freight", "avg_item_price", "total_orders"]

    # Remove outliers (top 1%)
    upper = df["monetary"].quantile(0.99)
    df = df[df["monetary"] <= upper].copy()
    print(f"✅ After outlier removal: {len(df):,} customers")

    X = df[features]
    y = df["monetary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 LTV Model Results:")
    print(f"   Mean Absolute Error : R$ {mae:.2f}")
    print(f"   R² Score            : {r2:.4f}")

    pickle.dump(model, open(f"{MODEL_PATH}ltv_model.pkl", "wb"))
    pickle.dump(scaler, open(f"{MODEL_PATH}ltv_scaler.pkl", "wb"))
    print(f"\n💾 Model saved to {MODEL_PATH}")

    return model, scaler, df, features

def save_predictions(model, scaler, df, features):
    X = scaler.transform(df[features])
    df["predicted_ltv"] = model.predict(X)

    conn = sqlite3.connect(DB_PATH)
    df[["customer_unique_id", "monetary", "predicted_ltv"]].to_sql(
        "ltv_predictions", conn, if_exists="replace", index=False
    )
    conn.close()
    print("✅ LTV predictions saved to database")

    print(f"\n💡 LTV Summary:")
    print(f"   Avg predicted LTV : R$ {df['predicted_ltv'].mean():.2f}")
    print(f"   Top 10% LTV       : R$ {df['predicted_ltv'].quantile(0.9):.2f}")
    print(f"   Max predicted LTV : R$ {df['predicted_ltv'].max():.2f}")

if __name__ == "__main__":
    print("🚀 Loading features...\n")
    df = load_features()
    print("\n🤖 Training LTV model...\n")
    model, scaler, df, features = train_ltv_model(df)
    print("\n💡 Saving LTV predictions...\n")
    save_predictions(model, scaler, df, features)
    print("\n🎉 LTV model complete!")