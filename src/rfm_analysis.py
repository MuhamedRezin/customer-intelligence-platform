import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "data/olist.db"

def run_rfm():
    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT
        c.customer_unique_id,
        MAX(o.order_purchase_timestamp) AS last_purchase_date,
        COUNT(DISTINCT o.order_id)      AS frequency,
        SUM(p.payment_value)            AS monetary
    FROM orders o
    JOIN customers c
        ON o.customer_id = c.customer_id
    JOIN order_payments p
        ON o.order_id = p.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Calculate Recency (days since last purchase)
    df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"])
    snapshot_date = df["last_purchase_date"].max() + pd.Timedelta(days=1)
    df["recency"] = (snapshot_date - df["last_purchase_date"]).dt.days

    # Score each metric 1-5
    df["R_score"] = pd.qcut(df["recency"],   q=5, labels=[5,4,3,2,1])
    df["F_score"] = pd.qcut(df["frequency"].rank(method="first"), q=5, labels=[1,2,3,4,5])
    df["M_score"] = pd.qcut(df["monetary"],  q=5, labels=[1,2,3,4,5])

    df["RFM_score"] = (
        df["R_score"].astype(int) +
        df["F_score"].astype(int) +
        df["M_score"].astype(int)
    )

    # Segment customers
    def segment(score):
        if score >= 13:
            return "Champions"
        elif score >= 10:
            return "Loyal Customers"
        elif score >= 7:
            return "Potential Loyalists"
        elif score >= 5:
            return "At Risk"
        else:
            return "Lost"

    df["segment"] = df["RFM_score"].apply(segment)

    # Save back to DB
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("rfm_scores", conn, if_exists="replace", index=False)
    conn.close()

    # Print summary
    print("✅ RFM Analysis Complete!\n")
    print("📊 Segment Distribution:")
    print(df["segment"].value_counts().to_string())
    print(f"\n📈 Total customers analysed: {len(df):,}")
    print("\n🔍 Sample output:")
    print(df[["customer_unique_id", "recency", "frequency", "monetary", "RFM_score", "segment"]].head(10).to_string())

if __name__ == "__main__":
    run_rfm()