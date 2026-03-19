import pandas as pd
import sqlite3
import os

# Paths
RAW_DATA_PATH = "data/raw/"
DB_PATH = "data/olist.db"

# All 9 CSV files
CSV_FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
    "geolocation": "olist_geolocation_dataset.csv",
}

def load_csvs():
    dataframes = {}
    for name, filename in CSV_FILES.items():
        path = os.path.join(RAW_DATA_PATH, filename)
        df = pd.read_csv(path)
        dataframes[name] = df
        print(f"✅ Loaded {name}: {df.shape[0]} rows, {df.shape[1]} columns")
    return dataframes

def save_to_sqlite(dataframes):
    conn = sqlite3.connect(DB_PATH)
    for name, df in dataframes.items():
        df.to_sql(name, conn, if_exists="replace", index=False)
        print(f"📦 Saved table: {name}")
    conn.close()
    print(f"\n🎉 Database created at {DB_PATH}")

def verify_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("\n📋 Tables in database:")
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
        count = cursor.fetchone()[0]
        print(f"   {table[0]}: {count} rows")
    conn.close()

if __name__ == "__main__":
    print("🚀 Loading CSVs...\n")
    dataframes = load_csvs()
    print("\n💾 Saving to SQLite...\n")
    save_to_sqlite(dataframes)
    verify_db()