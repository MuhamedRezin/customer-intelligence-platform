import sqlite3
import pandas as pd
from sqlalchemy import create_engine

SQLITE_PATH = "data/olist.db"
SUPABASE_URL = "postgresql://postgres:23188722mr22@db.tmmixkokqqohbgbzfabj.supabase.co:5432/postgres"

TABLES = ["rfm_scores", "churn_predictions", "ltv_predictions"]

def upload():
    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    pg_engine = create_engine(SUPABASE_URL)
    
    for table in TABLES:
        print(f"📤 Uploading {table}...")
        df = pd.read_sql(f"SELECT * FROM {table}", sqlite_conn)
        df.to_sql(table, pg_engine, if_exists="replace", index=False)
        print(f"✅ {table}: {len(df):,} rows uploaded")
    
    sqlite_conn.close()
    print("\n🎉 All tables uploaded to Supabase!")

if __name__ == "__main__":
    upload()