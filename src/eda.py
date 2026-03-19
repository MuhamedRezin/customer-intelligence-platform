import sqlite3
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend, no Tkinter needed
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DB_PATH = "data/olist.db"
OUTPUT_PATH = "notebooks/eda_outputs/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
rfm = pd.read_sql("SELECT * FROM rfm_scores", conn)
orders = pd.read_sql("SELECT * FROM orders", conn)
conn.close()

sns.set_theme(style="darkgrid")

# ── Plot 1: Segment Distribution ──────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
order = ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Lost"]
colors = ["#2ecc71", "#27ae60", "#f39c12", "#e67e22", "#e74c3c"]
sns.countplot(data=rfm, x="segment", order=order, palette=colors, ax=ax)
ax.set_title("Customer Segment Distribution", fontsize=16, fontweight="bold")
ax.set_xlabel("Segment")
ax.set_ylabel("Number of Customers")
for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}segment_distribution.png", dpi=150)
print("✅ Plot 1 saved: segment_distribution.png")

# ── Plot 2: Monetary Value by Segment ─────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=rfm, x="segment", y="monetary", order=order, palette=colors, ax=ax)
ax.set_title("Monetary Value by Segment", fontsize=16, fontweight="bold")
ax.set_xlabel("Segment")
ax.set_ylabel("Total Spend (BRL)")
ax.set_ylim(0, 1000)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}monetary_by_segment.png", dpi=150)
print("✅ Plot 2 saved: monetary_by_segment.png")

# ── Plot 3: Orders Over Time ───────────────────────────────
orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
orders["month"] = orders["order_purchase_timestamp"].dt.to_period("M")
monthly = orders.groupby("month").size().reset_index(name="order_count")
monthly["month"] = monthly["month"].astype(str)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(monthly["month"], monthly["order_count"], marker="o", linewidth=2, color="#3498db")
ax.set_title("Monthly Order Volume", fontsize=16, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}monthly_orders.png", dpi=150)
print("✅ Plot 3 saved: monthly_orders.png")

# ── Plot 4: Recency Distribution ──────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(rfm["recency"], bins=50, color="#9b59b6", ax=ax)
ax.set_title("Customer Recency Distribution", fontsize=16, fontweight="bold")
ax.set_xlabel("Days Since Last Purchase")
ax.set_ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}recency_distribution.png", dpi=150)
print("✅ Plot 4 saved: recency_distribution.png")

print(f"\n🎉 All plots saved to {OUTPUT_PATH}")