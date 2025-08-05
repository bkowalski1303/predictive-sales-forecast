import sqlite3

DB_FILE = r"C:\Users\kowal\Computer Science\Personal_Code_Projects\Relearning Python\PROJECTS\Predictive_Model\sales_data.db"

conn = sqlite3.connect(DB_FILE)


rows = conn.execute("""
    SELECT product_id, MAX(date) AS last_sale_date
    FROM sales
    GROUP BY product_id
    ORDER BY last_sale_date DESC
""").fetchall()

conn.close()

if rows:
    print("Last recorded sale date for each product:\n")
    for r in rows:
        print(f"Product {r[0]} — Last Sale: {r[1]}")
else:
    print("No sales records found.")
