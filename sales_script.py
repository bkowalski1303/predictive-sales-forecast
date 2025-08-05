import pandas as pd
import sqlite3

# File paths
csv_file = "sales.csv"         # Path to your large CSV file
db_file = "sales_data.db"      # SQLite database filename

# Step 1: Read the CSV in chunks if it's huge
chunksize = 100000  # Adjust if needed
first_chunk = True

# Connect to SQLite
conn = sqlite3.connect(db_file)

for chunk in pd.read_csv(csv_file, chunksize=chunksize):
    if first_chunk:
        # Create table with first chunk
        chunk.to_sql("sales", conn, if_exists="replace", index=False)
        first_chunk = False
    else:
        # Append data for subsequent chunks
        chunk.to_sql("sales", conn, if_exists="append", index=False)

print("✅ Sales data successfully loaded into sales_data.db")

# Test: Fetch first 5 rows
print(pd.read_sql("SELECT * FROM sales LIMIT 5;", conn))

conn.close()
