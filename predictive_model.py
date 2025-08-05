import pandas as pd
import sqlite3
import os
import random
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSSIBLE_PATHS = [
    os.path.join(BASE_DIR, "sales_data.db"),
    os.path.join(BASE_DIR, "../sales_data.db"),
    os.path.join(BASE_DIR, "../../sales_data.db"),
]

DB_FILE = None
for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        DB_FILE = os.path.abspath(path)
        break

if DB_FILE is None:
    DB_FILE = os.path.join(BASE_DIR, "sales_data.db")

print(f"[DEBUG] Using database file: {DB_FILE}")

def weighted_moving_average(values, window=7):
    weights = list(range(1, window + 1))
    wma = []
    for i in range(len(values)):
        if i < window - 1:
            wma.append(None)
        else:
            window_vals = values[i - window + 1: i + 1]
            weighted_sum = sum(w * v for w, v in zip(weights, window_vals))
            wma.append(weighted_sum / sum(weights))
    return wma

def build_seasonality_map(daily_sales):
    daily_sales['week_num'] = daily_sales['date'].dt.isocalendar().week
    weekly_avg = daily_sales.groupby('week_num')['sales'].mean()
    overall_avg = daily_sales['sales'].mean()
    return (weekly_avg / overall_avg).to_dict()

def monte_carlo_forecast(base_forecast, num_simulations=1000, volatility=0.1):
    sims = [base_forecast + random.gauss(0, volatility * base_forecast)
            for _ in range(num_simulations)]
    return np.mean(sims), np.percentile(sims, 5), np.percentile(sims, 95)

def load_daily_sales(product_id):
    conn = sqlite3.connect(DB_FILE)

    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    print(f"[DEBUG] Tables in {DB_FILE}:", tables)

    table_names = [t[0].lower() for t in tables]
    if "sales" not in table_names:
        conn.close()
        return {"error": f"No 'sales' table found in database {DB_FILE}"}

    query = """
    SELECT date, sales FROM sales WHERE product_id = ?
    ORDER BY date ASC
    """
    sales = pd.read_sql_query(query, conn, params=(product_id,))
    conn.close()

    if sales.empty:
        return None

    sales["date"] = pd.to_datetime(sales["date"])
    return sales.groupby("date")["sales"].sum().reset_index()

def aggregate_data(df, mode):
    if mode == "monthly":
        return df.resample("M", on="date").sum().reset_index()
    elif mode == "yearly":
        return df.resample("Y", on="date").sum().reset_index()
    return df  # daily

class AdvancedPredictor:
    def __init__(self, sales_df, window=7):
        self.sales_df = sales_df
        self.window = window
        self.sales_values = sales_df["sales"].tolist()
        self.wma = weighted_moving_average(self.sales_values, window=self.window)
        self.seasonality = build_seasonality_map(sales_df)

    def _predict_one_step(self, last_date, last_value):
        base = last_value
        next_date = last_date + pd.Timedelta(days=1)
        week_num = next_date.isocalendar().week
        season_factor = self.seasonality.get(week_num, 1.0)
        adjusted_forecast = base * season_factor
        mc_mean, _, _ = monte_carlo_forecast(adjusted_forecast)
        return next_date, round(mc_mean, 2)

    def predict_multi(self, steps=1):
        last_date = self.sales_df["date"].max()  # last recorded date
        last_value = self.wma[-1] or self.sales_values[-1]

        forecasts = []
        for _ in range(steps):
            next_date = last_date + pd.Timedelta(days=1)
            week_num = next_date.isocalendar().week
            season_factor = self.seasonality.get(week_num, 1.0)

            adjusted_forecast = last_value * season_factor
            mc_mean, _, _ = monte_carlo_forecast(adjusted_forecast)

            forecast_value = round(mc_mean, 2)
            forecasts.append({"date": str(next_date.date()), "forecast": forecast_value})

            last_date = next_date
            last_value = forecast_value

        return forecasts


def predict_for_product(product_id, mode="daily", steps=1):
    daily_sales = load_daily_sales(product_id)
    if isinstance(daily_sales, dict) and "error" in daily_sales:
        return daily_sales

    if daily_sales is None or len(daily_sales) < 7:
        return {"error": f"Not enough data for {product_id} (need at least 7 days)"}
    
    data = aggregate_data(daily_sales, mode)
    predictor = AdvancedPredictor(data)
    forecasts = predictor.predict_multi(steps)
    return {
        "product_id": product_id,
        "mode": mode,
        "history": data.to_dict(orient="records"),
        "forecasts": forecasts,
        "final_prediction": forecasts[-1]["forecast"],
        "date": forecasts[-1]["date"],
        "low_conf": None,
        "high_conf": None
    }

def predict_from_csv(filepath, mode="daily", steps=1):
    df = pd.read_csv(filepath)
    if "date" not in df or "sales" not in df:
        return {"error": "CSV must contain 'date' and 'sales' columns"}
    df["date"] = pd.to_datetime(df["date"])
    df = aggregate_data(df, mode)
    if len(df) < 7:
        return {"error": "Not enough data in uploaded file (need at least 7 entries)"}
    
    predictor = AdvancedPredictor(df)
    forecasts = predictor.predict_multi(steps)
    return {
        "product_id": "Uploaded CSV",
        "mode": mode,
        "history": df.to_dict(orient="records"),
        "forecasts": forecasts,
        "final_prediction": forecasts[-1]["forecast"],
        "date": forecasts[-1]["date"],
        "low_conf": None,
        "high_conf": None
    }
