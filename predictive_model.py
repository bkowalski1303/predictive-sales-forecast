import pandas as pd
import sqlite3
import os
import random
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Union

# ------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------
# Set default logging level to INFO for production.
# Change to DEBUG when detailed output is needed for debugging.
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ------------------------------------------------------------
# Database Path Resolution
# ------------------------------------------------------------
# Try multiple paths to locate the database file to handle various run contexts.
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

# Default to base directory if nothing found
if DB_FILE is None:
    DB_FILE = os.path.join(BASE_DIR, "sales_data.db")

logging.info(f"Using database file: {DB_FILE}")


# ------------------------------------------------------------
# Weighted Moving Average (Sliding Window Algorithm)
# ------------------------------------------------------------
def weighted_moving_average(values: List[float], window: int = 7) -> List[Optional[float]]:
    """
    Compute weighted moving average using a sliding window.

    Each element in the window is weighted incrementally (1 to window size),
    providing more weight to recent observations.

    Args:
        values: List of numerical sales values.
        window: Number of periods to include in the moving average.

    Returns:
        List of weighted averages (None for initial positions without enough data).
    """
    weights = list(range(1, window + 1))
    wma: List[Optional[float]] = []

    for i in range(len(values)):
        if i < window - 1:
            # Not enough data points to compute WMA
            wma.append(None)
        else:
            # Compute weighted sum for the last 'window' points
            window_vals = values[i - window + 1: i + 1]
            weighted_sum = sum(w * v for w, v in zip(weights, window_vals))
            wma.append(weighted_sum / sum(weights))
    return wma


# ------------------------------------------------------------
# Seasonality Factor Mapping (Hash Map Implementation)
# ------------------------------------------------------------
def build_seasonality_map(daily_sales: pd.DataFrame) -> Dict[int, float]:
    """
    Create a hash map of seasonal adjustment factors by ISO week number.

    Each week's average sales is compared to the overall mean to capture
    recurring seasonal effects.

    Args:
        daily_sales: DataFrame containing 'date' and 'sales' columns.

    Returns:
        Dictionary mapping week number to seasonal adjustment factor.
    """
    daily_sales['week_num'] = daily_sales['date'].dt.isocalendar().week
    weekly_avg = daily_sales.groupby('week_num')['sales'].mean()
    overall_avg = daily_sales['sales'].mean()
    return (weekly_avg / overall_avg).to_dict()


# ------------------------------------------------------------
# Monte Carlo Simulation for Forecast Uncertainty
# ------------------------------------------------------------
def monte_carlo_forecast(base_forecast: float,
                         num_simulations: int = 1000,
                         volatility: float = 0.1) -> Tuple[float, float, float]:
    """
    Perform Monte Carlo simulation to estimate uncertainty around a forecast.

    Adds normally-distributed noise to the base forecast across multiple
    simulations, returning mean and confidence interval percentiles.

    Args:
        base_forecast: Baseline forecast value.
        num_simulations: Number of simulation iterations.
        volatility: Percent variation (std dev as fraction of base_forecast).

    Returns:
        Tuple of (mean, 5th percentile, 95th percentile) forecast values.
    """
    sims = [
        base_forecast + random.gauss(0, volatility * base_forecast)
        for _ in range(num_simulations)
    ]
    return np.mean(sims), np.percentile(sims, 5), np.percentile(sims, 95)


# ------------------------------------------------------------
# Load Sales Data for Specific Product
# ------------------------------------------------------------
def load_daily_sales(product_id: str) -> Union[Optional[pd.DataFrame], Dict[str, str]]:
    """
    Retrieve daily sales data for a specific product from SQLite database.

    Validates table existence and returns None if no data is found.

    Args:
        product_id: Identifier of the product to query.

    Returns:
        - DataFrame of daily sales (date, sales)
        - None if no data
        - Dict with error message if 'sales' table is missing
    """
    conn = sqlite3.connect(DB_FILE)

    # Confirm 'sales' table exists
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    table_names = [t[0].lower() for t in tables]
    if "sales" not in table_names:
        conn.close()
        return {"error": f"No 'sales' table found in database {DB_FILE}"}

    # Query sales data for product
    query = """
    SELECT date, sales FROM sales WHERE product_id = ?
    ORDER BY date ASC
    """
    sales = pd.read_sql_query(query, conn, params=(product_id,))
    conn.close()

    if sales.empty:
        return None

    # Convert dates and combine duplicate dates if necessary
    sales["date"] = pd.to_datetime(sales["date"])
    return sales.groupby("date")["sales"].sum().reset_index()


# ------------------------------------------------------------
# Aggregate Data by Granularity
# ------------------------------------------------------------
def aggregate_data(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Resample sales data to daily, monthly, or yearly frequency.

    Args:
        df: DataFrame with daily sales data.
        mode: 'daily', 'monthly', or 'yearly'.

    Returns:
        Resampled DataFrame with aggregated sales.
    """
    if mode == "monthly":
        return df.resample("M", on="date").sum().reset_index()
    elif mode == "yearly":
        return df.resample("Y", on="date").sum().reset_index()
    return df  # Default is daily


# ------------------------------------------------------------
# Forecasting Class
# ------------------------------------------------------------
class AdvancedPredictor:
    """
    Forecast future sales using weighted moving averages,
    seasonal adjustment, and Monte Carlo simulations.
    """

    def __init__(self, sales_df: pd.DataFrame, window: int = 7):
        self.sales_df = sales_df
        self.window = window
        self.sales_values = sales_df["sales"].tolist()

        # Precompute WMA and seasonality map for efficiency
        self.wma = weighted_moving_average(self.sales_values, window=self.window)
        self.seasonality = build_seasonality_map(sales_df)

    def predict_multi(self, steps: int = 1) -> List[Dict[str, str | float]]:
        """
        Generate multi-step forecasts sequentially.

        Starts from last recorded date and iteratively applies seasonal
        adjustments and Monte Carlo noise for each predicted step.

        Args:
            steps: Number of periods to forecast.

        Returns:
            List of forecast dicts: [{'date': str, 'forecast': float}, ...]
        """
        last_date = self.sales_df["date"].max()
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

            # Update for next iteration
            last_date = next_date
            last_value = forecast_value

        return forecasts


# ------------------------------------------------------------
# Main Prediction Functions
# ------------------------------------------------------------
def predict_for_product(product_id: str,
                        mode: str = "daily",
                        steps: int = 1) -> Dict[str, Union[str, float, List[Dict]]]:
    """
    Predict sales for a product using historical database data.
    """
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


def predict_from_csv(filepath: str,
                     mode: str = "daily",
                     steps: int = 1) -> Dict[str, Union[str, float, List[Dict]]]:
    """
    Predict sales from an uploaded CSV file.
    """
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
