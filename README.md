# Predictive Sales Forecasting Web App

A full-stack web application for forecasting sales using **custom algorithms** that combine:
- **Weighted Moving Averages (Sliding Window)**
- **Hash Map-Based Seasonality Adjustments**
- **Monte Carlo Simulations** for confidence intervals

Built to efficiently process **large-scale historical sales data** and provide **daily, monthly, or yearly predictions**.

---

## Features

- **Multi-Step Forecasting**: Predict multiple future periods with dynamic adjustments.
- **Custom Algorithm Design**:
  - Sliding window WMA for smoothing trends.
  - Seasonal factor mapping using hash maps (week-based).
  - Monte Carlo sampling to estimate forecast uncertainty.
- **Flexible Granularity**: Switch between daily, monthly, or yearly forecasts.
- **Interactive Visualizations**: Chart.js plots for historical and forecasted sales.
- **User-Uploaded Datasets**: Upload CSV files for custom predictions.
- **Efficient Large-Data Handling**: Supports 1M+ data points with optimized aggregations.

---

## Data Structures & Algorithms

This project demonstrates applied **DSA concepts** for time-series forecasting:

- **Sliding Window Algorithm**: Used for weighted moving average calculations.
- **Hash Maps**: Fast seasonal factor lookups (week number → adjustment).
- **Monte Carlo Simulation**: Random sampling to generate probabilistic forecasts.
- **Dynamic Programming-Like Iteration**: Multi-step forecasts built iteratively using previous predictions.
- **Efficient Aggregation**: Time-based grouping (daily → monthly/yearly) with minimal overhead.

---

## Tech Stack

- **Backend**: Python, Flask, SQLite
- **Data Processing**: Pandas, NumPy
- **Visualization**: Chart.js, Bootstrap
- **Version Control**: Git, Git LFS (for large datasets)

---

## Setup & Installation

### 1. Clone Repository
```bash
git clone https://github.com/<your-username>/predictive-sales-forecast.git
cd predictive-sales-forecast
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Prepare Database
- Place `sales_data.db` in the project root (or configure path in `predictive_model.py`).
- Ensure table `sales` exists with columns:
  ```
  product_id | date | sales
  ```

### 4. Run Application
```bash
python app.py
```
Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Usage

- **Predict via Product ID**: Enter an existing product ID from `sales_data.db`.
- **Upload Custom Data**: Go to `/upload` and provide a CSV with columns `date` and `sales`.
- **Adjust Forecast Mode**: Choose daily, monthly, or yearly in the dropdown menu.
- **View Visualizations**: Forecasts and history are displayed interactively with Chart.js.

---

## Example CSV Format

```csv
date,sales
2019-01-01,120
2019-01-02,135
2019-01-03,98
```

---

## Portfolio / Resume Highlight

This project demonstrates:
- **Algorithmic Problem-Solving** (WMA, Monte Carlo, hash maps).
- **Efficient Large-Data Handling** (optimized grouping and forecasting).
- **Full-Stack Development** (Flask backend, Chart.js frontend).
- **Software Engineering Practices** (modular code, type hints, logging, clean architecture).


---

## License

MIT License
