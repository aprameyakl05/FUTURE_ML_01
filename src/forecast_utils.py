#preprocessing
#feature engineering
#forecasting

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def preprocess_data(df):
    df["Order Date"] = pd.to_datetime(df["Order Date"])

    daily_sales = (
        df.groupby("Order Date")["Sales"]
        .sum()
        .reset_index()
    )

    daily_sales = daily_sales.set_index("Order Date")
    daily_sales = daily_sales.asfreq("D", fill_value=0)
    daily_sales = daily_sales.reset_index()

    return daily_sales


def create_features(daily_sales):

    daily_sales["year"] = daily_sales["Order Date"].dt.year
    daily_sales["month"] = daily_sales["Order Date"].dt.month
    daily_sales["day"] = daily_sales["Order Date"].dt.day
    daily_sales["weekday"] = daily_sales["Order Date"].dt.weekday
    daily_sales["quarter"] = daily_sales["Order Date"].dt.quarter

    daily_sales["lag_1"] = daily_sales["Sales"].shift(1)
    daily_sales["lag_7"] = daily_sales["Sales"].shift(7)
    daily_sales["lag_14"] = daily_sales["Sales"].shift(14)
    daily_sales["lag_30"] = daily_sales["Sales"].shift(30)

    daily_sales["rolling_mean_7"] = (
        daily_sales["Sales"].rolling(7).mean()
    )

    daily_sales["rolling_mean_14"] = (
        daily_sales["Sales"].rolling(14).mean()
    )

    daily_sales["rolling_mean_30"] = (
        daily_sales["Sales"].rolling(30).mean()
    )

    daily_sales["is_weekend"] = (
        daily_sales["weekday"] >= 5
    ).astype(int)

    daily_sales = daily_sales.dropna()

    return daily_sales


def train_model(daily_sales):

    features = [
        "year",
        "month",
        "day",
        "weekday",
        "quarter",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_30",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_30",
        "is_weekend"
    ]

    X = daily_sales[features]
    y = daily_sales["Sales"]

    split_index = int(len(daily_sales) * 0.8)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, y_test, y_pred, mae, rmse, r2


def forecast_future(daily_sales, model, forecast_days):

    last_date = daily_sales["Order Date"].max()

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D"
    )

    future_predictions = []

    sales_history = daily_sales["Sales"].tolist()

    for date in future_dates:

        lag_1 = sales_history[-1]
        lag_7 = sales_history[-7]
        lag_14 = sales_history[-14]
        lag_30 = sales_history[-30]

        rolling_mean_7 = np.mean(sales_history[-7:])
        rolling_mean_14 = np.mean(sales_history[-14:])
        rolling_mean_30 = np.mean(sales_history[-30:])

        is_weekend = 1 if date.weekday() >= 5 else 0

        future_row = pd.DataFrame({
            "year": [date.year],
            "month": [date.month],
            "day": [date.day],
            "weekday": [date.weekday()],
            "quarter": [date.quarter],
            "lag_1": [lag_1],
            "lag_7": [lag_7],
            "lag_14": [lag_14],
            "lag_30": [lag_30],
            "rolling_mean_7": [rolling_mean_7],
            "rolling_mean_14": [rolling_mean_14],
            "rolling_mean_30": [rolling_mean_30],
            "is_weekend": [is_weekend]
        })

        pred = model.predict(future_row)[0]

        future_predictions.append(pred)

        sales_history.append(pred)

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Sales": future_predictions
    })

    return forecast_df