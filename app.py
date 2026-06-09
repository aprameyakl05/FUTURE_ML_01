import streamlit as st
import pandas as pd
import plotly.express as px

from src.forecast_utils import (
    preprocess_data,
    create_features,
    train_model,
    forecast_future
)

st.set_page_config(
    page_title="Sales & Demand Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales & Demand Forecasting Dashboard")

st.sidebar.title("Dashboard Controls")

forecast_days = st.sidebar.slider(
    "Forecast Days",
    min_value=7,
    max_value=90,
    value=30
)

uploaded_file = st.file_uploader(
    "Upload Sales Dataset (CSV)",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(
        uploaded_file,
        encoding="cp1252"
    )

    st.success("Dataset uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    # PREPROCESSING

    st.subheader("⚙️ Data Preprocessing")

    daily_sales = preprocess_data(df)

    st.success("Data preprocessing completed!")

    st.dataframe(
        daily_sales.head()
    )

    # HISTORICAL SALES

    st.subheader("📊 Historical Sales Trend")

    historical_fig = px.line(
        daily_sales,
        x="Order Date",
        y="Sales",
        title="Daily Sales Over Time"
    )

    st.plotly_chart(
        historical_fig,
        use_container_width=True
    )

    # FEATURE ENGINEERING

    st.subheader("🧠 Feature Engineering")

    daily_sales = create_features(
        daily_sales
    )

    st.success(
        "Feature Engineering Completed"
    )

    # TRAIN MODEL

    if st.button("🚀 Train Forecasting Model"):

        (
            model,
            y_test,
            y_pred,
            mae,
            rmse,
            r2
        ) = train_model(daily_sales)

        st.subheader("📊 Model Performance")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "MAE",
                f"{mae:.2f}"
            )

        with c2:
            st.metric(
                "RMSE",
                f"{rmse:.2f}"
            )

        with c3:
            st.metric(
                "R² Score",
                f"{r2:.3f}"
            )

        # ACTUAL VS PREDICTED

        st.subheader(
            "📉 Actual vs Predicted Sales"
        )

        comparison_df = pd.DataFrame({
            "Actual Sales": y_test.values,
            "Predicted Sales": y_pred
        })

        comparison_fig = px.line(
            comparison_df,
            title="Actual vs Predicted Sales"
        )

        st.plotly_chart(
            comparison_fig,
            use_container_width=True
        )

        # FORECAST

        st.subheader(
            f"🔮 {forecast_days}-Day Sales Forecast"
        )

        forecast_df = forecast_future(
            daily_sales,
            model,
            forecast_days
        )

        st.dataframe(
            forecast_df
        )

        forecast_fig = px.line(
            forecast_df,
            x="Date",
            y="Predicted Sales",
            title=f"Next {forecast_days} Days Forecast"
        )

        st.plotly_chart(
            forecast_fig,
            use_container_width=True
        )

        # INSIGHTS

        avg_forecast = forecast_df[
            "Predicted Sales"
        ].mean()

        st.subheader("💡 Business Insights")

        st.info(
            f"""
Average Forecasted Daily Sales: ${avg_forecast:,.2f}

This forecast can help businesses:
- Plan inventory levels
- Reduce stock shortages
- Optimize staffing
- Improve cash flow planning
"""
        )

        # DOWNLOAD

        csv = forecast_df.to_csv(
            index=False
        )

        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name="sales_forecast.csv",
            mime="text/csv"
        )

st.markdown("---")

st.markdown(
    """
Machine Learning Internship Project – Future Interns
"""
)