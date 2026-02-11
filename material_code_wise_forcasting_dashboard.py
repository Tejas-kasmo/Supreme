"""
SUPREME FORECAST DASHBOARD
Streamlit application for sales forecasting in Snowflake
Combined application with two tabs:
1. Future Forecast Tab - Forecast into the future
2. Historical Comparison Tab - Train on FY 2022-23, 2023-24 and predict for FY 2024-25
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import plotly.graph_objects as go
from prophet import Prophet
import io
import os

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="SUPREME FORECAST DASHBOARD",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get Snowflake session
from snowflake.snowpark.context import get_active_session
session = get_active_session()

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1f4788;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        border-bottom: 3px solid #1f4788;
    }
    .stButton>button {
        background-color: #1f4788;
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 4px;
        border: none;
        width: 100%;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #163a6d;
    }
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)


#######################
# FISCAL YEAR HELPERS
#######################

def get_forecast_start_date_tab1():
    """
    Returns April 1, 2025 as the fixed forecast start date for Tab 1 (Future Forecast)
    """
    return datetime(2025, 4, 1)


def get_forecast_start_date_tab2():
    """
    Returns April 1, 2024 as the fixed forecast start date for Tab 2 (Historical Comparison)
    """
    return datetime(2024, 4, 1)

def get_fiscal_year(date):
    """
    Get fiscal year for a given date.
    Fiscal year runs from April 1 to March 31.
    FY 2022-23 means April 1, 2022 to March 31, 2023
    """
    if date.month >= 4:  # April to December
        return f"{date.year}-{str(date.year + 1)[-2:]}"
    else:  # January to March
        return f"{date.year - 1}-{str(date.year)[-2:]}"


def get_fy_date_range(fy_string):
    """
    Get start and end dates for a fiscal year string like '2022-23'
    Returns: (start_date, end_date)
    """
    start_year = int(fy_string.split('-')[0])
    start_date = datetime(start_year, 4, 1)
    end_date = datetime(start_year + 1, 3, 31)
    return start_date, end_date


#######################
# COMMON CORE FUNCTIONS
#######################

@st.cache_data(ttl=3600)
def load_material_groups():
    """Load unique material groups sorted by data volume (descending)"""
    query = """
    SELECT MATERIAL_GROUP_COMBINED, COUNT(*) as data_points
    FROM SUPREME_DB.SUPREME_SCH.DAILY_AGG_SALES_DATA 
    WHERE MATERIAL_GROUP_COMBINED IS NOT NULL
    GROUP BY MATERIAL_GROUP_COMBINED
    ORDER BY data_points DESC, MATERIAL_GROUP_COMBINED
    """
    df = session.sql(query).to_pandas()
    return df['MATERIAL_GROUP_COMBINED'].tolist()


@st.cache_data(ttl=3600)
def load_material_codes_by_group(material_group):
    """Load material codes for a specific material group sorted by data volume (descending)"""
    query = f"""
    SELECT MATERIAL_CODE, COUNT(*) as data_points
    FROM SUPREME_DB.SUPREME_SCH.DAILY_AGG_SALES_DATA 
    WHERE MATERIAL_GROUP_COMBINED = '{material_group}'
      AND MATERIAL_CODE IS NOT NULL
    GROUP BY MATERIAL_CODE
    ORDER BY data_points DESC, MATERIAL_CODE
    """
    df = session.sql(query).to_pandas()
    return df['MATERIAL_CODE'].tolist()


def load_filtered_data(material_code: str) -> pd.DataFrame:
    """Load data for specific material code"""
    query = f"""
    SELECT 
        BILLING_DATE,
        MATERIAL_CODE,
        TOTAL_BILLING_QTY_BASE_UNIT,
        TOTAL_NET_WT,
        AVG_BILLING_QTY_BASE_UNIT,
        AVG_NET_WT
    FROM SUPREME_DB.SUPREME_SCH.DAILY_AGG_SALES_DATA
    WHERE MATERIAL_CODE = '{material_code}'
    ORDER BY BILLING_DATE
    """
    df = session.sql(query).to_pandas()
    df['BILLING_DATE'] = pd.to_datetime(df['BILLING_DATE'])
    return df


def aggregate_daily_for_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Returns daily aggregated dataframe with columns ds (date) and y (value)"""
    tmp = df[['BILLING_DATE', target_col]].copy()
    tmp = tmp.groupby('BILLING_DATE')[target_col].sum().reset_index()
    tmp.columns = ['ds', 'y']
    tmp = tmp.sort_values('ds').reset_index(drop=True)
    tmp['y'] = tmp['y'].clip(lower=0)
    return tmp


def monthly_aggregate(df: pd.DataFrame, value_col: str = 'y') -> pd.DataFrame:
    """Aggregate daily data to monthly"""
    tmp = df.copy()
    tmp['month'] = tmp['ds'].dt.to_period('M').dt.to_timestamp()
    monthly = tmp.groupby('month')[value_col].sum().reset_index()
    monthly.columns = ['ds', 'y']
    return monthly


def detect_density_patterns(hist_df: pd.DataFrame) -> dict:
    """Analyze density patterns in historical data"""
    df = hist_df.copy()
    df['dow'] = df['ds'].dt.dayofweek
    df['dom'] = df['ds'].dt.day
    
    presence = {}
    dow = df.groupby('dow').apply(lambda g: (g['y'] > 0).sum() / len(g)).to_dict()
    presence['dow_frac'] = dow
    dom = df.groupby('dom').apply(lambda g: (g['y'] > 0).sum() / len(g)).to_dict()
    presence['dom_frac'] = dom
    
    return presence


def detect_anomalies_iqr(hist_df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    """Mark anomalies using IQR method"""
    df = hist_df.copy()
    q1 = df['y'].quantile(0.25)
    q3 = df['y'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    
    if iqr == 0:
        mean = df['y'].mean()
        std = df['y'].std() if df['y'].std() > 0 else 1e-6
        df['z'] = (df['y'] - mean) / std
        df['is_anomaly'] = df['z'].abs() > 3.0
    else:
        df['is_anomaly'] = (df['y'] < lower) | (df['y'] > upper)
    
    med = df['y'].median() if df['y'].median() != 0 else 1e-6
    df['anomaly_mag'] = df.apply(lambda r: (r['y'] / med) if r['is_anomaly'] else np.nan, axis=1)
    
    return df


def build_prophet_model(df: pd.DataFrame, sparse_mode: bool = False) -> Prophet:
    """Create Prophet model with adaptive hyperparams"""
    cv = df['y'].std() / df['y'].mean() if df['y'].mean() > 0 else 0.0
    
    if sparse_mode or len(df) < 60:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive',
            changepoint_prior_scale=0.01,
            interval_width=0.80
        )
    else:
        seasonality_mode = 'multiplicative' if cv > 0.5 else 'additive'
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=0.1,
            interval_width=0.9
        )
        model.add_seasonality(name='monthly', period=30.5, fourier_order=3)
    
    model.fit(df)
    return model


def scale_forecast_to_hist(hist_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Scale forecast to match historical distribution"""
    f = forecast_df.copy()
    hist_mean = hist_df['y'].mean()
    hist_std = hist_df['y'].std()
    raw_mean = f['yhat'].mean()
    raw_std = f['yhat'].std()
    
    if raw_std > 0:
        z = (f['yhat'] - raw_mean) / raw_std
        f['yhat'] = z * (hist_std if hist_std > 0 else 1.0) + (hist_mean if hist_mean is not None else 0.0)
        
        if 'yhat_lower' in f.columns and 'yhat_upper' in f.columns:
            zl = (f['yhat_lower'] - raw_mean) / raw_std
            zu = (f['yhat_upper'] - raw_mean) / raw_std
            f['yhat_lower'] = zl * (hist_std if hist_std > 0 else 1.0) + (hist_mean if hist_mean is not None else 0.0)
            f['yhat_upper'] = zu * (hist_std if hist_std > 0 else 1.0) + (hist_mean if hist_mean is not None else 0.0)
    else:
        f['yhat'] = hist_mean
        if 'yhat_lower' in f.columns:
            f['yhat_lower'] = hist_df['y'].quantile(0.05)
            f['yhat_upper'] = hist_df['y'].quantile(0.95)
    
    return f


def apply_density_pattern(hist_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Apply historical density patterns to forecast"""
    f = forecast_df.copy()
    presence = detect_density_patterns(hist_df)
    dow_frac = presence['dow_frac']
    dom_frac = presence['dom_frac']
    
    max_dow = max(dow_frac.values()) if len(dow_frac) > 0 else 1.0
    
    def adjust_row(row):
        dow = row['ds'].dayofweek
        dom = row['ds'].day
        pf = dow_frac.get(dow, 0.0)
        pf_dom = dom_frac.get(dom, 0.0)
        pf_comb = max(pf, pf_dom)
        scale = (pf_comb / max_dow) if max_dow > 0 else 1.0
        if pf_comb < 0.05:
            return 0.0
        return row['yhat'] * scale
    
    f['yhat'] = f.apply(adjust_row, axis=1)
    if 'yhat_lower' in f.columns:
        f['yhat_lower'] = f['yhat_lower'] * 0.8
        f['yhat_upper'] = f['yhat_upper'] * 1.2
    return f


def inject_anomalies(hist_df: pd.DataFrame, forecast_df: pd.DataFrame, cap_factor: float = 1.05) -> pd.DataFrame:
    """Inject anomalies similar to historical patterns"""
    hist_an = detect_anomalies_iqr(hist_df)
    hist_an_counts = hist_an[hist_an['is_anomaly']].groupby(hist_an['ds'].dt.year).size().to_dict()
    mags = hist_an.loc[hist_an['is_anomaly'], 'anomaly_mag'].dropna()
    if len(mags) == 0:
        mags = pd.Series([2.0])
    
    if hist_an_counts:
        median_count = int(np.median(list(hist_an_counts.values())))
    else:
        median_count = 1
    
    f = forecast_df.copy()
    hist_last = hist_df['ds'].max()
    future_mask = f['ds'] > hist_last
    future = f[future_mask].copy()
    
    if len(future) == 0:
        return f
    
    future_med = future['yhat'].median() if future['yhat'].median() != 0 else 1.0
    future['score'] = (future['yhat'] - future_med).abs()
    
    n_anom = max(1, median_count)
    top_idx = future.sort_values('score', ascending=False).head(n_anom).index.tolist()
    
    sampled_mags = mags.sample(n=len(top_idx), replace=True).values
    
    hist_max = hist_df['y'].max() if hist_df['y'].max() > 0 else np.max([1.0, future['yhat'].max()])
    hard_cap = hist_max * cap_factor
    
    for i, idx in enumerate(top_idx):
        factor = float(sampled_mags[i])
        old = f.loc[idx, 'yhat']
        new = max(old * factor, old + 1.0)
        new = min(new, hard_cap)
        f.loc[idx, 'yhat'] = new
        if 'yhat_upper' in f.columns:
            f.loc[idx, 'yhat_upper'] = min(max(f.loc[idx, 'yhat_upper'], new * 0.90), hard_cap)
        if 'yhat_lower' in f.columns:
            f.loc[idx, 'yhat_lower'] = min(f.loc[idx, 'yhat_lower'], new)
    
    return f


def enforce_historical_cap(hist_df: pd.DataFrame, forecast_df: pd.DataFrame, cap_factor: float = 1.05) -> pd.DataFrame:
    """Enforce cap on forecast values"""
    f = forecast_df.copy()
    hist_max = hist_df['y'].max() if hist_df['y'].max() > 0 else 1.0
    hard_cap = hist_max * cap_factor
    
    f['yhat'] = f['yhat'].clip(upper=hard_cap, lower=0)
    if 'yhat_upper' in f.columns:
        f['yhat_upper'] = f['yhat_upper'].clip(upper=hard_cap, lower=0)
    if 'yhat_lower' in f.columns:
        f['yhat_lower'] = f['yhat_lower'].clip(upper=hard_cap, lower=0)
    
    return f


def apply_march_adjustment(hist_daily: pd.DataFrame, forecast_monthly_agg: pd.DataFrame) -> pd.DataFrame:
    """Apply 10% increase over previous year March for forecasted March months"""
    adjusted = forecast_monthly_agg.copy()
    
    # Get historical monthly data
    hist_monthly = monthly_aggregate(hist_daily, 'y')
    
    # Find all March months in historical data
    hist_monthly['month_num'] = pd.to_datetime(hist_monthly['ds']).dt.month
    hist_march = hist_monthly[hist_monthly['month_num'] == 3].copy()
    
    if len(hist_march) > 0:
        # Get the most recent March value from history
        hist_march_sorted = hist_march.sort_values('ds', ascending=False)
        previous_march_value = hist_march_sorted.iloc[0]['y']
        
        # Calculate adjusted March value (10% increase)
        adjusted_march_value = previous_march_value * 1.10
        
        # Apply to all March months in forecast
        adjusted['month_num'] = pd.to_datetime(adjusted['ds']).dt.month
        march_mask = adjusted['month_num'] == 3
        adjusted.loc[march_mask, 'y'] = adjusted_march_value
        adjusted = adjusted.drop('month_num', axis=1)
    
    return adjusted


def post_process_pipeline(hist_df: pd.DataFrame, raw_forecast: pd.DataFrame) -> pd.DataFrame:
    """Complete post-processing pipeline"""
    scaled = scale_forecast_to_hist(hist_df, raw_forecast)
    density_applied = apply_density_pattern(hist_df, scaled)
    with_anom = inject_anomalies(hist_df, density_applied, cap_factor=1.05)
    with_anom['yhat'] = with_anom['yhat'].clip(lower=0)
    if 'yhat_lower' in with_anom.columns:
        with_anom['yhat_lower'] = with_anom['yhat_lower'].clip(lower=0)
    final = enforce_historical_cap(hist_df, with_anom, cap_factor=1.05)
    return final


@st.cache_data
def convert_df_to_csv(df):
    """Convert dataframe to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')


#######################
# TAB 1: FUTURE FORECAST FUNCTIONS
#######################

def run_forecast(material_code: str, forecast_months: int = 12) -> tuple:
    """Main forecasting pipeline for future forecast"""
    # Load data
    df = load_filtered_data(material_code)
    
    if df.empty:
        raise ValueError(f"No data found for material code: {material_code}")
    
    # Prepare historical data
    hist_daily = aggregate_daily_for_target(df, 'TOTAL_BILLING_QTY_BASE_UNIT')
    
    # Check data sufficiency
    data_points = len(hist_daily)
    if data_points < 100:
        raise ValueError(f"Insufficient data: Only {data_points} day(s) of historical data available. Minimum 100 days required for reliable forecasting.")
    
    # Set warning flag for low data
    low_data_warning = (100 <= data_points <= 500)
    
    # Build and train model
    sparse_mode = (len(hist_daily) < 60)
    model = build_prophet_model(hist_daily, sparse_mode=sparse_mode)
    
    # Generate forecast
    forecast_days = forecast_months * 30
    future = model.make_future_dataframe(periods=forecast_days)
    raw_fc = model.predict(future)
    
    # Post-process
    processed = post_process_pipeline(hist_daily, raw_fc)
    
    # Extract forecast period only
    hist_last = hist_daily['ds'].max()
    final_forecast = processed[processed['ds'] > hist_last].copy()
    
    return hist_daily, final_forecast, material_code, low_data_warning, data_points


def create_monthly_chart_and_table(hist_daily: pd.DataFrame, forecast_daily: pd.DataFrame, material_code: str):
    """Create monthly aggregated chart using plotly and restructured table"""
    # Aggregate to monthly
    hist_monthly = monthly_aggregate(hist_daily, 'y')
    
    forecast_monthly = forecast_daily.copy()
    forecast_monthly['month'] = forecast_monthly['ds'].dt.to_period('M').dt.to_timestamp()
    forecast_monthly_agg = forecast_monthly.groupby('month')['yhat'].sum().reset_index()
    forecast_monthly_agg.columns = ['ds', 'y']
    
    forecast_monthly_agg = apply_march_adjustment(hist_daily, forecast_monthly_agg)
    
    # Create plotly chart
    fig = go.Figure()
    
    # Add historical trace
    fig.add_trace(go.Scatter(
        x=hist_monthly['ds'],
        y=hist_monthly['y'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=8, symbol='circle')
    ))
    
    # Add forecast trace
    fig.add_trace(go.Scatter(
        x=forecast_monthly_agg['ds'],
        y=forecast_monthly_agg['y'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#E63946', width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    # Add vertical line at forecast start
    forecast_start = hist_monthly['ds'].max()
    
    fig.add_shape(
        type="line",
        x0=forecast_start,
        x1=forecast_start,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="gray", width=2, dash="dash")
    )
    
    fig.add_annotation(
        x=forecast_start,
        y=1,
        yref="paper",
        text="Forecast Start",
        showarrow=False,
        yshift=10,
        font=dict(size=11, color="gray")
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Monthly Total Billing Quantity - {material_code}',
            font=dict(size=22, color='#1f4788', family='Arial Black')
        ),
        xaxis_title='Month',
        yaxis_title='Total Billing Quantity (Base Unit)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=13)
        ),
        height=600,
        template='plotly_white',
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Add gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    # Create restructured combined table with TYPE column
    hist_monthly_table = hist_monthly.copy()
    hist_monthly_table.columns = ['DATE', 'TOTAL_BILLING_QTY_BASE_UNIT']
    hist_monthly_table['TYPE'] = 'Historical'
    hist_monthly_table['MATERIAL_CODE'] = material_code
    
    forecast_monthly_table = forecast_monthly_agg.copy()
    forecast_monthly_table.columns = ['DATE', 'TOTAL_BILLING_QTY_BASE_UNIT']
    forecast_monthly_table['TYPE'] = 'Forecasted'
    forecast_monthly_table['MATERIAL_CODE'] = material_code
    
    combined_table = pd.concat([hist_monthly_table, forecast_monthly_table], ignore_index=True)
    combined_table = combined_table.sort_values('DATE').reset_index(drop=True)
    combined_table = combined_table[['DATE', 'TOTAL_BILLING_QTY_BASE_UNIT', 'TYPE', 'MATERIAL_CODE']]
    
    # Format date and numbers
    combined_table['DATE'] = pd.to_datetime(combined_table['DATE']).dt.strftime('%Y-%m')
    combined_table['TOTAL_BILLING_QTY_BASE_UNIT'] = combined_table['TOTAL_BILLING_QTY_BASE_UNIT'].round(0).astype(int)
    
    return fig, combined_table


#######################
# TAB 2: HISTORICAL COMPARISON FUNCTIONS
#######################

def run_forecast_fy_based(material_code: str) -> tuple:
    """
    Main forecasting pipeline for fiscal year based analysis.
    Trains on FY 2022-23 and 2023-24, predicts for FY 2024-25 starting from April 2024
    """
    # Load all data
    df = load_filtered_data(material_code)
    
    if df.empty:
        raise ValueError(f"No data found for material code: {material_code}")
    
    # Prepare historical data
    all_daily = aggregate_daily_for_target(df, 'TOTAL_BILLING_QTY_BASE_UNIT')
    
    # Define fiscal year boundaries
    fy_2022_23_start, fy_2022_23_end = get_fy_date_range('2022-23')
    fy_2023_24_start, fy_2023_24_end = get_fy_date_range('2023-24')
    forecast_start = get_forecast_start_date_tab2()  # April 1, 2024
    fy_2024_25_end = datetime(2025, 3, 31)  # End of FY 2024-25
    
    # Split data into training (FY 2022-23, 2023-24) and actual (FY 2024-25)
    train_data = all_daily[
        (all_daily['ds'] >= fy_2022_23_start) & 
        (all_daily['ds'] <= fy_2023_24_end)
    ].copy()
    
    # Get actual data for FY 2024-25 (April 2024 - March 2025)
    actual_2024_25 = all_daily[
        (all_daily['ds'] >= forecast_start) & 
        (all_daily['ds'] <= fy_2024_25_end)
    ].copy()
    
    # Check data sufficiency
    data_points = len(train_data)
    if data_points < 100:
        raise ValueError(f"Insufficient training data: Only {data_points} day(s) of historical data available in FY 2022-23 and 2023-24. Minimum 100 days required for reliable forecasting.")
    
    # Set warning flag for low data
    low_data_warning = (100 <= data_points <= 500)
    
    # Build and train model on FY 2022-23 and 2023-24
    sparse_mode = (len(train_data) < 100)
    model = build_prophet_model(train_data, sparse_mode=sparse_mode)
    
    # Generate forecast from April 2024 to March 2025
    forecast_days = (fy_2024_25_end - forecast_start).days + 1
    future_dates = pd.date_range(start=forecast_start, periods=forecast_days, freq='D')
    future = pd.DataFrame({'ds': future_dates})
    
    raw_fc = model.predict(future)
    
    # Post-process
    processed = post_process_pipeline(train_data, raw_fc)
    
    # Extract forecast for FY 2024-25 (April 2024 - March 2025)
    forecast_2024_25 = processed[
        (processed['ds'] >= forecast_start) & 
        (processed['ds'] <= fy_2024_25_end)
    ].copy()
    
    return train_data, forecast_2024_25, actual_2024_25, material_code, low_data_warning, data_points


def create_comparison_chart_and_table(train_data: pd.DataFrame, 
                                      forecast_2024_25: pd.DataFrame,
                                      actual_2024_25: pd.DataFrame,
                                      material_code: str):
    """
    Create chart showing training data (FY 2022-23, 2023-24), 
    actual FY 2024-25, and predicted FY 2024-25.
    Table shows only FY 2024-25 actual vs predicted comparison.
    """
    # Aggregate all to monthly
    train_monthly = monthly_aggregate(train_data, 'y')
    
    # Forecast monthly
    forecast_monthly = forecast_2024_25.copy()
    forecast_monthly['month'] = forecast_monthly['ds'].dt.to_period('M').dt.to_timestamp()
    forecast_monthly_agg = forecast_monthly.groupby('month')['yhat'].sum().reset_index()
    forecast_monthly_agg.columns = ['ds', 'y']

    forecast_monthly_agg = apply_march_adjustment(train_data, forecast_monthly_agg)
    
    # Actual monthly
    actual_monthly = monthly_aggregate(actual_2024_25, 'y') if not actual_2024_25.empty else pd.DataFrame(columns=['ds', 'y'])
    
    # Create plotly chart
    fig = go.Figure()
    
    # Add training data trace (FY 2022-23 and 2023-24)
    fig.add_trace(go.Scatter(
        x=train_monthly['ds'],
        y=train_monthly['y'],
        mode='lines+markers',
        name='Training Data (FY 2022-23, 2023-24)',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=8, symbol='circle')
    ))
    
    # Add actual FY 2024-25 trace
    if not actual_monthly.empty:
        fig.add_trace(go.Scatter(
            x=actual_monthly['ds'],
            y=actual_monthly['y'],
            mode='lines+markers',
            name='Actual (FY 2024-25)',
            line=dict(color='#06D6A0', width=3),
            marker=dict(size=10, symbol='square')
        ))
    
    # Add forecast FY 2024-25 trace
    fig.add_trace(go.Scatter(
        x=forecast_monthly_agg['ds'],
        y=forecast_monthly_agg['y'],
        mode='lines+markers',
        name='Predicted (FY 2024-25)',
        line=dict(color='#E63946', width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    # Add vertical line at forecast start (April 2024)
    forecast_start = train_monthly['ds'].max()
    
    fig.add_shape(
        type="line",
        x0=forecast_start,
        x1=forecast_start,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="gray", width=2, dash="dash")
    )
    
    fig.add_annotation(
        x=forecast_start,
        y=1,
        yref="paper",
        text="Forecast Start (April 2024)",
        showarrow=False,
        yshift=10,
        font=dict(size=11, color="gray")
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Sales Forecast: Actual vs Predicted - {material_code}',
            font=dict(size=22, color='#1f4788', family='Arial Black')
        ),
        xaxis_title='Month',
        yaxis_title='Total Billing Quantity (Base Unit)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=13)
        ),
        height=600,
        template='plotly_white',
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Add gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    # Create comparison table for FY 2024-25 ONLY
    # Merge actual and predicted on month
    if not actual_monthly.empty:
        comparison = actual_monthly.merge(
            forecast_monthly_agg, 
            on='ds', 
            how='outer', 
            suffixes=('_actual', '_predicted')
        )
        comparison = comparison.rename(columns={
            'y_actual': 'BILLING_QTY_BASE_UNIT',
            'y_predicted': 'FORECASTED_BILLING_QTY_BASE_UNIT'
        })
    else:
        # If no actual data, just show predicted
        comparison = forecast_monthly_agg.copy()
        comparison['BILLING_QTY_BASE_UNIT'] = np.nan
        comparison = comparison.rename(columns={'y': 'FORECASTED_BILLING_QTY_BASE_UNIT'})
    
    # Calculate difference
    comparison['DIFFERENCE'] = comparison['BILLING_QTY_BASE_UNIT'] - comparison['FORECASTED_BILLING_QTY_BASE_UNIT']
    
    # Format and reorder columns
    comparison['DATE'] = pd.to_datetime(comparison['ds']).dt.strftime('%m-%Y')
    comparison = comparison[['DATE', 'BILLING_QTY_BASE_UNIT', 'FORECASTED_BILLING_QTY_BASE_UNIT', 'DIFFERENCE']]
    
    # Round numeric columns
    comparison['BILLING_QTY_BASE_UNIT'] = comparison['BILLING_QTY_BASE_UNIT'].round(0)
    comparison['FORECASTED_BILLING_QTY_BASE_UNIT'] = comparison['FORECASTED_BILLING_QTY_BASE_UNIT'].round(0)
    comparison['DIFFERENCE'] = comparison['DIFFERENCE'].round(0)
    
    # Sort by date
    comparison = comparison.sort_values('DATE').reset_index(drop=True)
    
    return fig, comparison


#######################
# STREAMLIT UI
#######################

def main():
    # Main content area - Header
    st.markdown('<div class="main-title">SUPREME FORECAST DASHBOARD</div>', unsafe_allow_html=True)
    
    # Sidebar with logo at TOP and all inputs
    with st.sidebar:
        # Display logo at the very top
        logo_path = "Supreme Industries - Wikipedia.png"
        if os.path.exists(logo_path):
            st.image(logo_path, use_container_width=True)
        else:
            st.info("Place 'Supreme Industries - Wikipedia.png' in the app directory to display logo")
        
        st.markdown("---")
        st.subheader("Forecast Configuration")
        
        # Load material groups
        with st.spinner("Loading material groups..."):
            try:
                material_groups = load_material_groups()
            except Exception as e:
                st.error(f"Error loading material groups: {str(e)}")
                return
        
        # Material group selection
        selected_group = st.selectbox(
            "Material Group",
            options=material_groups,
            index=0,
            help="Select a material group (sorted by data volume - more data at top)"
        )
        
        # Load material codes for selected group
        with st.spinner("Loading material codes..."):
            try:
                material_codes = load_material_codes_by_group(selected_group)
            except Exception as e:
                st.error(f"Error loading material codes: {str(e)}")
                return
        
        # Material code selection
        selected_material = st.selectbox(
            "Material Code",
            options=material_codes,
            index=0,
            help="Select a material code (sorted by data volume - more data at top)"
        )
        
        # Forecast period slider
        forecast_months = st.slider(
            "Forecast Period (Months)",
            min_value=1,
            max_value=12,
            value=12,
            step=1,
            help="Number of months to forecast into the future"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Single Generate button for both tabs
        generate_button = st.button("Generate Forecast", use_container_width=True)
        
        st.markdown("---")
        st.caption("Groups and material codes with more historical data appear first in the dropdowns")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Future Forecast", "Historical Comparison (FY Analysis)"])
    
    #######################
    # TAB 1: FUTURE FORECAST
    #######################
    with tab1:
        if generate_button:
            if not selected_material:
                st.warning("Please select a material code")
            else:
                with st.spinner("Generating future forecast... This may take a moment."):
                    try:
                        # Run forecast
                        hist_daily, forecast_daily, material_code, low_data_warning, data_points = run_forecast(
                            selected_material, 
                            forecast_months=forecast_months
                        )
                        
                        # Create chart and table
                        fig, combined_table = create_monthly_chart_and_table(
                            hist_daily, 
                            forecast_daily, 
                            material_code
                        )
                        
                        # Display warning if low data
                        if low_data_warning:
                            st.warning(
                                f"⚠️ **Limited Historical Data Warning**\n\n"
                                f"The model was trained on only **{data_points} days** of historical data (100-500 days range). "
                                f"While forecasts have been generated, **the predicted values may not be highly reliable** "
                                f"due to the limited training data available."
                            )
                        
                        # Display success message
                        st.success(
                            f"Forecast generated successfully for material code: **{material_code}** "
                            f"(Training data: {data_points} days | Forecast period: {forecast_months} months).\n\n"
                            "**KPI Information:** Billing quantity (in base units) is forecasted on a month-wise basis " 
                            "for each material code, with analysis performed based on the selected material group "
                            "and the specific material code."
                        )
    
                        st.markdown("---")
                        
                        # Show CSV table FIRST
                        st.subheader("Monthly Forecast Data")
                        
                        # Display dataframe
                        st.dataframe(
                            combined_table,
                            use_container_width=True,
                            height=400,
                            hide_index=True
                        )
                        
                        # Download button for CSV
                        csv_data = convert_df_to_csv(combined_table)
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.download_button(
                                label="📥 Download Forecast Data (CSV)",
                                data=csv_data,
                                file_name=f"forecast_{material_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        st.markdown("---")
                        
                        # Show plotly chart SECOND
                        st.subheader("Forecast Visualization")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except ValueError as ve:
                        # Handle specific ValueError messages
                        error_msg = str(ve)
                        
                        if "Insufficient data" in error_msg:
                            import re
                            days_match = re.search(r'Only (\d+) day', error_msg)
                            days_available = days_match.group(1) if days_match else "a few"
                            
                            st.error(
                                f"**Insufficient Historical Data**\n\n"
                                f"The material code **'{selected_material}'** currently has only **{days_available}** day(s) of historical data available.\n\n"
                                f"**To generate a reliable forecast, we need at least 100 days of historical data.**\n\n"
                                f"**Suggestion:** Please select a different material code from the sidebar dropdown. "
                                f"Material codes are sorted by data volume, so codes at the top have more historical data available."
                            )
                        elif "No data found" in error_msg:
                            st.warning(
                                f"**No Data Available**\n\n"
                                f"No historical sales data was found for material code **'{selected_material}'**.\n\n"
                                f"**Suggestion:** Please select a different material code from the sidebar dropdown menu."
                            )
                        else:
                            st.error(f"**Validation Error**\n\n{error_msg}")
                            
                    except Exception as e:
                        st.error(
                            f"**An unexpected error occurred**\n\n"
                            f"We encountered an issue while generating the forecast. "
                            f"Please try selecting a different material code or contact support if the issue persists."
                        )
                        with st.expander("Show error details"):
                            st.exception(e)
        else:
            # Initial state - show instructions
            st.info(
                "**Welcome to the Future Forecast Tab!**\n\n"
                "**To get started:**\n"
                "1. Select a material code from the sidebar dropdown\n"
                "2. Choose your forecast period (in months)\n"
                "3. Click the 'Generate Forecast' button\n\n"
                "**Important Notes:**\n"
                "- Minimum **100 days** of historical data required\n"
                "- Material codes with more data appear first in the dropdown"
            )
            
            
    
    #######################
    # TAB 2: HISTORICAL COMPARISON
    #######################
    with tab2:
        if generate_button:
            if not selected_material:
                st.warning("Please select a material code")
            else:
                with st.spinner("Generating historical comparison... This may take a moment."):
                    try:
                        # Run forecast
                        train_data, forecast_2024_25, actual_2024_25, material_code, low_data_warning, data_points = run_forecast_fy_based(
                            selected_material
                        )
                        
                        # Create chart and table
                        fig, comparison_table = create_comparison_chart_and_table(
                            train_data, 
                            forecast_2024_25,
                            actual_2024_25,
                            material_code
                        )
                        
                        # Display warning if low data
                        if low_data_warning:
                            st.warning(
                                f"⚠️ **Limited Historical Data Warning**\n\n"
                                f"The model was trained on only **{data_points} days** of historical data (100-500 days range). "
                                f"While forecasts have been generated, **the predicted values may not be highly reliable** "
                                f"due to the limited training data available."
                            )
                        
                        # Display success message
                        st.success(f"Forecast generated successfully for material code: **{material_code}** (Training data: {data_points} days)")
                        
                        # Show metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Training Data Points (FY 2022-24)", 
                                f"{len(train_data)} days",
                                help="FY 2022-23 and 2023-24"
                            )
                        with col2:
                            st.metric(
                                "Actual Data Points (FY 2024-25)", 
                                f"{len(actual_2024_25)} days",
                                help="April 2024 - March 2025"
                            )
                        with col3:
                            if not comparison_table['DIFFERENCE'].isna().all():
                                avg_diff = comparison_table['DIFFERENCE'].mean()
                                st.metric(
                                    "Avg Monthly Difference", 
                                    f"{avg_diff:,.0f}",
                                    help="Actual - Predicted"
                                )
                            else:
                                st.metric("Avg Monthly Difference", "N/A")
                        
                        st.markdown("---")
                        
                        # Show CSV table FIRST
                        st.subheader("FY 2024-25: Actual vs Predicted Comparison")
                        st.caption("Forecast Period: April 2024 - March 2025")
                        
                        # Display dataframe
                        st.dataframe(
                            comparison_table,
                            use_container_width=True,
                            height=400,
                            hide_index=True
                        )
                        
                        # Download button for CSV
                        csv_data = convert_df_to_csv(comparison_table)
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.download_button(
                                label="📥 Download Comparison Data (CSV)",
                                data=csv_data,
                                file_name=f"forecast_comparison_{material_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        st.markdown("---")
                        
                        # Show plotly chart SECOND
                        st.subheader("Forecast Visualization")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except ValueError as ve:
                        # Handle specific ValueError messages
                        error_msg = str(ve)
                        
                        if "Insufficient" in error_msg:
                            import re
                            days_match = re.search(r'Only (\d+) day', error_msg)
                            days_available = days_match.group(1) if days_match else "a few"
                            
                            st.error(
                                f"**Insufficient Historical Data**\n\n"
                                f"The material code **'{selected_material}'** has only **{days_available}** day(s) of data in FY 2022-23 and 2023-24.\n\n"
                                f"**To generate a reliable forecast, we need at least 100 days of historical data.**\n\n"
                                f"**Suggestion:** Please select a different material code from the sidebar dropdown. "
                                f"Material codes are sorted by data volume, so codes at the top have more historical data available."
                            )
                        elif "No data found" in error_msg:
                            st.warning(
                                f"**No Data Available**\n\n"
                                f"No historical sales data was found for material code **'{selected_material}'**.\n\n"
                                f"**Suggestion:** Please select a different material code from the sidebar dropdown menu."
                            )
                        else:
                            st.error(f"**Validation Error**\n\n{error_msg}")
                            
                    except Exception as e:
                        st.error(
                            f"**An unexpected error occurred**\n\n"
                            f"We encountered an issue while generating the forecast. "
                            f"Please try selecting a different material code or contact support if the issue persists."
                        )
                        with st.expander("Show error details"):
                            st.exception(e)
        else:
            # Initial state - show instructions
            st.info(
                "**Welcome to the Historical Comparison Tab!**\n\n"
                "**How it works:**\n"
                "1. **Training:** Model learns from FY 2022-23 and FY 2023-24 data\n"
                "2. **Prediction:** Forecasts generated for FY 2024-25 (April 2024 - March 2025)\n"
                "3. **Comparison:** Actual vs Predicted values are compared\n\n"
                "**To get started:**\n"
                "1. Select a material code from the sidebar\n"
                "2. Click the 'Generate Forecast' button\n\n"
                "**Requirements:** Minimum 100 days of historical data"
            )


if __name__ == "__main__":
    main()
