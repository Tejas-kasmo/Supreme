import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from prophet import Prophet
import pickle

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')


# Target material groups
MATERIAL_GROUPS = ["M-C", "M-G", "M-R", "M-S", "P-C", "P-F", "P-G", "P-R", "P-S", "R-T"]

MARCH_SPIKE_GROUPS = ["M-R", "M-S", "P-C", "P-S", "R-T"]


#######################
# Utility / IO
#######################
def load_data(path: str) -> pd.DataFrame:
    print(f"Loading: {path}")
    df = pd.read_csv(path)
    if 'BILLING_DATE' in df.columns:
        df['BILLING_DATE'] = pd.to_datetime(df['BILLING_DATE'])
    else:
        raise ValueError("training data must contain 'BILLING_DATE'")
    print(f"Rows loaded: {len(df):,}")
    return df


def make_output_dir(base_dir: str, material_group: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"{material_group}_{timestamp}"
    out_dir = os.path.join(base_dir, folder)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


#######################
# Aggregation helpers
#######################
def aggregate_daily_for_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Returns daily aggregated dataframe with columns ds (date) and y (value)
    """
    tmp = df[['BILLING_DATE', target_col]].copy()
    tmp = tmp.groupby('BILLING_DATE')[target_col].sum().reset_index()
    tmp.columns = ['ds', 'y']
    tmp = tmp.sort_values('ds').reset_index(drop=True)
    return tmp


def monthly_aggregate(df: pd.DataFrame, method: str = 'sum') -> pd.DataFrame:
    tmp = df.copy()
    tmp['month'] = tmp['ds'].dt.to_period('M').dt.to_timestamp()
    if method == 'sum':
        m = tmp.groupby('month')['y'].sum().reset_index().rename(columns={'month': 'ds'})
    else:
        m = tmp.groupby('month')['y'].mean().reset_index().rename(columns={'month': 'ds'})
    return m


#######################
# Pattern & anomaly detection
#######################
def detect_density_patterns(hist_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze how many non-zero days exist by day_of_week and day_of_month across years.
    Return dict of presence fractions and counts, per dow and dom and per-year counts.
    """
    df = hist_df.copy()
    df['year'] = df['ds'].dt.year
    df['dow'] = df['ds'].dt.dayofweek  # 0=Mon
    df['dom'] = df['ds'].dt.day

    presence = {}
    # presence fraction per DOW
    dow = df.groupby('dow').apply(lambda g: (g['y'] > 0).sum() / len(g)).to_dict()
    presence['dow_frac'] = dow
    # presence fraction per DOM (1..31)
    dom = df.groupby('dom').apply(lambda g: (g['y'] > 0).sum() / len(g)).to_dict()
    presence['dom_frac'] = dom

    # per-year counts
    years = sorted(df['year'].unique())
    per_year_counts = {}
    for y in years:
        per_year_counts[y] = len(df[df['year'] == y])
    presence['per_year_counts'] = per_year_counts

    return presence


def detect_anomalies_iqr(hist_df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Mark anomalies using IQR method. Returns a copy of hist_df with 'is_anomaly' boolean.
    """
    df = hist_df.copy()
    q1 = df['y'].quantile(0.25)
    q3 = df['y'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    # fallback for very low variance
    if iqr == 0:
        mean = df['y'].mean()
        std = df['y'].std() if df['y'].std() > 0 else 1e-6
        df['z'] = (df['y'] - mean) / std
        df['is_anomaly'] = df['z'].abs() > 3.0
    else:
        df['is_anomaly'] = (df['y'] < lower) | (df['y'] > upper)

    # magnitude ratio for anomalies (y / median)
    med = df['y'].median() if df['y'].median() != 0 else 1e-6
    df['anomaly_mag'] = df.apply(lambda r: (r['y'] / med) if r['is_anomaly'] else np.nan, axis=1)

    return df


#######################
# Model training & forecasting
#######################
def build_prophet_model(df: pd.DataFrame, sparse_mode: bool = False, holidays: Optional[pd.DataFrame] = None) -> Prophet:
    """
    Create Prophet model with adaptive hyperparams depending on sparsity and variability.
    """
    cv = df['y'].std() / df['y'].mean() if df['y'].mean() > 0 else 0.0

    if sparse_mode or len(df) < 60:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive',
            changepoint_prior_scale=0.01,
            holidays=holidays,
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
            holidays=holidays,
            interval_width=0.9
        )
        model.add_seasonality(name='monthly', period=30.5, fourier_order=3)

    model.fit(df)
    return model


def generate_future_and_predict(model: Prophet, df: pd.DataFrame, days: int = 365) -> pd.DataFrame:
    future = model.make_future_dataframe(periods=days)
    fc = model.predict(future)
    return fc


#######################
# Post-processing
#######################
def scale_forecast_to_hist(hist_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale forecast to match historical distribution with CV-based confidence intervals.
    - Low CV (<0.5): Tight intervals (±10%)
    - Medium CV (0.5-0.8): Normal intervals (±20%)
    - High CV (>0.8): Wide intervals (±30%)
    """
    f = forecast_df.copy()
    hist_mean = hist_df['y'].mean()
    hist_std = hist_df['y'].std()
    raw_mean = f['yhat'].mean()
    raw_std = f['yhat'].std()
    
    # Calculate CV for interval width adjustment
    cv = hist_std / hist_mean if hist_mean > 0 else 0.5
    if cv < 0.5:
        interval_factor = 0.10  # Tight intervals
    elif cv < 0.8:
        interval_factor = 0.20  # Normal intervals
    else:
        interval_factor = 0.30  # Wide intervals

    if raw_std > 0:
        z = (f['yhat'] - raw_mean) / raw_std
        f['yhat'] = z * (hist_std if hist_std > 0 else 1.0) + (hist_mean if hist_mean is not None else 0.0)

        # Apply CV-based confidence interval width
        if 'yhat_lower' in f.columns and 'yhat_upper' in f.columns:
            f['yhat_lower'] = f['yhat'] * (1 - interval_factor)
            f['yhat_upper'] = f['yhat'] * (1 + interval_factor)
    else:
        f['yhat'] = hist_mean
        if 'yhat_lower' in f.columns:
            f['yhat_lower'] = hist_df['y'].quantile(0.05)
            f['yhat_upper'] = hist_df['y'].quantile(0.95)

    return f


def apply_density_pattern(hist_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce forecast predictions on days that historically had low presence.
    """
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


def apply_month_end_boost(hist_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply month-end boost based on historical month-end patterns.
    Analysis shows all material groups have 1.5-3x higher values on month-end days.
    """
    # Calculate historical month-end boost factor
    hist = hist_df.copy()
    hist['month_end'] = hist['ds'].dt.is_month_end
    me_mean = hist[hist['month_end']]['y'].mean() if hist['month_end'].any() else 0
    non_me_mean = hist[~hist['month_end']]['y'].mean()
    
    if non_me_mean > 0 and me_mean > non_me_mean:
        boost_factor = me_mean / non_me_mean
        # Cap boost factor at 3.0 to prevent extreme values
        boost_factor = min(boost_factor, 3.0)
    else:
        boost_factor = 1.0
    
    # Apply boost to forecast month-end days
    f = forecast_df.copy()
    f['is_month_end'] = f['ds'].dt.is_month_end
    
    # Apply boost only to yhat and yhat_upper for month-end days
    f['yhat'] = f.apply(lambda r: r['yhat'] * boost_factor if r['is_month_end'] else r['yhat'], axis=1)
    
    if 'yhat_upper' in f.columns:
        f['yhat_upper'] = f.apply(
            lambda r: r['yhat_upper'] * boost_factor if r['is_month_end'] else r['yhat_upper'], axis=1
        )
    
    f = f.drop(columns=['is_month_end'])
    return f


def apply_march_spike_boost(hist_df: pd.DataFrame, forecast_df: pd.DataFrame, material_group: str) -> pd.DataFrame:
    """
    Apply March spike boost for specific groups that show consistent YoY growth in March.
    Groups: M-R, M-S, P-C, P-S, R-T
    
    Logic: Calculate average YoY % growth for March from historical data,
    then apply that growth rate to the forecasted March values.
    """
    if material_group not in MARCH_SPIKE_GROUPS:
        return forecast_df
    
    f = forecast_df.copy()
    hist = hist_df.copy()
    
    # Get March totals for each year from historical data
    hist['year'] = hist['ds'].dt.year
    hist['month'] = hist['ds'].dt.month
    march_data = hist[hist['month'] == 3].groupby('year')['y'].sum()
    
    if len(march_data) < 2:
        return f
    
    # Calculate YoY growth rates for March
    march_values = march_data.values
    growths = []
    for i in range(1, len(march_values)):
        if march_values[i-1] > 0:
            g = (march_values[i] - march_values[i-1]) / march_values[i-1]
            growths.append(g)
    
    if len(growths) == 0:
        return f
    
    # Use average growth rate, capped at 50% to avoid extreme extrapolation
    avg_growth = min(np.mean(growths), 0.50)
    
    # Get the last March value from historical data
    last_march_value = march_values[-1]
    last_march_year = march_data.index[-1]
    
    # Apply growth to forecast March days
    f['month'] = f['ds'].dt.month
    f['year'] = f['ds'].dt.year
    
    # For each forecast year's March, apply cumulative growth
    forecast_years = f[f['month'] == 3]['year'].unique()
    
    for fc_year in forecast_years:
        years_ahead = fc_year - last_march_year
        if years_ahead > 0:
            # Calculate expected growth factor
            growth_factor = (1 + avg_growth) ** years_ahead
            
            # Get current March forecast mean for this year
            march_mask = (f['month'] == 3) & (f['year'] == fc_year)
            current_march_mean = f.loc[march_mask, 'yhat'].mean()
            
            # Calculate expected March mean based on historical growth
            days_in_march = march_mask.sum()
            expected_daily_mean = (last_march_value * growth_factor) / days_in_march if days_in_march > 0 else current_march_mean
            
            # If model underestimates, boost to expected level
            if current_march_mean > 0 and expected_daily_mean > current_march_mean:
                boost_ratio = expected_daily_mean / current_march_mean
                f.loc[march_mask, 'yhat'] = f.loc[march_mask, 'yhat'] * boost_ratio
                if 'yhat_upper' in f.columns:
                    f.loc[march_mask, 'yhat_upper'] = f.loc[march_mask, 'yhat_upper'] * boost_ratio
    
    f = f.drop(columns=['month', 'year'])
    return f


def apply_pf_stabilization(hist_df: pd.DataFrame, forecast_df: pd.DataFrame, material_group: str) -> pd.DataFrame:
    """
    Special handling for P-F which shows a boom-bust pattern:
    - FY2022-23: 1.0M (ramp-up)
    - FY2023-24: 2.3M (peak)  
    - FY2024-25: 1.4M (correction/stabilization)
    
    Strategy:
    1. Use recent FY (last 12 months) as baseline - this reflects the "new normal"
    2. Apply monthly seasonality pattern from historical data
    3. Add variability matching the high CV (0.85)
    """
    if material_group != 'P-F':
        return forecast_df
    
    f = forecast_df.copy()
    hist = hist_df.copy()
    hist = hist.sort_values('ds')
    
    # Get recent 12 months as baseline (the "new normal" after correction)
    recent_12mo = hist.tail(365)
    recent_daily_mean = recent_12mo['y'].mean()
    recent_daily_std = recent_12mo['y'].std()
    
    # Calculate monthly seasonality factors from ALL historical data
    hist['month'] = hist['ds'].dt.month
    monthly_means = hist.groupby('month')['y'].mean()
    overall_mean = hist['y'].mean()
    
    # Seasonality factors: how much each month differs from average
    seasonality_factors = (monthly_means / overall_mean).to_dict()
    
    # Get forecast start date
    hist_last = hist['ds'].max()
    
    # Apply to forecast
    f['month'] = f['ds'].dt.month
    f['is_future'] = f['ds'] > hist_last
    
    # For future dates, create realistic forecast
    np.random.seed(42)  # For reproducibility
    
    for idx in f[f['is_future']].index:
        month = f.loc[idx, 'month']
        
        # Base value: recent mean adjusted for seasonality
        seasonal_factor = seasonality_factors.get(month, 1.0)
        base_value = recent_daily_mean * seasonal_factor
        
        # Add realistic variability (use 40% of historical std to avoid wild swings)
        variability = np.random.normal(0, recent_daily_std * 0.4)
        
        # Apply with bounds
        new_value = max(base_value + variability, recent_daily_mean * 0.3)  # Floor at 30% of mean
        new_value = min(new_value, recent_daily_mean * 2.5)  # Cap at 250% of mean
        
        f.loc[idx, 'yhat'] = new_value
        
        # Adjust confidence intervals
        if 'yhat_lower' in f.columns:
            f.loc[idx, 'yhat_lower'] = new_value * 0.6
        if 'yhat_upper' in f.columns:
            f.loc[idx, 'yhat_upper'] = new_value * 1.4
    
    f = f.drop(columns=['month', 'is_future'])
    return f


def inject_anomalies(hist_df: pd.DataFrame, forecast_df: pd.DataFrame, cap_factor: float = 0.90) -> pd.DataFrame:
    """
    Inject anomalies into forecast similar to historical patterns.
    """
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


def enforce_historical_cap(hist_df: pd.DataFrame, forecast_df: pd.DataFrame, base_cap_factor: float = 0.90) -> pd.DataFrame:
    """
    Dynamic cap based on detected growth trend.
    - Strong growth (>30%): Allow up to 1.5x historical max
    - Moderate growth (>10%): Allow up to 1.2x historical max  
    - Stable/declining: Use base cap factor (0.90x)
    
    Analysis shows all material groups growing 23-49%, so dynamic cap is essential.
    """
    f = forecast_df.copy()
    
    # Detect growth trend from monthly aggregates
    hist = hist_df.copy()
    hist['month'] = hist['ds'].dt.to_period('M')
    monthly = hist.groupby('month')['y'].sum()
    
    if len(monthly) >= 6:
        first_half = monthly.head(len(monthly) // 2).mean()
        second_half = monthly.tail(len(monthly) // 2).mean()
        growth_pct = (second_half - first_half) / first_half * 100 if first_half > 0 else 0
    else:
        growth_pct = 0
    
    # Dynamic cap factor based on growth trend
    if growth_pct > 30:
        cap_factor = 1.5  # Allow 50% above historical max for strong growth
    elif growth_pct > 10:
        cap_factor = 1.2  # Allow 20% above historical max for moderate growth
    else:
        cap_factor = base_cap_factor  # Use conservative cap for stable trends
    
    hist_max = hist_df['y'].max() if hist_df['y'].max() > 0 else 1.0
    hard_cap = hist_max * cap_factor

    f['yhat'] = f['yhat'].clip(upper=hard_cap)
    if 'yhat_upper' in f.columns:
        f['yhat_upper'] = f['yhat_upper'].clip(upper=hard_cap)
    if 'yhat_lower' in f.columns:
        f['yhat_lower'] = f['yhat_lower'].clip(upper=hard_cap)
        f['yhat_lower'] = f['yhat_lower'].clip(lower=0)
    f['yhat'] = f['yhat'].clip(lower=0)

    return f


def post_process_pipeline(hist_df: pd.DataFrame, raw_forecast: pd.DataFrame, material_group: str = None) -> pd.DataFrame:
    """
    Full post-processing pipeline with trend-aware adjustments:
    1. Scale forecast to match historical distribution (CV-based intervals)
    2. Apply density patterns (day-of-week/month adjustments)
    3. Apply month-end boost (1.5-3x based on historical pattern)
    4. Apply March spike boost for specific groups (M-R, M-S, P-C, P-S, R-T)
    5. Apply P-F stabilization to prevent unrealistic decline
    6. Inject anomalies similar to historical patterns
    7. Enforce growth-aware dynamic cap
    """
    # Step 1: Scale to historical distribution with CV-based confidence intervals
    scaled = scale_forecast_to_hist(hist_df, raw_forecast)
    
    # Step 2: Apply density patterns
    density_applied = apply_density_pattern(hist_df, scaled)
    
    # Step 3: Apply month-end boost
    month_end_boosted = apply_month_end_boost(hist_df, density_applied)
    
    # Step 4: Apply March spike boost for specific groups
    march_boosted = apply_march_spike_boost(hist_df, month_end_boosted, material_group)
    
    # Step 5: Apply P-F stabilization
    pf_stabilized = apply_pf_stabilization(hist_df, march_boosted, material_group)
    
    # Step 6: Inject anomalies
    with_anom = inject_anomalies(hist_df, pf_stabilized, cap_factor=0.90)
    with_anom['yhat'] = with_anom['yhat'].clip(lower=0)
    if 'yhat_lower' in with_anom.columns:
        with_anom['yhat_lower'] = with_anom['yhat_lower'].clip(lower=0)
    
    # Step 7: Enforce dynamic cap (growth-aware)
    final = enforce_historical_cap(hist_df, with_anom)
    return final


#######################
# Plotting
#######################
def plot_daily_and_monthly(historical_df: pd.DataFrame, forecast_df: pd.DataFrame,
                          out_dir: str, material_group: str) -> Tuple[str, str]:
    """
    Create daily and monthly plots for a material group
    """
    daily_path = os.path.join(out_dir, f"{material_group}_daily_forecast.png")
    monthly_path = os.path.join(out_dir, f"{material_group}_monthly_forecast.png")

    # DAILY plot
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Top: full historical + forecast
    ax1 = axes[0]
    ax1.plot(historical_df['ds'], historical_df['y'], label='Historical', linewidth=0.9, color='#2E86AB')
    ax1.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast (12mo)', linewidth=1.5, color='#E63946')
    if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
        ax1.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'], alpha=0.2, color='#E63946')

    ax1.axvline(historical_df['ds'].max(), linestyle='--', color='gray', label='Forecast Start')
    ax1.set_title(f"{material_group} - TOTAL_BILLING_QTY_BASE_UNIT - Historical + Forecast (Daily)")
    ax1.set_ylabel('TOTAL_BILLING_QTY_BASE_UNIT')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Bottom: zoom last 180 days + forecast
    ax2 = axes[1]
    zoom_start = historical_df['ds'].max() - pd.Timedelta(days=180)
    recent = historical_df[historical_df['ds'] >= zoom_start]
    ax2.plot(recent['ds'], recent['y'], label='Recent Historical (6mo)', marker='o', markersize=3, linewidth=1, color='#2E86AB')
    ax2.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast (12mo)', linewidth=1.5, color='#E63946')
    if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
        ax2.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'], alpha=0.2, color='#E63946')
    ax2.axvline(historical_df['ds'].max(), linestyle='--', color='gray')
    ax2.set_title('Zoom: Last 6 months Historical + Forecast')
    ax2.set_ylabel('TOTAL_BILLING_QTY_BASE_UNIT')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(daily_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # MONTHLY aggregation
    hist_month = monthly_aggregate(historical_df, method='sum')
    fc_month = forecast_df.copy()
    fc_month['month'] = fc_month['ds'].dt.to_period('M').dt.to_timestamp()
    fc_month_agg = fc_month.groupby('month')['yhat'].sum().reset_index().rename(columns={'month': 'ds', 'yhat': 'y'})

    fig2, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.plot(hist_month['ds'], hist_month['y'], label='Historical (Monthly)', marker='o', color='#2E86AB')
    ax.plot(fc_month_agg['ds'], fc_month_agg['y'], label='Forecast (Monthly)', marker='o', color='#E63946')
    ax.set_title(f"{material_group} - TOTAL_BILLING_QTY_BASE_UNIT - Monthly Aggregation")
    ax.set_ylabel('TOTAL_BILLING_QTY_BASE_UNIT (sum)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(monthly_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)

    return daily_path, monthly_path


#######################
# Material group processing
#######################
def process_material_group(df: pd.DataFrame, material_group: str, out_base: str = "material_group_forecasts",
                          forecast_months: int = 12) -> Dict[str, Any]:
    """
    Process forecast for a single material group
    """
    print(f"\n{'='*60}")
    print(f"Processing Material Group: {material_group}")
    print(f"{'='*60}")
    
    # Filter data for this material group
    filtered = df[df['MATERIAL_GROUP_T'] == material_group].copy()
    
    if filtered.empty:
        print(f"WARNING: No data found for material group {material_group}")
        return {'material_group': material_group, 'status': 'NO_DATA'}
    
    print(f"Rows for {material_group}: {len(filtered):,}")
    
    # Create output directory
    out_dir = make_output_dir(out_base, material_group)
    print(f"Output directory: {out_dir}")
    
    # Prepare daily aggregated historical
    hist_daily = aggregate_daily_for_target(filtered, 'TOTAL_BILLING_QTY_BASE_UNIT')
    hist_daily['y'] = hist_daily['y'].clip(lower=0)
    
    print(f"Historical data points: {len(hist_daily)}")
    print(f"Date range: {hist_daily['ds'].min()} to {hist_daily['ds'].max()}")
    print(f"Mean daily quantity: {hist_daily['y'].mean():.2f}")
    print(f"Total quantity: {hist_daily['y'].sum():.2f}")
    
    # Build model
    sparse_mode = (len(hist_daily) < 60)
    model = build_prophet_model(hist_daily, sparse_mode=sparse_mode)
    
    # Forecast
    forecast_days = forecast_months * 30
    raw_fc = generate_future_and_predict(model, hist_daily, days=forecast_days)
    
    # Post-process with material group context
    processed = post_process_pipeline(hist_daily, raw_fc, material_group=material_group)
    
    # Keep only forecast period
    hist_last = hist_daily['ds'].max()
    final_forecast = processed[processed['ds'] > hist_last].copy()
    
    # Save daily CSV
    out_csv = final_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    out_csv = out_csv.rename(columns={'yhat': 'FORECAST', 'yhat_lower': 'FORECAST_LOWER', 'yhat_upper': 'FORECAST_UPPER'})
    out_csv['MATERIAL_GROUP'] = material_group
    
    csv_name = os.path.join(out_dir, f"{material_group}_daily_forecast.csv")
    out_csv.to_csv(csv_name, index=False)
    print(f"Saved daily CSV: {csv_name}")
    
    # Save monthly CSV
    monthly_fc = final_forecast.copy()
    monthly_fc['MONTH'] = monthly_fc['ds'].dt.to_period('M').astype(str)
    monthly_agg = monthly_fc.groupby('MONTH').agg({
        'yhat': 'sum',
        'yhat_lower': 'sum',
        'yhat_upper': 'sum'
    }).reset_index()
    monthly_agg = monthly_agg.rename(columns={
        'yhat': 'FORECAST',
        'yhat_lower': 'FORECAST_LOWER',
        'yhat_upper': 'FORECAST_UPPER'
    })
    monthly_agg['MATERIAL_GROUP'] = material_group
    
    monthly_csv_name = os.path.join(out_dir, f"{material_group}_monthly_forecast.csv")
    monthly_agg.to_csv(monthly_csv_name, index=False)
    print(f"Saved monthly CSV: {monthly_csv_name}")
    
    # Save combined monthly CSV (historical + forecast)
    # Create historical monthly aggregation
    hist_monthly = hist_daily.copy()
    hist_monthly['MONTH'] = hist_monthly['ds'].dt.to_period('M').astype(str)
    hist_monthly_agg = hist_monthly.groupby('MONTH')['y'].sum().reset_index()
    hist_monthly_agg = hist_monthly_agg.rename(columns={'y': 'ACTUAL'})
    hist_monthly_agg['TYPE'] = 'HISTORICAL'
    hist_monthly_agg['MATERIAL_GROUP'] = material_group
    
    # Prepare forecast monthly data with TYPE column
    forecast_monthly_combined = monthly_agg.copy()
    forecast_monthly_combined['ACTUAL'] = None
    forecast_monthly_combined['TYPE'] = 'FORECAST'
    forecast_monthly_combined = forecast_monthly_combined.rename(columns={'FORECAST': 'FORECAST_VALUE'})
    
    # Combine historical and forecast
    combined_monthly = pd.concat([
        hist_monthly_agg[['MONTH', 'ACTUAL', 'TYPE', 'MATERIAL_GROUP']].assign(FORECAST_VALUE=None, FORECAST_LOWER=None, FORECAST_UPPER=None),
        forecast_monthly_combined[['MONTH', 'ACTUAL', 'FORECAST_VALUE', 'FORECAST_LOWER', 'FORECAST_UPPER', 'TYPE', 'MATERIAL_GROUP']]
    ], ignore_index=True)
    
    # Sort by month
    combined_monthly = combined_monthly.sort_values('MONTH').reset_index(drop=True)
    
    combined_csv_name = os.path.join(out_dir, f"{material_group}_monthly_combined.csv")
    combined_monthly.to_csv(combined_csv_name, index=False)
    print(f"Saved combined monthly CSV: {combined_csv_name}")
    
    # Create plots
    daily_path, monthly_path = plot_daily_and_monthly(hist_daily, final_forecast, out_dir, material_group)
    print(f"Saved daily plot: {daily_path}")
    print(f"Saved monthly plot: {monthly_path}")
    
    # Save model
    model_path = os.path.join(out_dir, f"{material_group}_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved model: {model_path}")
    
    return {
        'material_group': material_group,
        'status': 'SUCCESS',
        'out_dir': out_dir,
        'csv': csv_name,
        'daily_plot': daily_path,
        'monthly_plot': monthly_path,
        'model_path': model_path,
        'historical_records': len(filtered),
        'historical_days': len(hist_daily),
        'forecast_mean': final_forecast['yhat'].mean(),
        'historical_mean': hist_daily['y'].mean()
    }


#######################
# Main execution
#######################
def run_material_group_analysis(data_path: str, out_base: str = "material_group_forecasts"):
    """
    Run forecasting analysis for all specified material groups
    """
    df = load_data(data_path)
    
    # Filter to only specified material groups
    df = df[df['MATERIAL_GROUP_T'].isin(MATERIAL_GROUPS)]
    print(f"\nFiltered to material groups: {MATERIAL_GROUPS}")
    print(f"Total rows after filtering: {len(df):,}")
    
    # Show distribution
    print("\nMaterial group distribution:")
    print(df['MATERIAL_GROUP_T'].value_counts().sort_index())
    
    results = []
    for material_group in MATERIAL_GROUPS:
        try:
            result = process_material_group(df, material_group, out_base=out_base)
            results.append(result)
        except Exception as e:
            print(f"ERROR processing {material_group}: {e}")
            results.append({
                'material_group': material_group,
                'status': 'ERROR',
                'error': str(e)
            })
    
    # Save summary
    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(out_base, f"material_group_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    os.makedirs(out_base, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print(f"Saved summary: {summary_path}")
    print(f"{'='*60}")
    
    # Print final summary
    print("\nFinal Summary:")
    for result in results:
        status = result['status']
        mg = result['material_group']
        if status == 'SUCCESS':
            print(f"✓ {mg}: {result['historical_records']:,} records, {result['historical_days']} days")
        else:
            print(f"✗ {mg}: {status}")
    
    return results


if __name__ == "__main__":
    DATA_PATH = "training_data_v2.csv"
    OUT_BASE = "material_group_forecasts"
    
    print("="*60)
    print("Material Group Forecasting Analysis")
    print("Target Groups:", ", ".join(MATERIAL_GROUPS))
    print("="*60)
    
    results = run_material_group_analysis(DATA_PATH, out_base=OUT_BASE)
