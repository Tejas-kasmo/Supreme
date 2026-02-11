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


def make_output_dir(base_dir: str, filter_slug: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"{filter_slug}_{timestamp}"
    out_dir = os.path.join(base_dir, folder)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def slugify_filters(filters: Dict[str, List[Any]]) -> str:
    if not filters:
        return "all"
    parts = []
    for k, v in filters.items():
        vals = "-".join(map(str, v))
        parts.append(f"{k[:8]}_{vals[:30]}")
    slug = "__".join(parts)
    return slug.replace(" ", "_")[:80]


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


def monthly_aggregate(df: pd.DataFrame, target_col: str, method: str = 'sum') -> pd.DataFrame:
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

    # per-year counts and anomalies counts will be computed elsewhere
    years = sorted(df['year'].unique())
    per_year_counts = {}
    for y in years:
        per_year_counts[y] = len(df[df['year'] == y])
    presence['per_year_counts'] = per_year_counts

    return presence


def detect_anomalies_iqr(hist_df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Mark anomalies using IQR method. Returns a copy of hist_df with 'is_anomaly' boolean.
    Also compute anomaly magnitude relative to median (ratio).
    """
    df = hist_df.copy()
    q1 = df['y'].quantile(0.25)
    q3 = df['y'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    # fallback for very low variance
    if iqr == 0:
        # use z-score
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


def summarize_anomalies_by_window(an_df: pd.DataFrame, window: str = 'year') -> Dict:
    """
    Count anomalies per year/month/quarter as needed.
    window options: 'year', 'month', 'quarter'
    """
    df = an_df.copy()
    if window == 'year':
        df['period'] = df['ds'].dt.year
    elif window == 'month':
        df['period'] = df['ds'].dt.to_period('M').dt.to_timestamp()
    elif window == 'quarter':
        df['period'] = df['ds'].dt.to_period('Q').dt.to_timestamp()
    else:
        raise ValueError("window must be one of year/month/quarter")

    counts = df.groupby('period')['is_anomaly'].sum().to_dict()
    return counts


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
# Post-processing:
# - scale forecast to match historical mean/std
# - apply day-of-week/day-of-month density pattern
# - inject anomalies into forecast matching historical counts & magnitudes
# - ENFORCE final cap: forecast values <= 1.05 * historical_max
#######################
def scale_forecast_to_hist(hist_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    f = forecast_df.copy()
    hist_mean = hist_df['y'].mean()
    hist_std = hist_df['y'].std()
    raw_mean = f['yhat'].mean()
    raw_std = f['yhat'].std()

    if raw_std > 0:
        z = (f['yhat'] - raw_mean) / raw_std
        f['yhat'] = z * (hist_std if hist_std > 0 else 1.0) + (hist_mean if hist_mean is not None else 0.0)

        # adjust bounds too (if present)
        if 'yhat_lower' in f.columns and 'yhat_upper' in f.columns:
            zl = (f['yhat_lower'] - raw_mean) / raw_std
            zu = (f['yhat_upper'] - raw_mean) / raw_std
            f['yhat_lower'] = zl * (hist_std if hist_std > 0 else 1.0) + (hist_mean if hist_mean is not None else 0.0)
            f['yhat_upper'] = zu * (hist_std if hist_std > 0 else 1.0) + (hist_mean if hist_mean is not None else 0.0)
    else:
        # constant forecast: set to historical mean
        f['yhat'] = hist_mean
        if 'yhat_lower' in f.columns:
            f['yhat_lower'] = hist_df['y'].quantile(0.05)
            f['yhat_upper'] = hist_df['y'].quantile(0.95)

    return f


def apply_density_pattern(hist_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce (or zero) forecast predictions on days that historically had low presence.
    This keeps the forecasted 'density' similar to historical.
    Strategy:
      - compute presence fraction per DOW (and DOM as fallback)
      - for forecast day: multiply yhat by presence_frac[dow] / max_presence
      - if presence_frac small (<0.05) we zero-out (rare days)
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
        # combine heuristically
        pf_comb = max(pf, pf_dom)
        # scale factor
        scale = (pf_comb / max_dow) if max_dow > 0 else 1.0
        # if pf_comb extremely low -> drop to near-zero
        if pf_comb < 0.05:
            return 0.0
        return row['yhat'] * scale

    f['yhat'] = f.apply(adjust_row, axis=1)
    # adjust yhat bounds if present
    if 'yhat_lower' in f.columns:
        f['yhat_lower'] = f['yhat_lower'] * 0.8
        f['yhat_upper'] = f['yhat_upper'] * 1.2
    return f


def inject_anomalies(hist_df: pd.DataFrame, forecast_df: pd.DataFrame, cap_factor: float = 0.90) -> pd.DataFrame:
    """
    Identify number/magnitude of anomalies historically (per year) and inject similar number
    into the forecast period by selecting top candidate forecast days and scaling by sampled anomaly magnitude.
    NOTE: Any injected anomaly will be capped at cap_factor * historical_max to prevent excessive spikes.
    """
    hist_an = detect_anomalies_iqr(hist_df)
    # anomaly counts historically per year
    hist_an_counts = hist_an[hist_an['is_anomaly']].groupby(hist_an['ds'].dt.year).size().to_dict()
    # get distribution of anomaly magnitude ratios
    mags = hist_an.loc[hist_an['is_anomaly'], 'anomaly_mag'].dropna()
    if len(mags) == 0:
        mags = pd.Series([2.0])  # fallback: anomalies are 2x typical

    # total anomalies per historical year (median)
    if hist_an_counts:
        median_count = int(np.median(list(hist_an_counts.values())))
    else:
        median_count = 1

    f = forecast_df.copy()
    # only inject anomalies into forecast future timeframe (ds > hist max)
    hist_last = hist_df['ds'].max()
    future_mask = f['ds'] > hist_last
    future = f[future_mask].copy()

    # choose candidate days: days where predicted value is relatively high vs local median
    if len(future) == 0:
        return f
    future_med = future['yhat'].median() if future['yhat'].median() != 0 else 1.0
    future['score'] = (future['yhat'] - future_med).abs()

    # pick top median_count days to be anomalies
    n_anom = max(1, median_count)
    top_idx = future.sort_values('score', ascending=False).head(n_anom).index.tolist()

    # sample magnitudes from historical anomaly magnitude distribution
    sampled_mags = mags.sample(n=len(top_idx), replace=True).values

    # HISTORICAL CAP (strict): 1.05 * historical_max
    hist_max = hist_df['y'].max() if hist_df['y'].max() > 0 else np.max([1.0, future['yhat'].max()])
    hard_cap = hist_max * cap_factor

    for i, idx in enumerate(top_idx):
        factor = float(sampled_mags[i])
        # apply anomaly by multiplying predicted value by factor (but limit explosion)
        old = f.loc[idx, 'yhat']
        new = max(old * factor, old + 1.0)
        # cap injection to strict historical cap (1.05x historical max)
        new = min(new, hard_cap)
        f.loc[idx, 'yhat'] = new
        if 'yhat_upper' in f.columns:
            f.loc[idx, 'yhat_upper'] = min(max(f.loc[idx, 'yhat_upper'], new * 0.90), hard_cap)
        if 'yhat_lower' in f.columns:
            f.loc[idx, 'yhat_lower'] = min(f.loc[idx, 'yhat_lower'], new)

    return f


def enforce_historical_cap(hist_df: pd.DataFrame, forecast_df: pd.DataFrame, cap_factor: float = 0.90) -> pd.DataFrame:
    """
    Final enforcement: Ensure no forecast yhat or yhat_upper exceeds cap_factor * historical_max.
    Also ensure yhat_lower not greater than cap.
    """
    f = forecast_df.copy()
    hist_max = hist_df['y'].max() if hist_df['y'].max() > 0 else 1.0
    hard_cap = hist_max * cap_factor

    f['yhat'] = f['yhat'].clip(upper=hard_cap)
    if 'yhat_upper' in f.columns:
        f['yhat_upper'] = f['yhat_upper'].clip(upper=hard_cap)
    if 'yhat_lower' in f.columns:
        f['yhat_lower'] = f['yhat_lower'].clip(upper=hard_cap)

    # Also ensure non-negative lower bound
    if 'yhat_lower' in f.columns:
        f['yhat_lower'] = f['yhat_lower'].clip(lower=0)
    f['yhat'] = f['yhat'].clip(lower=0)

    return f


def post_process_pipeline(hist_df: pd.DataFrame, raw_forecast: pd.DataFrame) -> pd.DataFrame:
    # Step 1: scale to historical distribution
    scaled = scale_forecast_to_hist(hist_df, raw_forecast)
    # Step 2: apply density pattern
    density_applied = apply_density_pattern(hist_df, scaled)
    # Step 3: inject anomalies (counts/magnitudes similar to history) with strict cap
    with_anom = inject_anomalies(hist_df, density_applied, cap_factor=0.90)
    # Clip negatives
    with_anom['yhat'] = with_anom['yhat'].clip(lower=0)
    if 'yhat_lower' in with_anom.columns:
        with_anom['yhat_lower'] = with_anom['yhat_lower'].clip(lower=0)
    # FINAL: enforce strict historical cap across the board
    final = enforce_historical_cap(hist_df, with_anom, cap_factor=0.90)
    return final


#######################
# Accuracy metrics & plotting
#######################
def calculate_accuracy_metrics(historical_df: pd.DataFrame, forecast_df: pd.DataFrame) -> Dict[str, float]:
    hist_mean = historical_df['y'].mean()
    hist_std = historical_df['y'].std()
    hist_max = historical_df['y'].max()
    hist_min = historical_df['y'].min()

    forecast_mean = forecast_df['yhat'].mean()
    forecast_std = forecast_df['yhat'].std()

    mean_ratio = (forecast_mean / hist_mean) if hist_mean != 0 else 1.0
    std_ratio = (forecast_std / hist_std) if hist_std != 0 else 1.0

    mean_alignment = max(0, 100 - abs(1 - mean_ratio) * 100)
    std_alignment = max(0, 100 - abs(1 - std_ratio) * 100)

    in_range = ((forecast_df['yhat'] >= hist_min * 0.5) & (forecast_df['yhat'] <= hist_max * 1.5)).sum()
    range_score = (in_range / len(forecast_df)) * 100

    overall = (mean_alignment * 0.5 + std_alignment * 0.3 + range_score * 0.2)
    return {
        'mean_alignment': float(min(100, mean_alignment)),
        'std_alignment': float(min(100, std_alignment)),
        'range_score': float(range_score),
        'overall_score': float(overall),
        'mean_ratio': float(mean_ratio),
        'std_ratio': float(std_ratio)
    }


def plot_daily_and_monthly(historical_df: pd.DataFrame, forecast_df: pd.DataFrame,
                           out_dir: str, target_name: str, filters: Dict[str, List[Any]],
                           accuracy: Dict[str, float], plot_prefix: str, agg_method: str = 'sum') -> Tuple[str, str]:
    """
    Create two plots for a single target:
      - daily overlay historical + forecast (full + zoom)
      - monthly aggregated historical + forecast
    Returns (daily_plot_path, monthly_plot_path)
    """
    # Filenames
    daily_path = os.path.join(out_dir, f"{plot_prefix}_{target_name}_daily.png")
    monthly_path = os.path.join(out_dir, f"{plot_prefix}_{target_name}_monthly.png")

    # DAILY plot
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    hist_color = '#2E86AB'
    forecast_color = '#E63946'
    filter_str = ", ".join([f"{k}={v}" for k, v in filters.items()]) if filters else "No filters"

    # Top: full historical + forecast
    ax1 = axes[0]
    ax1.plot(historical_df['ds'], historical_df['y'], label='Historical', linewidth=0.9)
    ax1.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast (12mo)', linewidth=1.5)
    if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
        ax1.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'], alpha=0.2)

    ax1.axvline(historical_df['ds'].max(), linestyle='--', color='gray', label='Forecast Start')
    ax1.set_title(f"{target_name} - Historical + Forecast (Daily)\nFilters: {filter_str}")
    ax1.set_ylabel(target_name)
    ax1.legend(loc='upper left')

    # accuracy text
    acc_txt = f"Overall: {accuracy['overall_score']:.1f}% | MeanAlign: {accuracy['mean_alignment']:.1f}% | StdAlign: {accuracy['std_alignment']:.1f}%"
    ax1.text(0.02, 0.95, acc_txt, transform=ax1.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Bottom: zoom last 180 days + forecast
    ax2 = axes[1]
    zoom_start = historical_df['ds'].max() - pd.Timedelta(days=180)
    recent = historical_df[historical_df['ds'] >= zoom_start]
    ax2.plot(recent['ds'], recent['y'], label='Recent Historical (6mo)', marker='o', markersize=3, linewidth=1)
    ax2.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast (12mo)', linewidth=1.5)
    if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
        ax2.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'], alpha=0.2)
    ax2.axvline(historical_df['ds'].max(), linestyle='--', color='gray')
    ax2.set_title('Zoom: Last 6 months Historical + Forecast')
    ax2.set_ylabel(target_name)
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(daily_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # MONTHLY aggregation
    hist_month = monthly_aggregate(historical_df, target_name, method=agg_method)
    # For forecast monthly aggregation: use forecast dates beyond historical max (and possibly include last historical month partial)
    fc_used = forecast_df.copy()
    # Include historical tail to align months where needed
    fc_month = fc_used.copy()

    fc_month['month'] = fc_month['ds'].dt.to_period('M').dt.to_timestamp()
    print("test", fc_month)
    if agg_method == 'sum':
        fc_month_agg = fc_month.groupby('month')['yhat'].sum().reset_index().rename(columns={'month': 'ds', 'yhat': 'y'})
    else:
        fc_month_agg = fc_month.groupby('month')['yhat'].mean().reset_index().rename(columns={'month': 'ds', 'yhat': 'y'})
    


    # Build combined monthly DataFrame for plotting (historical months followed by forecast months)
    fig2, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.plot(hist_month['ds'], hist_month['y'], label='Historical (Monthly)', marker='o')
    ax.plot(fc_month_agg['ds'], fc_month_agg['y'], label='Forecast (Monthly)', marker='o')
    ax.set_title(f"{target_name} - Monthly Aggregation ({agg_method})\nFilters: {filter_str}")
    ax.set_ylabel(f"{target_name} ({agg_method})")
    ax.legend(loc='upper left')
    # show simple accuracy summary
    ax.text(0.98, 0.95, acc_txt, transform=ax.transAxes, fontsize=9, ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(monthly_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)

    return daily_path, monthly_path


#######################
# Single test case run
#######################
def run_test_case(df: pd.DataFrame, filters: Dict[str, List[Any]], out_base: str = "forecasts",
                  forecast_months: int = 12, save_models: bool = False) -> Dict[str, Any]:
    """
    For a filter combination:
      - create folder
      - for each target: train model, forecast, post-process, save csv & plots
    """
    # Subset with filters
    filtered = df.copy()
    if filters:
        for col, vals in filters.items():
            if col not in filtered.columns:
                print(f"Warning: Column {col} not in data, skipping filter.")
                continue
            filtered = filtered[filtered[col].isin(vals)]

    if filtered.empty:
        raise ValueError("No data after filters")

    slug = slugify_filters(filters)
    out_dir = make_output_dir(out_base, slug)
    print(f"Outputs will be saved in: {out_dir}")

    # Targets
    targets = [
        ("TOTAL_BILLING_QTY_BASE_UNIT", 'sum'),
        ("TOTAL_NET_WT", 'sum'),
        ("AVG_BILLING_QTY_BASE_UNIT", 'mean'),
        ("AVG_NET_WT", 'mean')
    ]

    summary = {'filters': filters, 'out_dir': out_dir, 'targets': {}}

    for target_name, agg_method in targets:
        if target_name not in filtered.columns:
            print(f"Target {target_name} not found in filtered data. Skipping.")
            continue

        print(f"\n=== Processing target: {target_name} ===")
        # Prepare daily aggregated historical
        hist_daily = aggregate_daily_for_target(filtered, target_name)
        if len(hist_daily) < 7:
            print("Warning: Very few data points (less than 7). Forecast will use conservative settings.")

        # Ensure no negative values
        hist_daily['y'] = hist_daily['y'].clip(lower=0)

        # Detect anomalies and patterns
        an_df = detect_anomalies_iqr(hist_daily)
        anomalies_summary = summarize_anomalies_by_window(an_df, window='year')

        # Build holidays simple - use month-end markers if present
        patterns = detect_density_patterns(hist_daily)
        # create holidays df (very simple): month-end days where historics show spikes
        holidays = None
        if any([patterns['dom_frac'].get(d, 0) > 0.5 for d in range(28, 32)]):
            # build month-end holidays for future year
            last = hist_daily['ds'].max()
            future_dates = pd.date_range(last + timedelta(days=1), periods=forecast_months * 30)
            month_end = future_dates[future_dates.day >= 28]
            holidays = pd.DataFrame({'holiday': 'month_end', 'ds': month_end, 'lower_window': 0, 'upper_window': 0})

        # model
        sparse_mode = (len(hist_daily) < 60)
        model = build_prophet_model(hist_daily, sparse_mode=sparse_mode, holidays=holidays)

        # forecast (days)
        forecast_days = forecast_months * 30
        raw_fc = generate_future_and_predict(model, hist_daily, days=forecast_days)

        # post-process
        processed = post_process_pipeline(hist_daily, raw_fc)

        # keep only forecast period (strictly after historical max)
        hist_last = hist_daily['ds'].max()
        final_forecast = processed[processed['ds'] > hist_last].copy()
        # keep columns of interest and round
        out_csv = final_forecast[['ds', 'yhat', 'yhat_lower']].copy() if 'yhat_lower' in final_forecast.columns else final_forecast[['ds', 'yhat']].copy()
        out_csv = out_csv.rename(columns={'yhat': 'FORECAST', 'yhat_lower': 'FORECAST_LOWER'})
        # add extra metadata columns
        out_csv['TARGET'] = target_name
        out_csv['FILTERS'] = str(filters)

        csv_name = os.path.join(out_dir, f"forecast_{target_name}.csv")
        out_csv.to_csv(csv_name, index=False)
        print(f"Saved CSV: {csv_name}")

        # Save model optionally
        model_path = None
        if save_models:
            model_path = os.path.join(out_dir, f"model_{target_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved model: {model_path}")

        # accuracy compute (compare hist to forecast)
        accuracy = calculate_accuracy_metrics(hist_daily, final_forecast)

        # plotting daily + monthly
        plot_prefix = datetime.now().strftime("%Y%m%d")
        daily_path, monthly_path = plot_daily_and_monthly(hist_daily, final_forecast, out_dir,
                                                          target_name, filters, accuracy, plot_prefix,
                                                          agg_method=agg_method)

        # Save summary
        summary['targets'][target_name] = {
            'csv': csv_name,
            'daily_plot': daily_path,
            'monthly_plot': monthly_path,
            'accuracy': accuracy,
            'anomalies_summary': anomalies_summary,
            'model_path': model_path
        }
        print(f"Completed target: {target_name} -> overall score {accuracy['overall_score']:.1f}%")

    return summary


#######################
# Bulk validation (some new test filters included)
#######################
def run_validation_suite(data_path: str, out_base: str = "forecasts", max_tests: Optional[int] = None):
    df = load_data(data_path)

    # Comprehensive test cases with 1, 2, 3, 4, 5 filters and multiple filter values
    test_cases = [
        # 1-FILTER TEST CASES
        {"MATERIAL_CODE": ['MP1G8ELB006E']},
        {"CUSTOMER_STATE_NAME": ['MAHARASHTRA']},
        {"BILLING_PLANT_CODE": ['3010']},
        {"MATERIAL_GROUP_T": ['M-S']},
        {"BILLING_UNIT_BASE": ['NOS']},
        
        # 2-FILTER TEST CASES
        {"MATERIAL_CODE": ['MP1SRPBE110L'], "CUSTOMER_STATE_NAME": ['BIHAR']},
        {"CUSTOMER_STATE_NAME": ['MAHARASHTRA'], "BILLING_PLANT_CODE": ['3010']},
        {"MATERIAL_GROUP_T": ['M-R'], "BILLING_UNIT_BASE": ['NOS']},
        {"MATERIAL_CODE": ['MP1G8ELB003E'], "BILLING_PLANT_CODE": ['3010']},
        
        # 3-FILTER TEST CASES
        {"MATERIAL_CODE": ['MP1A4ELB111L'], "CUSTOMER_STATE_NAME": ['KERALA'], "BILLING_PLANT_CODE": ['3010']},
        {"CUSTOMER_STATE_NAME": ['MAHARASHTRA'], "BILLING_PLANT_CODE": ['3010'], "MATERIAL_GROUP_T": ['M-R']},
        {"MATERIAL_GROUP_T": ['M-G'], "BILLING_UNIT_BASE": ['NOS'], "CUSTOMER_STATE_NAME": ['TAMIL NADU']},
        {"MATERIAL_CODE": ['MP1G8BDF003E'], "BILLING_PLANT_CODE": ['7020'], "BILLING_UNIT_BASE": ['NOS']},
        
        # 4-FILTER TEST CASES
        {"MATERIAL_CODE": ['MP1A6ELB050L'], "CUSTOMER_STATE_NAME": ['KERALA'], "BILLING_PLANT_CODE": ['3010'], "MATERIAL_GROUP_T": ['M-R']},
        {"CUSTOMER_STATE_NAME": ['MAHARASHTRA'], "BILLING_PLANT_CODE": ['3010'], "MATERIAL_GROUP_T": ['M-S'], "BILLING_UNIT_BASE": ['NOS']},
        {"MATERIAL_CODE": ['PPBJQF2M060B'], "CUSTOMER_STATE_NAME": ['ORISSA'], "BILLING_PLANT_CODE": ['2030'], "BILLING_UNIT_BASE": ['NOS']},
        {"MATERIAL_GROUP_T": ['P-R'], "BILLING_UNIT_BASE": ['NOS'], "CUSTOMER_STATE_NAME": ['UTTAR PRADESH'], "BILLING_PLANT_CODE": ['1050']},
        
        # 5-FILTER TEST CASES
        {"MATERIAL_CODE": ['MP1SRPBF110L'], "CUSTOMER_STATE_NAME": ['MAHARASHTRA'], "BILLING_PLANT_CODE": ['3010'], "MATERIAL_GROUP_T": ['M-S'], "BILLING_UNIT_BASE": ['NOS']},
        {"MATERIAL_CODE": ['MP1A6ELB075L'], "CUSTOMER_STATE_NAME": ['MAHARASHTRA'], "BILLING_PLANT_CODE": ['7020'], "MATERIAL_GROUP_T": ['M-R'], "BILLING_UNIT_BASE": ['NOS']},
        {"MATERIAL_CODE": ['PPALF2QF6M090L'], "CUSTOMER_STATE_NAME": ['MAHARASHTRA'], "BILLING_PLANT_CODE": ['3030'], "MATERIAL_GROUP_T": ['P-R'], "BILLING_UNIT_BASE": ['NOS']},
        
        # MULTIPLE VALUES IN SAME PARAMETER
        {"MATERIAL_CODE": ['MP1G8ELB006E', 'MP1SRPBE110L']},  # 2 materials
        {"MATERIAL_CODE": ['MP1A4ELB111L', 'MP1A4ELB075L', 'MP1A4ELB063L']},  # 3 materials
        {"CUSTOMER_STATE_NAME": ['MAHARASHTRA', 'KERALA']},  # 2 states
        {"CUSTOMER_STATE_NAME": ['MAHARASHTRA', 'KERALA', 'TAMIL NADU']},  # 3 states
        {"BILLING_PLANT_CODE": ['3010', '3030']},  # 2 plants
        {"BILLING_PLANT_CODE": ['3010', '3030', '7020']},  # 3 plants
        {"MATERIAL_GROUP_T": ['M-R', 'M-S']},  # 2 groups
        {"MATERIAL_GROUP_T": ['M-R', 'M-S', 'M-G']},  # 3 groups
        
        # MULTIPLE VALUES + MULTIPLE FILTERS
        {"MATERIAL_CODE": ['MP1SRPBF110L', 'MP1SRPBE110L'], "CUSTOMER_STATE_NAME": ['MAHARASHTRA', 'BIHAR']},
        {"CUSTOMER_STATE_NAME": ['MAHARASHTRA', 'KERALA', 'TAMIL NADU'], "BILLING_PLANT_CODE": ['3010', '7020']},
        {"MATERIAL_GROUP_T": ['M-R', 'M-S'], "BILLING_UNIT_BASE": ['NOS'], "CUSTOMER_STATE_NAME": ['MAHARASHTRA']},
        {"MATERIAL_CODE": ['MP1A6ELB050L', 'MP1A6ELB075L', 'MP1A6ELB110L'], "CUSTOMER_STATE_NAME": ['MAHARASHTRA', 'KERALA'], "BILLING_PLANT_CODE": ['3010']},
    ]

    if max_tests:
        test_cases = test_cases[:max_tests]

    results = []
    for i, filters in enumerate(test_cases):
        print("\n" + "=" * 60)
        print(f"VALIDATION TEST {i+1}/{len(test_cases)}: filters={filters}")
        print("=" * 60)
        try:
            res = run_test_case(df, filters, out_base=out_base, forecast_months=12, save_models=False)
            results.append({'test_num': i+1, 'filters': filters, 'result': res, 'status': 'OK'})
        except Exception as e:
            print(f"ERROR running test case {i+1}: {e}")
            results.append({'test_num': i+1, 'filters': filters, 'error': str(e), 'status': 'ERROR'})

    # Save overall validation manifest
    manifest = []
    for r in results:
        if r['status'] == 'OK':
            out_dir = r['result']['out_dir']
            manifest.append({
                'test_num': r['test_num'],
                'filters': r['filters'],
                'out_dir': out_dir
            })
        else:
            manifest.append({
                'test_num': r['test_num'],
                'filters': r['filters'],
                'error': r.get('error')
            })

    manifest_df = pd.DataFrame(manifest)
    meta_path = os.path.join(out_base, f"validation_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    os.makedirs(out_base, exist_ok=True)
    manifest_df.to_csv(meta_path, index=False)
    print(f"\nSaved validation manifest: {meta_path}")
    return results


#######################
# If run as script
#######################
if __name__ == "__main__":
    DATA_PATH = "training_data_v2.csv" 
    OUT_BASE = "forecasts"
    run_validation_suite(DATA_PATH, out_base=OUT_BASE, max_tests=None)
