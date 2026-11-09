import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import warnings

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import (
    InterpolationWarning,
    ConvergenceWarning,
    EstimationWarning,
)

# Suppress warnings for cleaner console output
warnings.simplefilter(action="ignore", category=InterpolationWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", EstimationWarning)

# ==========================================
# 1. Load and Inspect Data
# ==========================================
filename = "airline-passengers.csv"
df = pd.read_csv(filename, parse_dates=["Month"], index_col="Month")
print(df.head())
print(df.describe())

s = df["Passengers"].asfreq("MS")
print(f"\nFrequency: {s.index.freq} | rows: {len(s)} | missing: {s.isna().sum()}")

# ==========================================
# 2. Exploratory Data Analysis
# ==========================================
# Initial rolling means
fig, ax = plt.subplots()
s.plot(ax=ax, lw=0.8, alpha=0.7, label="Monthly")
s.rolling(3, center=True).mean().plot(ax=ax, lw=1.5, label="3-month rolling mean")
s.rolling(12, center=True).mean().plot(ax=ax, lw=2, label="12-month rolling mean")
ax.legend()
plt.title("Monthly Passengers with Rolling Means")
plt.show()

# Year-over-year comparison
plt.figure(figsize=(12, 5))
for year, group in s.groupby(s.index.year):
    plt.plot(group.index.month, group.values, marker="o", label=str(year))
plt.title("Each Year's Monthly Passenger Trend on the Same Calendar Axis")
plt.xlabel("Month")
plt.ylabel("Passenger Count")
plt.xticks(
    range(1, 13),
    [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ],
)
plt.legend(ncols=5, fontsize=8, loc="upper left")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


# ==========================================
# 3. Stationarity Testing & Transformations
# ==========================================
def stationarity_report(x, name):
    x = x.dropna()
    adf_p = adfuller(x, autolag="AIC")[1]
    kpss_p = kpss(x, regression="c", nlags="auto")[1]
    verdict = "stationary" if (adf_p < 0.05 and kpss_p > 0.05) else "NOT stationary"
    return {
        "series": name,
        "n": len(x),
        "ADF p": round(adf_p, 4),
        "KPSS p": round(kpss_p, 4),
        "verdict": verdict,
    }


PERIOD = 12  # yearly seasonality in monthly data
WINDOW = 12

results = {}
results["original"] = s

log_s = np.log(s)
results["log"] = log_s
results["log - rolling mean"] = log_s - log_s.rolling(WINDOW).mean()
results["log - EWM mean"] = log_s - log_s.ewm(span=WINDOW, adjust=False).mean()
results["log diff(1)"] = log_s.diff()
results["log seasonal diff(12)"] = log_s.diff(PERIOD)

stl = STL(log_s, period=PERIOD, robust=True).fit()
results["log STL residual"] = stl.resid
results["log diff(1) + seasonal diff(12)"] = log_s.diff().diff(PERIOD)
results["log - rolling mean, then diff(1)"] = (
    log_s - log_s.rolling(WINDOW).mean()
).diff()
results["log STL deseasonalised, diff(1)"] = (stl.trend + stl.resid).diff()

report = pd.DataFrame([stationarity_report(v, k) for k, v in results.items()])
print("\n--- Stationarity Report ---")
print(report)

# Plot all transformations
fig, axes = plt.subplots(len(results), 1, figsize=(10, 2.2 * len(results)), sharex=True)
for ax, (name, ser) in zip(axes, results.items()):
    ser.plot(ax=ax, lw=0.8)
    ax.set_title(name, fontsize=9, loc="left")
plt.tight_layout()
plt.show()

# ==========================================
# 4. ACF / PACF Analysis
# ==========================================
best_diff = results["log diff(1) + seasonal diff(12)"].dropna()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(best_diff, lags=36, ax=ax[0])
plot_pacf(best_diff, lags=36, ax=ax[1])
plt.suptitle("ACF and PACF of Log + 1st Diff + Seasonal Diff(12)")
plt.show()

# ==========================================
# 5. STL Decomposition
# ==========================================
stl_log = STL(np.log(s), period=12, robust=True).fit()
fig = stl_log.plot()
fig.set_size_inches(12, 8)
plt.show()

var_total_log = np.log(s).var()
print("\n--- Variance Shares (Log-Transformed) ---")
print(f"Seasonal share of variance: {stl_log.seasonal.var() / var_total_log:.2%}")
print(f"Trend share of variance: {stl_log.trend.var() / var_total_log:.2%}")
print(f"Residual share of variance: {stl_log.resid.var() / var_total_log:.2%}")

# ==========================================
# 6. Train/Test Split & Grid Search
# ==========================================
m = s.resample("MS").mean()
m_train, m_test = m[:"1959-12-01"], m["1960-01-01":]
log_train = np.log(m_train)

print("\n--- Grid Search for SARIMA Parameters ---")
best_model = None
for p, q, P, Q in itertools.product([0, 1, 2], [0, 1], [0, 1], [0, 1]):
    try:
        res = SARIMAX(log_train, order=(p, 1, q), seasonal_order=(P, 1, Q, 12)).fit(
            disp=False, maxiter=200
        )
        if best_model is None or res.aic < best_model[0]:
            best_model = (res.aic, (p, 1, q), (P, 1, Q, 12), res)
    except Exception:
        pass  # Suppress individual fit errors for cleaner output

aic, order, seasonal_order, sarima = best_model
print(f"Best model: SARIMAX{order}x{seasonal_order} with AIC: {aic:.2f}")


# ==========================================
# 7. Model Evaluation
# ==========================================
def m_eval(pred, name):
    e = m_test - pred
    return {"model": name, "MAE": e.abs().mean(), "RMSE": np.sqrt((e**2).mean())}


steps = len(m_test)
forecast_log = sarima.get_forecast(steps=steps)
pred = np.exp(forecast_log.predicted_mean)
pred.index = m_test.index

result_dict = m_eval(pred, f"Optimal SARIMAX{order}x{seasonal_order}")
m_results = pd.DataFrame([result_dict])
print("\n--- Test Set Evaluation ---")
print(m_results)

# ==========================================
# 8. Final Forecast & Visualization
# ==========================================
print("\n--- Model Summary ---")
print(sarima.summary())

fig, ax = plt.subplots(figsize=(14, 5))
m_train.plot(ax=ax, label="Train")
m_test.plot(ax=ax, label="Test")
pred.plot(ax=ax, label="Forecast", color="red")

cir = forecast_log.conf_int()
cir.index = m_test.index
cir = np.exp(cir)
ax.fill_between(
    cir.index, cir.iloc[:, 0], cir.iloc[:, 1], color="red", alpha=0.2, label="95% CI"
)

ax.set(
    title=f"SARIMAX{order}x{seasonal_order} - AIC: {aic:.2f}",
    xlabel="Date",
    ylabel="Passengers",
)
ax.legend()
plt.show()
