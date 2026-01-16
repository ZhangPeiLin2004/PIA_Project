# ======================================================================
# PFAS NEWS–TENDER DAILY TIME-SERIES REGRESSION PIPELINE (FIXED)
# ======================================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import io
import sys

# ----------------------------------------------------------------------
# Console encoding (Windows-safe)
# ----------------------------------------------------------------------
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ======================================================================
# 1. LOAD DATA (FIXED TIMEZONE HANDLING)
# ======================================================================

print("Step 1: Loading data...")

NEWS_PATH = "clean_pfas_news_vader_analysis.csv"
TENDERS_PATH = "tenders_with_text.csv"
PFAS_TFIDF_PATH = "tenders_with_pfas_tfidf.csv"

news_df = pd.read_csv(NEWS_PATH)
tenders_df = pd.read_csv(TENDERS_PATH)
pfas_df = pd.read_csv(PFAS_TFIDF_PATH)

# FORCE UTC → guarantees datetime64[ns, UTC]
news_df["date"] = pd.to_datetime(
    news_df["date"], errors="coerce", utc=True
)

tenders_df["publication_date"] = pd.to_datetime(
    tenders_df["publication_date"], errors="coerce", utc=True
)

# Drop invalid rows
news_df = news_df.dropna(subset=["compound", "date"])
tenders_df = tenders_df.dropna(subset=["all_text", "publication_date"])

# OPTIONAL: remove timezone info after harmonization
news_df["date"] = news_df["date"].dt.tz_convert(None)
tenders_df["publication_date"] = tenders_df["publication_date"].dt.tz_convert(None)

# ======================================================================
# 2. PFAS INTENSITY (FROM PRECOMPUTED TF-IDF)
# ======================================================================

print("Step 2: Loading precomputed PFAS TF-IDF...")

# Ensure matching key types
tenders_df["tender_id"] = tenders_df["tender_id"].astype(int)
pfas_df["tender_id"] = pfas_df["tender_id"].astype(int)

# Merge TF-IDF PFAS scores
tenders_df = tenders_df.merge(
    pfas_df,
    on="tender_id",
    how="left"
)

# Replace missing values (no PFAS mention)
tenders_df["tfidf_pfas"] = tenders_df["tfidf_pfas"].fillna(0)

# Normalize PFAS intensity
max_val = tenders_df["tfidf_pfas"].max()
tenders_df["pfas_intensity_norm"] = (
    tenders_df["tfidf_pfas"] / max_val
    if max_val > 0 else 0
)

# ======================================================================
# 3. PROCEDURE TRANSPARENCY PROXY
# ======================================================================

procedure_map = {
    "openbaar": 1.0, "openbare": 1.0, "open": 1.0,
    "niet-openbaar": 0.5, "restricted": 0.5,
    "onderhands": 0.0, "negotiated": 0.0
}

tenders_df["procedure_type_clean"] = (
    tenders_df["procedure_type"].astype(str).str.lower().str.strip()
)

tenders_df["procedure_proxy"] = tenders_df["procedure_type_clean"].apply(
    lambda x: next((v for k, v in procedure_map.items() if k in x), 0.5)
)

# ======================================================================
# 4. TENDER SCOPE PROXY
# ======================================================================

def classify_scope(text):
    text = str(text).lower()
    if any(k in text for k in ["sanering", "reinig", "verwijder", "herstel", "remediat"]):
        return "cleanup"
    if any(k in text for k in ["onderzoek", "studie", "monitor", "analyse", "meting"]):
        return "research"
    if any(k in text for k in ["juridisch", "advies", "beleid", "regelgeving"]):
        return "legal"
    return "unknown"

scope_map = {"research": 0.0, "legal": 0.5, "cleanup": 1.0, "unknown": 0.5}

tenders_df["scope_proxy"] = tenders_df["all_text"].apply(classify_scope).map(scope_map)

# ======================================================================
# 5. DAILY TEMPORAL ALIGNMENT (NOW SAFE)
# ======================================================================

print("Step 3: Daily aggregation...")

news_df["day"] = news_df["date"].dt.floor("D")
tenders_df["day"] = tenders_df["publication_date"].dt.floor("D")

news_daily = news_df.groupby("day").agg(
    mean_article_sentiment=("compound", "mean"),
    article_count=("compound", "count")
).reset_index()

tenders_daily = tenders_df.groupby("day").agg(
    pfas_volume=("pfas_intensity_norm", "sum"),
    mean_scope=("scope_proxy", "mean"),
    mean_transparency=("procedure_proxy", "mean"),
    tender_count=("scope_proxy", "count")
).reset_index()

date_index = pd.DataFrame({
    "day": pd.date_range(
        start=min(news_daily["day"].min(), tenders_daily["day"].min()),
        end=max(news_daily["day"].max(), tenders_daily["day"].max()),
        freq="D"
    )
})

df_ts = (
    date_index
    .merge(news_daily, on="day", how="left")
    .merge(tenders_daily, on="day", how="left")
)

df_ts[["pfas_volume", "tender_count"]] = df_ts[["pfas_volume", "tender_count"]].fillna(0)
df_ts[["mean_scope", "mean_transparency"]] = df_ts[
    ["mean_scope", "mean_transparency"]
].fillna(0.5)
df_ts["article_count"] = df_ts["article_count"].fillna(0)

# ======================================================================
# 6. TRANSFORMS & DISTRIBUTED LAG (FIXED)
# ======================================================================

df_ts["pfas_log"] = np.log1p(df_ts["pfas_volume"])

df_ts["pfas_dli"] = (
    df_ts["pfas_log"].shift(1) +
    df_ts["pfas_log"].shift(2) +
    df_ts["pfas_log"].shift(3)
) / 3

df_ts["article_count_log"] = np.log1p(df_ts["article_count"])

df_reg = df_ts.dropna(
    subset=[
        "mean_article_sentiment",
        "pfas_dli",
        "mean_scope",
        "mean_transparency",
        "article_count_log"
    ]
)

# ======================================================================
# 7. REGRESSION (STANDARDIZED, HAC(1))
# ======================================================================

from sklearn.preprocessing import StandardScaler

Y = df_reg["mean_article_sentiment"]

X = df_reg[
    [
        "pfas_dli",
        "mean_scope",
        "mean_transparency",
        "article_count_log"
    ]
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns,
    index=X.index
)

X_scaled = sm.add_constant(X_scaled)

model = sm.OLS(Y, X_scaled).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 1}
)

print("=" * 60)
print("DAILY TIME-SERIES REGRESSION (DLI, HAC(1))")
print("=" * 60)
print(model.summary())

# ======================================================================
# 7b. SIGNIFICANCE TABLE (ALPHA LEVELS)
# ======================================================================

results_table = pd.DataFrame({
    "coef": model.params,
    "std_err": model.bse,
    "z_value": model.tvalues,
    "p_value": model.pvalues
})

def significance_stars(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    else:
        return ""

results_table["significance"] = results_table["p_value"].apply(significance_stars)

print("\nCoefficient significance (α-levels):")
print(results_table)

with open("regression_results_daily.txt", "w", encoding="utf-8") as f:
    f.write(model.summary().as_text())
    f.write("\n\nSignificance levels: * p<0.10, ** p<0.05, *** p<0.01\n\n")
    f.write(results_table.to_string())

# ======================================================================
# 8. DAILY TIME-SERIES PLOT
# ======================================================================

plt.figure(figsize=(14, 6))
plt.plot(df_ts["day"], df_ts["mean_article_sentiment"], label="Daily Sentiment")
plt.plot(df_ts["day"], df_ts["pfas_log"], label="Log PFAS Tender Volume", alpha=0.7)
plt.title("Daily PFAS Tender Activity and Media Sentiment")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("daily_time_series.png", dpi=300)
plt.show()

# ======================================================================
# 9. OBSERVED VS FITTED PLOT
# ======================================================================

df_reg["fitted_sentiment"] = model.fittedvalues

plt.figure(figsize=(10, 6))
plt.scatter(
    df_reg["fitted_sentiment"],
    df_reg["mean_article_sentiment"],
    alpha=0.7,
    label="Observed"
)

min_val = min(df_reg["fitted_sentiment"].min(), df_reg["mean_article_sentiment"].min())
max_val = max(df_reg["fitted_sentiment"].max(), df_reg["mean_article_sentiment"].max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    label="Perfect fit"
)

plt.xlabel("Fitted Sentiment")
plt.ylabel("Observed Sentiment")
plt.title("Observed vs Fitted Media Sentiment (DLI Model)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("observed_vs_fitted_dli.png", dpi=300)
plt.show()
