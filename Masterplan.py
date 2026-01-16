import pandas as pd
import numpy as np
import subprocess
import statsmodels.api as sm
from sklearn.feature_extraction.text import TfidfVectorizer
from stargazer.stargazer import Stargazer
from tqdm import tqdm
import sys
import io

# Set encoding to UTF-8 for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# 1. LOAD CLEANED NEWS VADER ANALYSIS AND TENDER_TEXT


print("Step 1: Loading and preparing data...")

NEWS_PATH = "clean_pfas_news_vader_analysis.csv"
TENDERS_PATH = "tenders_with_text.csv"

news_df = pd.read_csv(NEWS_PATH)
tenders_df = pd.read_csv(TENDERS_PATH)

print("=== NEWS DATA ===")
print(f"Shape: {news_df.shape}")
print(f"Columns: {list(news_df.columns)}")

print("\n=== TENDERS DATA ===")
print(f"Shape: {tenders_df.shape}")
print(f"Columns: {list(tenders_df.columns)}")

# Check what columns we actually have
print("\nChecking news columns for sentiment data...")
sentiment_cols = [col for col in news_df.columns if 'sentiment' in col.lower() or 'compound' in col]
print(f"Sentiment-related columns: {sentiment_cols}")

# Update required columns based on actual data
required_news_cols = [
    "date",
    "compound",
    "source",
    "text_en"
]

required_tender_cols = [
    "all_text",
    "procedure_type",
    "publication_date"
]

missing_news = [c for c in required_news_cols if c not in news_df.columns]
missing_tenders = [c for c in required_tender_cols if c not in tenders_df.columns]

if missing_news:
    print(f"Missing news columns: {missing_news}")
    print("Available news columns:", list(news_df.columns))

if missing_tenders:
    print(f"Missing tender columns: {missing_tenders}")
    print("Available tender columns:", list(tenders_df.columns))

# Convert dates
news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce", utc=True)
tenders_df["publication_date"] = pd.to_datetime(
    tenders_df["publication_date"],
    errors="coerce",
    utc=True
)

print("\nDate ranges:")
print(f"News: {news_df['date'].min()} → {news_df['date'].max()}")
print(f"Tenders: {tenders_df['publication_date'].min()} → {tenders_df['publication_date'].max()}")

# Clean data
news_df = news_df.dropna(subset=["compound", "date"])
tenders_df = tenders_df.dropna(subset=["all_text", "publication_date"])

print("\nAfter cleaning:")
print(f"News rows: {len(news_df)}")
print(f"Tender rows: {len(tenders_df)}")


# 2. CREATE TF-IDF SCORE COLUMN


print("\n" + "="*50)
print("Step 2: Creating TF-IDF scores for PFAS intensity...")

# Use the tenders_df we already have
df_tfidf = tenders_df.copy()

# Explicit, auditable list of PFAS terms
PFAS_TERMS = [
    "pfas",
    "pfos",
    "pfoa",
    "perfluor",
    "polyfluor",
    "fluorverbinding",
]

# Dutch stop words list
DUTCH_STOP_WORDS = [
    'de', 'en', 'van', 'ik', 'te', 'dat', 'die', 'in', 'een', 'hij',
    'het', 'niet', 'zijn', 'is', 'was', 'op', 'aan', 'met', 'als', 'voor',
    'had', 'er', 'maar', 'om', 'hem', 'dan', 'zou', 'of', 'wat', 'mijn',
    'men', 'dit', 'zo', 'door', 'over', 'ze', 'zich', 'bij', 'ook', 'tot',
    'je', 'mij', 'uit', 'der', 'daar', 'haar', 'naar', 'heb', 'hoe', 'heeft',
    'hebben', 'deze', 'u', 'want', 'nog', 'zal', 'me', 'zij', 'nu', 'ge',
    'geen', 'omdat', 'iets', 'worden', 'toch', 'al', 'waren', 'veel', 'meer',
    'doen', 'toen', 'moet', 'ben', 'zonder', 'kan', 'hun', 'dus', 'alles',
    'onder', 'ja', 'eens', 'hier', 'wie', 'werd', 'altijd', 'doch', 'wordt',
    'wezen', 'kunnen', 'ons', 'zelf', 'tegen', 'na', 'reeds', 'wil', 'kon',
    'niets', 'uw', 'iemand', 'geweest', 'andere'
]

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    stop_words=DUTCH_STOP_WORDS,
    min_df=2,
    max_features=10000
)

# Fit and transform
try:
    tfidf_matrix = vectorizer.fit_transform(df_tfidf["all_text"])
    feature_names = np.array(vectorizer.get_feature_names_out())
    print(f"TF-IDF matrix created with {len(feature_names)} features")
except Exception as e:
    print(f"Error in TF-IDF: {e}")
    print("Trying without stop words...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=10000
    )
    tfidf_matrix = vectorizer.fit_transform(df_tfidf["all_text"])
    feature_names = np.array(vectorizer.get_feature_names_out())

# Find PFAS-related terms
pfas_mask = np.array([
    any(term in feature for term in PFAS_TERMS)
    for feature in feature_names
])

print(f"\nFound {pfas_mask.sum()} PFAS-related features out of {len(feature_names)} total features")

if not pfas_mask.any():
    print("Warning: No PFAS-related terms found in TF-IDF features")
    print("PFAS terms we're looking for:", PFAS_TERMS)
    print("\nSample of features found:")
    # Safely print features
    safe_features = []
    for feat in feature_names[:50]:
        try:
            safe_features.append(str(feat))
        except:
            safe_features.append("[encoding error]")
    print(safe_features)
    
    df_tfidf["pfas_tfidf_intensity"] = 0
    df_tfidf["pfas_tfidf_intensity_norm"] = 0
else:
    pfas_tfidf = tfidf_matrix[:, pfas_mask]
    df_tfidf["pfas_tfidf_intensity"] = pfas_tfidf.sum(axis=1).A1
    
    # Show which PFAS terms were found (with safe encoding)
    pfas_features = feature_names[pfas_mask]
    print(f"\nNumber of PFAS-related features found: {len(pfas_features)}")
    
    # Print a sample of PFAS features (first 20)
    print("Sample of PFAS-related features (first 20):")
    for i, feat in enumerate(pfas_features[:20]):
        try:
            print(f"  {i+1}. {feat}")
        except UnicodeEncodeError:
            print(f"  {i+1}. [feature with special characters]")
    
    # Normalize (recommended for regression)
    max_intensity = df_tfidf["pfas_tfidf_intensity"].max()
    if max_intensity > 0:
        df_tfidf["pfas_tfidf_intensity_norm"] = df_tfidf["pfas_tfidf_intensity"] / max_intensity
    else:
        df_tfidf["pfas_tfidf_intensity_norm"] = 0

print("\nTF-IDF intensity summary:")
print(df_tfidf["pfas_tfidf_intensity_norm"].describe())


# 3. TENDER PROCEDURE PROXY


print("\n" + "="*50)
print("Step 3: Creating procedure transparency proxy...")

# Define transparency mapping
procedure_map = {
    "openbaar": 1.0,
    "niet-openbaar": 0.5,
    "restricted": 0.5,
    "onderhands": 0.0,
    "negotiated": 0.0,
    "openbare": 1.0,
    "open": 1.0
}

# Normalize text
df_tfidf["procedure_type_clean"] = (
    df_tfidf["procedure_type"]
    .astype(str)
    .str.lower()
    .str.strip()
)

# Apply mapping
def map_procedure(value):
    value_lower = str(value).lower()
    for key, score in procedure_map.items():
        if key in value_lower:
            return score
    return 0.5  # Default middle value for unknown procedures

df_tfidf["procedure_proxy"] = df_tfidf["procedure_type_clean"].apply(map_procedure)

# Sanity check
print("\nProcedure type distribution:")
try:
    procedure_counts = df_tfidf[["procedure_type", "procedure_proxy"]].value_counts().head(10)
    print(procedure_counts)
except Exception as e:
    print(f"Could not print procedure counts: {e}")


# 4. TENDER SCOPE/CATEGORY PROXY


print("\n" + "="*50)
print("Step 4: Classifying tender scope...")

def classify_tender_scope(text: str) -> str:
    """
    Uses Ollama to classify tender text into:
    research, legal, or cleanup.
    Returns lowercase category or None.
    """
    # Truncate text if too long
    if len(text) > 4000:
        text = text[:4000] + "..."
    
    prompt = (
        "Classify the following public procurement tender into exactly one category:\n"
        "- research (studies, monitoring, analysis, investigation, testing)\n"
        "- legal (legal, policy, compliance, administrative, advisory, consulting)\n"
        "- cleanup (physical remediation, removal, treatment, decontamination, disposal)\n\n"
        "Only return the category name (research, legal, or cleanup).\n\n"
        f"Tender text:\n{text}"
    )

    try:
        result = subprocess.run(
            ["ollama", "run", "llama3"],
            input=prompt,
            text=True,
            encoding='utf-8',
            capture_output=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return None
        
        output = result.stdout.strip().lower()
        
        # Clean the output
        if "research" in output:
            return "research"
        elif "legal" in output:
            return "legal"
        elif "cleanup" in output or "remediation" in output:
            return "cleanup"
        else:
            return None
            
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        return None

# Try classification, but have a fallback
print("Attempting to classify with Ollama...")
try:
    # Check if Ollama is available
    test_result = subprocess.run(["ollama", "list"], capture_output=True, text=True, encoding='utf-8')
    if test_result.returncode != 0:
        print("Ollama not available. Using keyword-based classification...")
        raise Exception("Ollama not installed")
    
    # Classify only first 10 tenders for testing (to save time)
    print("Classifying first 10 tenders with Ollama (for testing)...")
    test_tenders = df_tfidf.head(10).copy()
    test_tenders["tender_scope"] = test_tenders["all_text"].apply(classify_tender_scope)
    
    # Check how many were classified
    classified_count = test_tenders["tender_scope"].notna().sum()
    print(f"Successfully classified {classified_count} out of 10 test tenders with Ollama")
    
    if classified_count > 5:  # If more than half worked, use Ollama for all
        print("Ollama working well. Classifying all tenders...")
        tqdm.pandas(desc="Classifying tenders")
        df_tfidf["tender_scope"] = df_tfidf["all_text"].progress_apply(classify_tender_scope)
    else:
        print("Ollama not reliable. Using keyword-based classification...")
        raise Exception("Ollama not reliable")
        
except Exception as e:
    print(f"Using keyword-based classification: {e}")
    
    def simple_classify(text):
        try:
            text_lower = str(text).lower()
        except:
            text_lower = ""
        
        # Cleanup/remediation keywords
        cleanup_keywords = ["sanering", "reinig", "verwijder", "behandel", "schoon", "opruim", "herstel",
                           "remediat", "decontaminat", "ontdoen", "afvoer", "verwerking"]
        
        # Research keywords
        research_keywords = ["onderzoek", "studie", "monitor", "analyse", "test", "inventarisat",
                            "meting", "bepaling", "karakterisering", "screen", "verkenning"]
        
        # Legal/administrative keywords
        legal_keywords = ["juridisch", "advies", "beleid", "compliance", "consult", "recht", "wet",
                         "regelgeving", "toezicht", "handhav", "vergunning", "procedur"]
        
        # Check for keywords
        if any(keyword in text_lower for keyword in cleanup_keywords):
            return "cleanup"
        elif any(keyword in text_lower for keyword in research_keywords):
            return "research"
        elif any(keyword in text_lower for keyword in legal_keywords):
            return "legal"
        else:
            return "unknown"
    
    df_tfidf["tender_scope"] = df_tfidf["all_text"].apply(simple_classify)

# Encode scope
scope_encoding = {
    "research": 0.0,
    "legal": 0.5,
    "cleanup": 1.0,
    "unknown": 0.5  # Default middle value for unknown
}

df_tfidf["scope_proxy"] = df_tfidf["tender_scope"].map(scope_encoding)

# For any remaining NaN values, use the median
if df_tfidf["scope_proxy"].isna().any():
    median_scope = df_tfidf["scope_proxy"].median()
    df_tfidf["scope_proxy"] = df_tfidf["scope_proxy"].fillna(median_scope)

print("\nScope distribution:")
try:
    print(df_tfidf["tender_scope"].value_counts(dropna=False))
except Exception as e:
    print(f"Could not print scope distribution: {e}")

# Save intermediate file
INTERMEDIATE_PATH = "tenders_ready_for_mlr.csv"
df_tfidf.to_csv(INTERMEDIATE_PATH, index=False, encoding='utf-8')
print(f"\nSaved classified tenders to {INTERMEDIATE_PATH}")


# 5. TEMPORAL ALIGNMENT


print("\n" + "="*50)
print("Step 5: Temporal alignment (monthly aggregation)...")

# Use the dataframes we already have
news = news_df.copy()
tenders = df_tfidf.copy()

# Create month column for aggregation
news["month"] = news["date"].dt.to_period("M").dt.to_timestamp()
tenders["month"] = tenders["publication_date"].dt.to_period("M").dt.to_timestamp()

print(f"\nNews date range: {news['month'].min()} to {news['month'].max()}")
print(f"Tenders date range: {tenders['month'].min()} to {tenders['month'].max()}")

# Aggregate news data by month
news_monthly = (
    news
    .groupby("month", as_index=False)
    .agg(
        mean_article_sentiment=("compound", "mean"),
        article_count=("compound", "count")
    )
)
print(f"\nNews aggregated into {len(news_monthly)} months")

# Aggregate tender data by month
tenders_monthly = (
    tenders
    .groupby("month", as_index=False)
    .agg(
        mean_pfas_intensity=("pfas_tfidf_intensity_norm", "mean"),
        mean_scope=("scope_proxy", "mean"),
        mean_transparency=("procedure_proxy", "mean"),
        tender_count=("scope_proxy", "count")
    )
)
print(f"Tenders aggregated into {len(tenders_monthly)} months")

# Merge news and tender data
df_mlr = pd.merge(
    news_monthly,
    tenders_monthly,
    on="month",
    how="inner"
)

print(f"\nAfter merging: {len(df_mlr)} months with both news and tenders")

if len(df_mlr) == 0:
    print("\nWARNING: No overlapping months found between news and tenders!")
    print("Trying alternative approach: using all available months...")
    
    # Create complete month range
    all_months = pd.concat([
        news_monthly[["month"]],
        tenders_monthly[["month"]]
    ]).drop_duplicates().sort_values("month")
    
    # Merge with news
    df_mlr = pd.merge(all_months, news_monthly, on="month", how="left")
    
    # Merge with tenders
    df_mlr = pd.merge(df_mlr, tenders_monthly, on="month", how="left")
    
    # Fill missing values with means or zeros
    for col in ["mean_article_sentiment", "mean_pfas_intensity", "mean_scope", "mean_transparency"]:
        if col in df_mlr.columns:
            if col == "mean_pfas_intensity":
                df_mlr[col] = df_mlr[col].fillna(0)
            else:
                df_mlr[col] = df_mlr[col].fillna(df_mlr[col].mean())
    
    # Fill count columns with zeros
    for col in ["article_count", "tender_count"]:
        if col in df_mlr.columns:
            df_mlr[col] = df_mlr[col].fillna(0)

print(f"\nFinal MLR dataset shape: {df_mlr.shape}")
print(f"Time range: {df_mlr['month'].min()} to {df_mlr['month'].max()}")

# Display first few rows safely
print("\nFirst few rows:")
try:
    print(df_mlr.head().to_string())
except:
    print("Preview of data (first 5 rows):")
    for i in range(min(5, len(df_mlr))):
        row = df_mlr.iloc[i]
        print(f"Month: {row['month']}, Sentiment: {row.get('mean_article_sentiment', 'N/A'):.4f}, "
              f"PFAS Intensity: {row.get('mean_pfas_intensity', 'N/A'):.4f}")

# Save MLR dataset
MLR_PATH = "mlr_monthly_dataset.csv"
df_mlr.to_csv(MLR_PATH, index=False, encoding='utf-8')
print(f"\nSaved MLR dataset to {MLR_PATH}")


# 6. MULTIPLE LINEAR REGRESSION + STARGAZER


print("\n" + "="*50)
print("Step 6: Running multiple linear regression...")

# Check if we have enough data
if len(df_mlr) < 3:
    print("ERROR: Not enough data points for regression analysis")
    print(f"Only {len(df_mlr)} months available, need at least 3")
    print("\nCreating summary statistics only...")
    
    stats_df = df_mlr.describe().transpose()
    stats_df.to_csv("descriptive_statistics.csv", encoding='utf-8')
    print("Saved descriptive statistics to descriptive_statistics.csv")
    
else:
    print(f"Running regression with {len(df_mlr)} data points...")
    
    # Prepare variables
    Y = df_mlr["mean_article_sentiment"]
    X = df_mlr[["mean_pfas_intensity", "mean_scope", "mean_transparency"]]
    
    # Add intercept
    X = sm.add_constant(X)
    
    # Run regression
    try:
        model = sm.OLS(Y, X).fit()

        print("\n" + "="*50)
        print("REGRESSION RESULTS")
        print("="*50)
        
        # Create a safe summary
        results_summary = []
        results_summary.append("="*50)
        results_summary.append("MULTIPLE LINEAR REGRESSION RESULTS")
        results_summary.append("="*50)
        results_summary.append(f"Dependent variable: Mean Article Sentiment")
        results_summary.append(f"Number of observations: {len(df_mlr)}")
        results_summary.append(f"R-squared: {model.rsquared:.4f}")
        results_summary.append(f"Adj. R-squared: {model.rsquared_adj:.4f}")
        results_summary.append("="*50)
        results_summary.append("COEFFICIENTS:")
        results_summary.append("="*50)
        
        variables = ["Intercept", "PFAS Intensity", "Action Scope", "Transparency"]
        for i, (coef, se, t, p) in enumerate(zip(model.params, model.bse, model.tvalues, model.pvalues)):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            results_summary.append(f"{variables[i]:20} {coef:10.4f} ({se:.4f}) t={t:.3f} p={p:.4f} {sig}")
        
        results_summary.append("="*50)
        results_summary.append("* p<0.05, ** p<0.01, *** p<0.001")
        
        # Print results
        for line in results_summary:
            print(line)
        
        # Save results to text file
        with open("regression_results.txt", "w", encoding='utf-8') as f:
            f.write("\n".join(results_summary))
        
        print("\nSaved regression results to regression_results.txt")
        
        # Try to create Stargazer table
        try:
            stargazer = Stargazer([model])
            stargazer.title("Multiple Linear Regression: PFAS Communication Tone")
            
            stargazer.custom_columns(["Article Sentiment"], [1])
            
            stargazer.covariate_order([
                "const",
                "mean_pfas_intensity",
                "mean_scope",
                "mean_transparency"
            ])
            
            stargazer.rename_covariates({
                "const": "Intercept",
                "mean_pfas_intensity": "PFAS Intensity",
                "mean_scope": "Action Scope",
                "mean_transparency": "Transparency"
            })
            
            # Save HTML table
            with open("mlr_results.html", "w", encoding='utf-8') as f:
                f.write(stargazer.render_html())
            
            print("Saved HTML regression table to mlr_results.html")
            
        except Exception as e:
            print(f"Note: Could not create Stargazer table: {e}")
            
        # Save detailed results to CSV
        results_df = pd.DataFrame({
            'Variable': ['Intercept', 'PFAS Intensity', 'Action Scope', 'Transparency'],
            'Coefficient': model.params.values,
            'Std_Error': model.bse.values,
            't_value': model.tvalues.values,
            'p_value': model.pvalues.values,
            'Significant_05': model.pvalues.values < 0.05,
            'Significant_01': model.pvalues.values < 0.01,
            'Significant_001': model.pvalues.values < 0.001
        })
        
        results_df.to_csv("mlr_detailed_results.csv", index=False, encoding='utf-8')
        print("Saved detailed results to mlr_detailed_results.csv")
        
    except Exception as e:
        print(f"Error running regression: {e}")


# 7. SUMMARY STATISTICS


print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)

print(f"\nTotal news articles: {len(news_df)}")
print(f"Total tenders: {len(tenders_df)}")
print(f"Months in analysis: {len(df_mlr)}")

if len(df_mlr) > 0:
    print("\nDependent variable (mean_article_sentiment):")
    print(f"  Mean: {df_mlr['mean_article_sentiment'].mean():.4f}")
    print(f"  Std: {df_mlr['mean_article_sentiment'].std():.4f}")
    print(f"  Min: {df_mlr['mean_article_sentiment'].min():.4f}")
    print(f"  Max: {df_mlr['mean_article_sentiment'].max():.4f}")
    
    print("\nIndependent variables:")
    print(f"  PFAS Intensity: mean={df_mlr['mean_pfas_intensity'].mean():.4f}, std={df_mlr['mean_pfas_intensity'].std():.4f}")
    print(f"  Action Scope: mean={df_mlr['mean_scope'].mean():.4f}, std={df_mlr['mean_scope'].std():.4f}")
    print(f"  Transparency: mean={df_mlr['mean_transparency'].mean():.4f}, std={df_mlr['mean_transparency'].std():.4f}")
    
    # Correlation matrix
    print("\nCorrelation matrix:")
    try:
        corr_matrix = df_mlr[["mean_article_sentiment", "mean_pfas_intensity", "mean_scope", "mean_transparency"]].corr()
        print(corr_matrix.to_string())
        corr_matrix.to_csv("correlation_matrix.csv", encoding='utf-8')
        print("Saved correlation matrix to correlation_matrix.csv")
    except Exception as e:
        print(f"Could not calculate correlation matrix: {e}")

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)

# Save final dataset info
dataset_info = {
    'total_news_articles': len(news_df),
    'total_tenders': len(tenders_df),
    'months_in_analysis': len(df_mlr),
    'news_date_range': f"{news_df['date'].min()} to {news_df['date'].max()}",
    'tenders_date_range': f"{tenders_df['publication_date'].min()} to {tenders_df['publication_date'].max()}"
}

import json
with open('analysis_summary.json', 'w', encoding='utf-8') as f:
    json.dump(dataset_info, f, indent=2, default=str)


print("\nSaved analysis summary to analysis_summary.json")
