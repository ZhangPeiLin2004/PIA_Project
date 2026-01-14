# scrape_omgevingsdienst_pfas.py
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import json
import re
from datetime import datetime
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ==============================
# STEP 1 — Scrape data
# ==============================
opts = Options()
opts.headless = True
driver = webdriver.Chrome(options=opts)

BASE = "https://www.omgevingsdienst.nl/?s=PFAS&engine=default"
driver.get(BASE)
time.sleep(2)

# Search results contain article links inside <a> tags pointing to /nieuws/ or similar
items = driver.find_elements(By.CSS_SELECTOR, "a[href*='pfas'], a[href*='PFAS'], a[href*='/nieuws/']")
seen = set()
links = []

for a in items:
    href = a.get_attribute("href")
    if href and "omgevingsdienst.nl" in href and href not in seen:
        seen.add(href)
        links.append(href)

parsed = []
for link in links:
    driver.get(link)
    time.sleep(1)

    # title
    try:
        title = driver.find_element(By.CSS_SELECTOR, "h1").text.strip()
    except:
        title = ""

    # main text
    try:
        main_text = driver.find_element(By.CSS_SELECTOR, "main").text
    except:
        main_text = ""

    # date extraction (Omgevingsdienst uses time/date inside <time> or meta)
    date = ""
    try:
        date_elem = driver.find_element(By.CSS_SELECTOR, "time")
        date = date_elem.get_attribute("datetime") or date_elem.text
    except:
        # fallback: search for yyyy-mm-dd
        match = re.search(r"\b20\d{2}-\d{2}-\d{2}\b", main_text)
        if match:
            date = match.group(0)
        else:
            # Try to find any date pattern
            match = re.search(r"\b\d{1,2}\s+\w+\s+20\d{2}\b", main_text[:500])
            if match:
                date = match.group(0)

    # No "ministry" concept on this website
    ministry = ""

    parsed.append({
        "url": link,
        "title": title,
        "date": date,
        "ministry": ministry,
        "body_snippet": main_text[:1500] if main_text else "",
        "full_text": main_text  # Add full text for VADER analysis
    })

driver.quit()

# ==============================
# STEP 2 — Save scraped data
# ==============================
INPUT_FILE = r"E:\Csci_2\omgevingsdienst_pfas_news.json"
with open(INPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(parsed, f, indent=2, ensure_ascii=False)

print(f"Scraped {len(parsed)} PFAS items saved to {INPUT_FILE}")

# ==============================
# STEP 3 — Load data for VADER processing
# ==============================
df = pd.read_json(INPUT_FILE)

# Initialize VADER analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get VADER sentiment scores
def get_vader_scores(text):
    if pd.isna(text) or not text.strip():
        return {'compound': 0, 'neg': 0, 'neu': 0, 'pos': 0}
    return analyzer.polarity_scores(str(text))

# Function to get sentiment label
def get_sentiment_label(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply VADER sentiment analysis
print("\nPerforming VADER sentiment analysis...")

# Get sentiment scores for titles and full text
df['title_en'] = df['title']
df['text_en'] = df['full_text']

# Apply VADER to title and text
df['title_sentiment'] = df['title_en'].apply(lambda x: get_vader_scores(x)['compound'])
df['text_sentiment'] = df['text_en'].apply(lambda x: get_vader_scores(x)['compound'])

# Get detailed sentiment scores for text
sentiment_details = df['text_en'].apply(lambda x: get_vader_scores(x))
df[['neg', 'neu', 'pos', 'compound']] = pd.DataFrame(sentiment_details.tolist())

# Add sentiment label
df['sentiment_label'] = df['compound'].apply(get_sentiment_label)

# Add source column (for consistency with previous analysis)
df['source'] = 'omgevingsdienst'

# Format date for consistency
def format_date(date_str):
    if not date_str:
        return ''
    # Try to parse various date formats
    date_patterns = [
        r'(\d{4}-\d{2}-\d{2})',  # yyyy-mm-dd
        r'(\d{1,2}\s+\w+\s+\d{4})',  # dd month yyyy
        r'(\d{1,2}-\d{1,2}-\d{4})',  # dd-mm-yyyy
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, str(date_str))
        if match:
            date_part = match.group(1)
            try:
                # Try to parse and standardize
                for fmt in ['%Y-%m-%d', '%d %B %Y', '%d-%m-%Y', '%d %b %Y']:
                    try:
                        dt = datetime.strptime(date_part, fmt)
                        return dt.strftime('%Y-%m-%d')
                    except:
                        continue
            except:
                pass
    return date_str

df['date'] = df['date'].apply(format_date)

# ==============================
# STEP 4 — Create final dataframe with same structure as previous analysis
# ==============================
# Select and rename columns to match previous analysis
final_df = df[['source', 'date', 'url', 'title_en', 'text_en', 
               'title_sentiment', 'text_sentiment', 'neg', 'neu', 'pos', 
               'compound', 'sentiment_label']].copy()

# Add any missing columns for consistency
if 'ministry' in df.columns:
    final_df['ministry'] = df['ministry']

print("\n" + "=" * 80)
print("VADER SENTIMENT ANALYSIS RESULTS")
print("=" * 80)
print(f"\nTotal articles: {len(final_df)}")
print(f"Data shape: {final_df.shape}")
print(f"Date range: {final_df['date'].min() if final_df['date'].notna().any() else 'N/A'} "
      f"to {final_df['date'].max() if final_df['date'].notna().any() else 'N/A'}")

print(f"\nSentiment Distribution:")
print(final_df['sentiment_label'].value_counts())

print(f"\nSentiment Score Statistics:")
print(f"Compound score mean: {final_df['compound'].mean():.4f}")
print(f"Compound score std: {final_df['compound'].std():.4f}")
print(f"Compound score min: {final_df['compound'].min():.4f}")
print(f"Compound score max: {final_df['compound'].max():.4f}")

# Show sample of results
print("\n" + "=" * 80)
print("SAMPLE ARTICLES WITH SENTIMENT ANALYSIS")
print("=" * 80)
sample_size = min(3, len(final_df))
for idx, row in final_df.head(sample_size).iterrows():
    print(f"\nTitle: {row['title_en'][:80]}..." if len(str(row['title_en'])) > 80 else f"\nTitle: {row['title_en']}")
    print(f"Date: {row['date']}")
    print(f"Sentiment Score: {row['compound']:.4f} ({row['sentiment_label']})")
    print(f"Negative: {row['neg']:.4f}, Neutral: {row['neu']:.4f}, Positive: {row['pos']:.4f}")

# ==============================
# STEP 5 — Save to CSV (matching previous format)
# ==============================
OUTPUT_FILE = r"E:\Csci_2\omgevingsdienst_pfas_vader_analysis.csv"
final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

print(f"\n" + "=" * 80)
print(f"Analysis complete!")
print(f"Scraped {len(parsed)} articles")
print(f"Saved VADER analysis to: {OUTPUT_FILE}")
print("=" * 80)

# Show final column structure
print("\nFinal DataFrame Columns:")
print(final_df.columns.tolist())
print(f"\nFirst few rows:")
print(final_df.head())