# scrape_rijksoverheid_pfas.py
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

BASE = "https://www.rijksoverheid.nl/onderwerpen/pfas/nieuws"
driver.get(BASE)
time.sleep(1)

items = driver.find_elements(By.CSS_SELECTOR, "main a[href*='/onderwerpen/pfas/nieuws/']")
seen = set()
links = []

for a in items:
    href = a.get_attribute("href")
    if href and href not in seen:
        seen.add(href)
        links.append(href)

parsed = []
for link in links:
    driver.get(link)
    time.sleep(1)
    try:
        title = driver.find_element(By.CSS_SELECTOR, "h1").text.strip()
    except:
        title = ""
    try:
        main_text = driver.find_element(By.CSS_SELECTOR, "main").text
    except:
        main_text = ""
    # extract date
    date = ""
    try:
        header = driver.find_element(By.CSS_SELECTOR, ".page__intro, main").text
        for line in header.splitlines():
            if "Nieuwsbericht" in line:
                date = line.replace("Nieuwsbericht", "").strip().strip("| ").strip()
                break
    except:
        pass

    ministry = ""
    try:
        ministry_elem = driver.find_element(By.XPATH, "//main//*[contains(text(),'Verantwoordelijk')]/following-sibling::*[1]")
        ministry = ministry_elem.text.strip()
    except:
        pass

    parsed.append({
        "url": link,
        "title": title,
        "date": date,
        "ministry": ministry,
        "body_snippet": main_text[:1500],
        "full_text": main_text  # Add full text for VADER analysis
    })

driver.quit()

# ==============================
# STEP 2 — Save scraped data
# ==============================
INPUT_FILE = r"E:\Csci_2\rijksoverheid_pfas_news.json"
with open(INPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(parsed, f, indent=2, ensure_ascii=False)

print(f"Scraped {len(parsed)} PFAS news articles saved to {INPUT_FILE}")

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
df['source'] = 'rijksoverheid'

# Format date for consistency
def format_date(date_str):
    if not date_str:
        return ''
    
    # Common date patterns on rijksoverheid.nl
    date_patterns = [
        r'(\d{1,2}-\d{1,2}-\d{4})',  # dd-mm-yyyy
        r'(\d{1,2}\s+\w+\s+\d{4})',  # dd month yyyy
        r'(\d{4}-\d{2}-\d{2})',  # yyyy-mm-dd
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, str(date_str))
        if match:
            date_part = match.group(1)
            try:
                # Try to parse various date formats
                for fmt in ['%d-%m-%Y', '%d %B %Y', '%Y-%m-%d', '%d %b %Y']:
                    try:
                        dt = datetime.strptime(date_part, fmt)
                        return dt.strftime('%Y-%m-%d')
                    except:
                        continue
            except:
                pass
    
    # If no pattern matches, try to extract just the year
    year_match = re.search(r'\b(20\d{2})\b', str(date_str))
    if year_match:
        return f"{year_match.group(1)}-01-01"  # Default to Jan 1 if only year found
    
    return date_str

df['date'] = df['date'].apply(format_date)

# ==============================
# STEP 4 — Create final dataframe with same structure as previous analysis
# ==============================
# Select and rename columns to match previous analysis
final_df = df[['source', 'date', 'url', 'title_en', 'text_en', 
               'title_sentiment', 'text_sentiment', 'neg', 'neu', 'pos', 
               'compound', 'sentiment_label']].copy()

# Add ministry column if it exists
if 'ministry' in df.columns:
    final_df['ministry'] = df['ministry']

print("\n" + "=" * 80)
print("VADER SENTIMENT ANALYSIS RESULTS - RIJKSOVERHEID")
print("=" * 80)
print(f"\nTotal articles: {len(final_df)}")
print(f"Data shape: {final_df.shape}")
print(f"Date range: {final_df['date'].min() if final_df['date'].notna().any() else 'N/A'} "
      f"to {final_df['date'].max() if final_df['date'].notna().any() else 'N/A'}")

print(f"\nSentiment Distribution:")
sentiment_counts = final_df['sentiment_label'].value_counts()
print(sentiment_counts)

print(f"\nSentiment Score Statistics:")
print(f"Compound score mean: {final_df['compound'].mean():.4f}")
print(f"Compound score std: {final_df['compound'].std():.4f}")
print(f"Compound score min: {final_df['compound'].min():.4f}")
print(f"Compound score max: {final_df['compound'].max():.4f}")

# Calculate correlation between title and text sentiment
correlation = final_df['title_sentiment'].corr(final_df['text_sentiment'])
print(f"Title vs Text sentiment correlation: {correlation:.4f}")

# Show sample of results
print("\n" + "=" * 80)
print("SAMPLE ARTICLES WITH SENTIMENT ANALYSIS")
print("=" * 80)
sample_size = min(3, len(final_df))
for idx, row in final_df.head(sample_size).iterrows():
    title_display = f"{row['title_en'][:80]}..." if len(str(row['title_en'])) > 80 else str(row['title_en'])
    print(f"\nTitle: {title_display}")
    print(f"Date: {row['date']}")
    if 'ministry' in row and pd.notna(row['ministry']):
        print(f"Ministry: {row['ministry']}")
    print(f"Sentiment Score: {row['compound']:.4f} ({row['sentiment_label']})")
    print(f"Negative: {row['neg']:.4f}, Neutral: {row['neu']:.4f}, Positive: {row['pos']:.4f}")

# Show most positive and negative articles
if len(final_df) > 0:
    print("\n" + "=" * 80)
    print("MOST NEGATIVE ARTICLE")
    print("=" * 80)
    most_negative = final_df.loc[final_df['compound'].idxmin()]
    print(f"Title: {most_negative['title_en'][:100]}..." if len(str(most_negative['title_en'])) > 100 else f"Title: {most_negative['title_en']}")
    print(f"Sentiment: {most_negative['compound']:.4f} ({most_negative['sentiment_label']})")
    
    print("\n" + "=" * 80)
    print("MOST POSITIVE ARTICLE")
    print("=" * 80)
    most_positive = final_df.loc[final_df['compound'].idxmax()]
    print(f"Title: {most_positive['title_en'][:100]}..." if len(str(most_positive['title_en'])) > 100 else f"Title: {most_positive['title_en']}")
    print(f"Sentiment: {most_positive['compound']:.4f} ({most_positive['sentiment_label']})")

# ==============================
# STEP 5 — Save to CSV (matching previous format)
# ==============================
OUTPUT_FILE = r"E:\Csci_2\rijksoverheid_pfas_vader_analysis.csv"
final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

print(f"\n" + "=" * 80)
print(f"Analysis complete!")
print(f"Scraped {len(parsed)} articles from Rijksoverheid")
print(f"Sentiment distribution: {dict(sentiment_counts)}")
print(f"Saved VADER analysis to: {OUTPUT_FILE}")
print("=" * 80)

# Show final column structure
print("\nFinal DataFrame Columns:")
print(final_df.columns.tolist())
print(f"\nFirst few rows:")
print(final_df.head())

# ==============================
# STEP 6 — Optional: Save aggregated summary
# ==============================
summary_stats = {
    'total_articles': len(final_df),
    'avg_sentiment': final_df['compound'].mean(),
    'sentiment_distribution': dict(final_df['sentiment_label'].value_counts()),
    'date_range': f"{final_df['date'].min()} to {final_df['date'].max()}",
    'source': 'rijksoverheid'
}

summary_file = r"E:\Csci_2\rijksoverheid_pfas_summary.json"
with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    
print(f"\nSummary statistics saved to: {summary_file}")