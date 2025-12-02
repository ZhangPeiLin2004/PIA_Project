# scrape_omgevingsdienst_pfas.py
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import json
import re
from textblob import TextBlob
from datetime import datetime
import pandas as pd

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

    # No "ministry" concept on this website → keep consistent with your variables
    ministry = ""

    parsed.append({
        "url": link,
        "title": title,
        "date": date,
        "ministry": ministry,
        "body_snippet": main_text[:1500]
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
# STEP 3 — Load data for ML processing
# ==============================
df = pd.read_json(INPUT_FILE)

# Dictionary limitation --- Words cannot warrant their categories as assigned due to the blind-search method
ACTION_WORDS = [
    "verminderen", "aanpakken", "onderzoeken", "regelen",
    "handhaven", "stoppen", "verbieden", "beschermen", "minimaliseren"
]
RISK_WORDS = [
    "gevaar", "risico", "zorgwekkend", "schadelijk", "bedreiging",
    "ziekte", "vervuiling", "toxiciteit", "angst", "probleem"
]
TRUST_WORDS = [
    "veilig", "transparant", "bescherming", "betrouwbaar",
    "verantwoord", "samenwerking", "gezondheid", "openheid", "duidelijk"
]

def count_keywords(text, keywords):
    text = text.lower()
    return sum(1 for kw in keywords if kw in text)

def get_sentiment(text):
    if not text.strip():
        return 0
    blob = TextBlob(text)
    return round(blob.sentiment.polarity, 3)

def extract_year(date_str):
    match = re.search(r"\b(20\d{2})\b", date_str)
    return int(match.group(1)) if match else None

# ==============================
# STEP 4 — Feature extraction
# ==============================
features = []
for _, row in df.iterrows():
    text = row["body_snippet"]
    features.append({
        "url": row["url"],
        "title": row["title"],
        "year": extract_year(row["date"]),
        "ministry": row["ministry"],
        "title_length": len(row["title"].split()),
        "body_length": len(text.split()),
        "n_action_words": count_keywords(text, ACTION_WORDS),
        "n_risk_words": count_keywords(text, RISK_WORDS),
        "n_trust_cues": count_keywords(text, TRUST_WORDS),
        "tone_sentiment": get_sentiment(text),
        "trust_signal_score": round(
            (count_keywords(text, TRUST_WORDS) - count_keywords(text, RISK_WORDS)) /
            max(1, len(text.split())) * 1000, 3
        )
    })

df_features = pd.DataFrame(features)

def classify_frame(title, body):
    text = (title + " " + body).lower()
    if "verbod" in text or "regels" in text:
        return "regulatory"
    elif "gezondheid" in text or "ziekte" in text:
        return "health"
    elif "milieu" in text or "beschermen" in text:
        return "environmental"
    else:
        return "general"

df_features["framing_focus"] = [
    classify_frame(row["title"], row["url"]) for _, row in df_features.iterrows()
]

# ==============================
# STEP 5 — Save ML-ready dataset
# ==============================
OUTPUT_FILE = r"E:\Csci_2\omgevingsdienst_pfas_ml_features.csv"
df_features.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print(f"ML-ready dataset saved to {OUTPUT_FILE}")
print("\nSample of extracted features:\n")
print(df_features.head())

# ==============================
# STEP 6 — Test
# ==============================
df = pd.read_csv(OUTPUT_FILE)
print(df.columns)
print(df.head(10))
df.describe()
