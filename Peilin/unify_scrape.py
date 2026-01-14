import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

# Load the data
df = pd.read_csv('pfas_news_final.csv')

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Function to get VADER sentiment scores
def get_vader_scores(text):
    if pd.isna(text):
        return {'compound': 0, 'neg': 0, 'neu': 0, 'pos': 0}
    return analyzer.polarity_scores(str(text))

# Apply VADER to title and text
df['title_sentiment'] = df['title_en'].apply(lambda x: get_vader_scores(x)['compound'])
df['text_sentiment'] = df['text_en'].apply(lambda x: get_vader_scores(x)['compound'])

# Get detailed sentiment scores for text
sentiment_details = df['text_en'].apply(lambda x: get_vader_scores(x))
df[['neg', 'neu', 'pos', 'compound']] = pd.DataFrame(sentiment_details.tolist())

# Add sentiment label
def get_sentiment_label(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['sentiment_label'] = df['compound'].apply(get_sentiment_label)

# Show basic statistics
print("=" * 80)
print("VADER SENTIMENT ANALYSIS RESULTS")
print("=" * 80)
print(f"\nTotal articles: {len(df)}")
print(f"Data shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

print(f"\nSentiment Distribution:")
print(df['sentiment_label'].value_counts())

print(f"\nSentiment Score Statistics:")
print(f"Compound score mean: {df['compound'].mean():.4f}")
print(f"Compound score std: {df['compound'].std():.4f}")
print(f"Compound score min: {df['compound'].min():.4f}")
print(f"Compound score max: {df['compound'].max():.4f}")

print(f"\nText vs Title Sentiment Correlation: {df['text_sentiment'].corr(df['title_sentiment']):.4f}")

# Show top 5 most negative articles
print("\n" + "=" * 80)
print("TOP 5 MOST NEGATIVE ARTICLES")
print("=" * 80)
most_negative = df.nsmallest(5, 'compound')[['date', 'title_en', 'compound', 'sentiment_label']]
for idx, row in most_negative.iterrows():
    print(f"\nDate: {row['date'][:10]}")
    print(f"Title: {row['title_en'][:100]}...")
    print(f"Sentiment Score: {row['compound']:.4f} ({row['sentiment_label']})")

# Show top 5 most positive articles
print("\n" + "=" * 80)
print("TOP 5 MOST POSITIVE ARTICLES")
print("=" * 80)
most_positive = df.nlargest(5, 'compound')[['date', 'title_en', 'compound', 'sentiment_label']]
for idx, row in most_positive.iterrows():
    print(f"\nDate: {row['date'][:10]}")
    print(f"Title: {row['title_en'][:100]}...")
    print(f"Sentiment Score: {row['compound']:.4f} ({row['sentiment_label']})")

# Show sentiment by source
print("\n" + "=" * 80)
print("SENTIMENT BY SOURCE")
print("=" * 80)
sentiment_by_source = df.groupby('source').agg({
    'compound': ['mean', 'std', 'count'],
    'sentiment_label': lambda x: x.value_counts().to_dict()
}).round(4)

for source in df['source'].unique():
    source_data = df[df['source'] == source]
    print(f"\n{source}:")
    print(f"  Articles: {len(source_data)}")
    print(f"  Avg sentiment: {source_data['compound'].mean():.4f}")
    print(f"  Sentiment distribution: {dict(source_data['sentiment_label'].value_counts())}")

# Show a few examples with their scores
print("\n" + "=" * 80)
print("SAMPLE ANALYZED ARTICLES")
print("=" * 80)
sample = df.sample(3, random_state=42)
for idx, row in sample.iterrows():
    print(f"\nSource: {row['source']}")
    print(f"Date: {row['date'][:10]}")
    print(f"Title: {row['title_en'][:80]}...")
    print(f"Title sentiment: {row['title_sentiment']:.4f}")
    print(f"Text sentiment: {row['compound']:.4f}")
    print(f"Label: {row['sentiment_label']}")
    print(f"Negative: {row['neg']:.4f}, Neutral: {row['neu']:.4f}, Positive: {row['pos']:.4f}")
    
# Save results to CSV
output_file = 'pfas_news_vader_analysis.csv'
df.to_csv(output_file, index=False)
print(f"\nDetailed results saved to: {output_file}")