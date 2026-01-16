import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("tenders_with_text.csv")

documents = df['all_text'].fillna("")
documents = documents.str.lower()

vectorizer = TfidfVectorizer(vocabulary=["pfas"])  # Force PFAS as the only term
tfidf_matrix = vectorizer.fit_transform(documents)

df["tfidf_pfas"] = tfidf_matrix.toarray()[:, 0]

# Optional: inspect results
print(df[["tender_id", "tfidf_pfas"]])

# Save to CSV
df[["tender_id", "tfidf_pfas"]].to_csv(
    "tenders_with_pfas_tfidf.csv",
    index=False
)
