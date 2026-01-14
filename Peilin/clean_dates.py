import pandas as pd
import locale

# Try to ensure Dutch locale for month names (only needed if you have text dates)
try:
    locale.setlocale(locale.LC_TIME, "nl_NL.UTF-8")
except locale.Error:
    try:
        locale.setlocale(locale.LC_TIME, "nl_NL")  # Try without UTF-8
    except locale.Error:
        pass  # If this fails, continue anyway

# Load CSV with error handling for inconsistent fields
try:
    df = pd.read_csv("pfas_news_vader_analysis.csv", on_bad_lines='warn')
except pd.errors.ParserError:
    print("Error reading CSV. Trying with error handling...")
    # Try reading with different error handling strategies
    try:
        # First try with quoting
        df = pd.read_csv("pfas_news_vader_analysis.csv", 
                        on_bad_lines='skip',  # Skip problematic lines
                        quoting=3)  # QUOTE_NONE
    except:
        # If that fails, try reading line by line
        import csv
        rows = []
        with open("pfas_news_vader_analysis.csv", 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            for i, row in enumerate(reader, start=2):  # Start at line 2 (after header)
                if len(row) != len(header):
                    print(f"Line {i}: Expected {len(header)} fields, found {len(row)} fields. Adjusting...")
                    # Adjust row to match header length
                    if len(row) > len(header):
                        # Combine extra fields into the last column
                        row = row[:len(header)-1] + [','.join(row[len(header)-1:])]
                    else:
                        # Add empty strings for missing fields
                        row = row + [''] * (len(header) - len(row))
                rows.append(row)
        
        df = pd.DataFrame(rows, columns=header)

# Check what the date column looks like
print("\nDate column sample:")
print(df["date"].head())
print("\nDate column dtype:", df["date"].dtype)
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Parse dates - your dates appear to already be in ISO format
df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

# Check how many dates failed to parse
failed_dates = df[df["date_parsed"].isna()]
if len(failed_dates) > 0:
    print(f"\nWarning: {len(failed_dates)} dates could not be parsed:")
    print(failed_dates["date"].unique()[:10])  # Show first 10 unique failed dates

# Drop rows without valid dates
df_clean = df.dropna(subset=["date_parsed"]).copy()

# Standardize format (already in YYYY-MM-DD format)
df_clean["date_standardized"] = df_clean["date_parsed"].dt.strftime("%Y-%m-%d")

# Optional: extract year / month / day as separate columns
df_clean["year"] = df_clean["date_parsed"].dt.year
df_clean["month"] = df_clean["date_parsed"].dt.month
df_clean["day"] = df_clean["date_parsed"].dt.day

# Extract time components if needed
df_clean["hour"] = df_clean["date_parsed"].dt.hour
df_clean["minute"] = df_clean["date_parsed"].dt.minute

# Drop the intermediate column if you want
df_clean = df_clean.drop(columns=["date_parsed"])

# Reorder columns if desired (optional)
# Create a list of new columns to add at the beginning
new_cols = ["date_standardized", "year", "month", "day", "hour", "minute"]
existing_cols = [col for col in df_clean.columns if col not in new_cols]
df_clean = df_clean[existing_cols[:1] + new_cols + existing_cols[1:]]

# Save cleaned file
df_clean.to_csv("clean_pfas_news_vader_analysis.csv", index=False)

print(f"\nCleaning complete. Original rows: {len(df)}, Cleaned rows: {len(df_clean)}")
print(f"Removed {len(df) - len(df_clean)} rows with invalid dates.")
print(f"Saved to: clean_pfas_news_vader_analysis.csv")