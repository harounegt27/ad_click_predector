import requests
from bs4 import BeautifulSoup
import pandas as pd
import sys
import os
import datetime
import holidays

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_cnbc_trending_keywords():
    """
    Scrapes CNBC Technology section with a User-Agent header to avoid blocking.
    """
    url = "https://www.cnbc.com/technology/"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    
    try:
        response = requests.get(url, headers=headers)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            headlines = []
            
            # Try specific class
            for item in soup.find_all('a', class_='Card-title'):
                headlines.append(item.get_text())
            
            # Fallback to H3
            if not headlines:
                print("Specific class not found, trying generic H3 tags...")
                for item in soup.find_all('h3'):
                    headlines.append(item.get_text())

            keywords = set()
            for head in headlines:
                for word in head.split():
                    if len(word) > 4:
                        keywords.add(word.lower().replace('.', '').replace(',', ''))
            
            return list(keywords)
        else:
            print(f"Failed to retrieve CNBC page. Status Code: {response.status_code}")
            return []
    except Exception as e:
        print(f"Scraping error: {e}")
        return []

def check_is_holiday():
    """
    External Enrichment: Checks if today is a holiday in TUNISIA.
    Returns 1 if True, 0 if False.
    """
    tn_holidays = holidays.TN()
    today = datetime.date.today()
    
    if today in tn_holidays:
        holiday_name = tn_holidays[today]
        print(f"Today is a Tunisian holiday: {holiday_name}")
        return 1
    else:
        print("Today is NOT a holiday.")
        return 0

def get_tech_savvy_status(row):
    """
    Feature Engineering: Determines Tech-Savvy level based on Age and Device.
    """
    # Handle missing age: treat as average if missing
    if pd.isna(row['age']):
        age = 35 # Average assumption
    else:
        age = row['age']
        
    device = row['device_type']
    
    # Logic
    if age < 30 and device in ['Mobile', 'Tablet']:
        return 'High'
    elif age > 50 and device == 'Desktop':
        return 'Low'
    else:
        return 'Medium'

def enrich_dataset():
    # 1. Load Data (RELATIVE PATH)
    try:
        df = pd.read_csv('../data/ad_click_dataset.csv')
    except FileNotFoundError:
        print("Error: 'ad_click_dataset.csv' not found in ../data/ folder.")
        return

    # 2. Get Trending Keywords (From CNBC)
    print("\n--- Feature 1: Trending Keywords ---")
    print("Fetching trending keywords from CNBC...")
    trending_keywords = get_cnbc_trending_keywords()
    print(f"Trending Keywords: {trending_keywords}")

    def check_trending(history):
        if pd.isna(history):
            return 0
        for keyword in trending_keywords:
            if keyword in str(history).lower():
                return 1
        return 0
    
    df['is_trending'] = df['browsing_history'].apply(check_trending)

    # 3. Tech-Savvy Proxy (Feature Engineering)
    print("\n--- Feature 2: Tech-Savvy Proxy ---")
    print("Calculating tech-savviness based on Age and Device...")
    df['tech_savvy_segment'] = df.apply(get_tech_savvy_status, axis=1)
    print("Segments created: High, Medium, Low.")

    # 4. Holiday Check (External Enrichment)
    print("\n--- Feature 3: Holiday Enrichment ---")
    is_holiday_today = check_is_holiday()
    # Apply the same holiday status to all rows (snapshot logic)
    df['is_holiday_today'] = is_holiday_today

    # 5. Summary & Save (RELATIVE PATH)
    print("\n--------------------------------------------------")
    match_count = df['is_trending'].sum()
    total_rows = len(df)
    
    print(f"Total rows processed: {total_rows}")
    print(f"Rows matching keywords: {match_count}")
    print(f"Percentage matches: {round((match_count/total_rows)*100, 2)}%")
    print(f"Today is Holiday: {'Yes' if is_holiday_today else 'No'}")
    print(f"--------------------------------------------------")
    
    # Save to Relative Path
    df.to_csv('../data/ad_data_enriched.csv', index=False)
    print(f"\n✅ Enriched dataset saved to: ../data/ad_data_enriched.csv")
    
    # Optional: Print the new columns to verify
    print("\nPreview of new columns:")
    print(df[['is_trending', 'tech_savvy_segment', 'is_holiday_today']].head())
    
    return df.head()

if __name__ == "__main__":
    enrich_dataset()