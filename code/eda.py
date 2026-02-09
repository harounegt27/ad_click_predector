import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for better looking plots
sns.set(style="whitegrid")

def run_eda():
    # 1. Load the Enriched Data
    print("Loading enriched dataset...")
    try:
        df = pd.read_csv('data/ad_data_enriched.csv')
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print("Error: 'ad_data_enriched.csv' not found. Run scraping.py first.")
        return

    print("\n--- 1. Dataset Overview ---")
    print(f"Total Rows: {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")
    print("\nColumn Data Types:")
    print(df.dtypes)

    # 2. Missing Values Analysis (Crucial for Week 2 Pipeline)
    print("\n--- 2. Missing Values Analysis ---")
    missing = df.isnull().sum()
    print(missing)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.savefig('data/01_missing_values.png')
    plt.show()

    # 3. Target Variable Distribution (Click vs No Click)
    print("\n--- 3. Target Distribution ---")
    plt.figure(figsize=(6, 4))
    sns.countplot(x='click', data=df, palette='viridis')
    plt.title('Target Distribution (Click vs No Click)')
    plt.savefig('data/02_target_distribution.png')
    plt.show()

    # 4. Device Type Analysis
    print("\n--- 4. Click Behavior by Device ---")
    plt.figure(figsize=(8, 5))
    sns.countplot(x='device_type', hue='click', data=df, palette='coolwarm')
    plt.title('Clicks by Device Type')
    plt.savefig('data/03_device_analysis.png')
    plt.show()

    # 5. NEW FEATURE: Tech-Savvy Segment Analysis
    # Check if your logic worked: Do High Tech-Savvy users click more?
    print("\n--- 5. Analysis: Tech-Savvy Segment ---")
    plt.figure(figsize=(8, 5))
    sns.countplot(x='tech_savvy_segment', hue='click', data=df, order=['Low', 'Medium', 'High'], palette='magma')
    plt.title('Clicks by Tech-Savvy Segment (New Feature)')
    plt.savefig('data/04_tech_savvy_segment.png')
    plt.show()

    # 6. NEW FEATURE: Trending Enrichment Analysis
    # Check if users viewing trending topics click more
    print("\n--- 6. Analysis: Trending Topic Influence ---")
    plt.figure(figsize=(6, 4))
    sns.countplot(x='is_trending', hue='click', data=df, palette='Set2')
    plt.title('Clicks vs Trending Keywords (New Feature)')
    plt.xticks([0, 1], ['Not Trending', 'Trending'])
    plt.savefig('data/05_trending_influence.png')
    plt.show()

    # 7. Numerical Analysis: Age
    print("\n--- 7. Age Distribution by Click Status ---")
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x='age', hue='click', kde=True, element='step', stat='density', common_norm=False, palette='pastel')
    plt.title('Age Distribution: Clickers vs Non-Clickers')
    plt.savefig('data/06_age_distribution.png')
    plt.show()

    # 8. Correlation Matrix
    print("\n--- 8. Correlation Matrix ---")
    # We only select numeric columns for correlation
    # Dropping id and full_name as they are not predictive
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix (Numeric Features)')
    plt.savefig('data/07_correlation_matrix.png')
    plt.show()

    print("\n✅ EDA Complete. All charts saved to 'data/' folder.")

if __name__ == "__main__":
    run_eda()