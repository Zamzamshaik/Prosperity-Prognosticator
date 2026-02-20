import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------
# Paths
# ------------------------------
DATA_PATH = "../data/raw/startup data.csv"
FIGURE_PATH = "../reports/figures/"
os.makedirs(FIGURE_PATH, exist_ok=True)


# ------------------------------
# Load & Clean Data
# ------------------------------
def load_data():

    data = pd.read_csv(DATA_PATH)

    print("Initial Dataset Shape:", data.shape)

    # Drop unnecessary columns safely
    cols_to_drop = [
        'Unnamed: 0', 'Unnamed: 6', 'state_code.1',
        'zip_code', 'id', 'object_id', 'name', 'labels'
    ]

    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])

    # Convert date column
    data['founded_at'] = pd.to_datetime(data['founded_at'], errors='coerce')

    # Create founded_year safely
    data['founded_year'] = data['founded_at'].dt.year

    # Drop rows where year missing
    data = data.dropna(subset=['founded_year'])
    data['founded_year'] = data['founded_year'].astype(int)

    # Clean funding column
    data['funding_total_usd'] = pd.to_numeric(
        data['funding_total_usd'], errors='coerce'
    )

    # Convert target variable
    data['status'] = data['status'].map({
        'acquired': 1,
        'closed': 0
    })

    # Create category column
    data['category'] = data['category_code']

    print("Final Dataset Shape:", data.shape)
    print("Columns Available:\n", data.columns)

    return data


# ------------------------------
# State Analysis
# ------------------------------
def state_analysis(data):

    data['State'] = data['state_code'].apply(
        lambda x: x if x in ['CA', 'NY', 'MA', 'TX', 'WA'] else 'other'
    )

    state_count = data['State'].value_counts()

    plt.figure(figsize=(6,6))
    plt.pie(state_count, labels=state_count.index, autopct='%1.1f%%')
    plt.title("Distribution of Startups by State")
    plt.savefig(FIGURE_PATH + "state_distribution.png")
    plt.close()


# ------------------------------
# Category Analysis
# ------------------------------
def category_analysis(data):

    category_count = data['category'].value_counts()

    plt.figure(figsize=(8,8))
    plt.pie(category_count, labels=category_count.index, autopct='%1.1f%%')
    plt.title("Distribution of Startup Categories")
    plt.savefig(FIGURE_PATH + "category_distribution.png")
    plt.close()


# ------------------------------
# Status Distribution
# ------------------------------
def status_distribution(data):

    prop_df = data.groupby('status').size().reset_index(name='counts')
    prop_df['proportions'] = prop_df['counts'] / prop_df['counts'].sum()

    plt.figure(figsize=(6,4))
    sns.barplot(data=prop_df, x='status', y='proportions')
    plt.title("Distribution of Startup Status")
    plt.savefig(FIGURE_PATH + "status_distribution.png")
    plt.close()


# ------------------------------
# State vs Status
# ------------------------------
def state_vs_status(data):

    prop_df = data.groupby(['State','status']).size().reset_index(name='counts')
    prop_df['proportions'] = prop_df.groupby('State')['counts'].transform(lambda x: x/x.sum())

    plt.figure(figsize=(8,6))
    sns.barplot(data=prop_df, x='State', y='proportions', hue='status')
    plt.title("State vs Status")
    plt.savefig(FIGURE_PATH + "state_vs_status.png")
    plt.close()


# ------------------------------
# Category vs Status
# ------------------------------
def category_vs_status(data):

    prop_df = data.groupby(['category','status']).size().reset_index(name='counts')
    prop_df['proportions'] = prop_df.groupby('category')['counts'].transform(lambda x: x/x.sum())

    plt.figure(figsize=(14,6))
    sns.barplot(data=prop_df, x='category', y='proportions', hue='status')
    plt.xticks(rotation=45)
    plt.title("Category vs Status")
    plt.savefig(FIGURE_PATH + "category_vs_status.png")
    plt.close()


# ------------------------------
# Category vs Founded Year
# ------------------------------
def category_vs_year(data):

    # Ensure founded_year exists
    if 'founded_year' not in data.columns:
        data['founded_at'] = pd.to_datetime(data['founded_at'], errors='coerce')
        data['founded_year'] = data['founded_at'].dt.year

    data = data.dropna(subset=['founded_year'])

    cat_year = pd.crosstab(
        index=data['founded_year'],
        columns=data['category']
    )

    plt.figure(figsize=(14,6))
    sns.lineplot(data=cat_year)
    plt.title("Category-wise Evolution Over Years")
    plt.savefig(FIGURE_PATH + "category_vs_year.png")
    plt.close()


# ------------------------------
# Founded Year vs Funding
# ------------------------------
def year_vs_funding(data):

    if 'founded_year' not in data.columns:
        data['founded_at'] = pd.to_datetime(data['founded_at'], errors='coerce')
        data['founded_year'] = data['founded_at'].dt.year

    data = data.dropna(subset=['founded_year'])

    # Remove extreme outliers for readability
    data = data[data['funding_total_usd'] < 1e9]

    plt.figure(figsize=(14,6))
    sns.boxplot(data=data, x="founded_year", y="funding_total_usd")
    plt.xticks(rotation=90)
    plt.title("Founded Year vs Total Funding")
    plt.savefig(FIGURE_PATH + "year_vs_funding.png")
    plt.close()


# ------------------------------
# Funding Round Analysis
# ------------------------------
def funding_round_analysis(data):

    funding_cols = [
        "has_VC", "has_angel", "has_roundA",
        "has_roundB", "has_roundC", "has_roundD"
    ]

    d = data[data['status'] == 1]

    melted = pd.melt(d[funding_cols])

    plt.figure(figsize=(10,6))
    sns.countplot(data=melted, x='variable', hue='value')
    plt.xticks(rotation=45)
    plt.title("Funding Indicators in Successful Startups")
    plt.savefig(FIGURE_PATH + "funding_rounds.png")
    plt.close()


# ------------------------------
# Statistical Summary
# ------------------------------
def statistical_analysis(data):
    print("\nStatistical Summary:")
    print(data.describe())


# ------------------------------
# Correlation Heatmap
# ------------------------------
def correlation_plot(data):

    corr = data.select_dtypes(include=['int64','float64']).corr()

    plt.figure(figsize=(14,10))
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title("Correlation Heatmap")
    plt.savefig(FIGURE_PATH + "correlation_heatmap.png")
    plt.close()


# ------------------------------
# Run EDA
# ------------------------------
def run_eda():

    data = load_data()

    state_analysis(data)
    category_analysis(data)
    status_distribution(data)
    state_vs_status(data)
    category_vs_status(data)
    category_vs_year(data)
    year_vs_funding(data)
    funding_round_analysis(data)
    statistical_analysis(data)
    correlation_plot(data)

    print("\nAll EDA plots saved in:", FIGURE_PATH)


if __name__ == "__main__":
    run_eda()