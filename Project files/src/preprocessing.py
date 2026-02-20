import pandas as pd
import numpy as np
import os

DATA_PATH = "../data/raw/startup data.csv"
PROCESSED_PATH = "../data/processed/"
os.makedirs(PROCESSED_PATH, exist_ok=True)


def preprocess_data():

    data = pd.read_csv(DATA_PATH)

    print("Original Shape:", data.shape)

    # ---------------------------------
    # 1. Fix State Columns
    # ---------------------------------
    if 'state_code.1' in data.columns:
        print("State columns equal:",
              data['state_code'].equals(data['state_code.1']))

        # Drop duplicate column
        data = data.drop(columns=['state_code.1'])

    # ---------------------------------
    # 2. Reduce State Categories
    # ---------------------------------
    top_states = ['CA', 'NY', 'MA', 'TX', 'WA']

    data['state_reduced'] = data['state_code'].apply(
        lambda x: x if x in top_states else 'other'
    )

    # ---------------------------------
    # 3. Convert Date Columns
    # ---------------------------------
    date_cols = [
        'founded_at',
        'closed_at',
        'first_funding_at',
        'last_funding_at'
    ]

    for col in date_cols:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')

    # Create founded_year
    data['founded_year'] = data['founded_at'].dt.year

    # Startup age (till 2023)
    data['startup_age'] = 2023 - data['founded_year']

    # ---------------------------------
    # 4. Clean Funding Column
    # ---------------------------------
    data['funding_total_usd'] = pd.to_numeric(
        data['funding_total_usd'], errors='coerce'
    )

    # ---------------------------------
    # 5. Convert Target Variable
    # ---------------------------------
    data['status'] = data['status'].map({
        'acquired': 1,
        'closed': 0
    })

    # ---------------------------------
    # 6. Drop Irrelevant Columns
    # ---------------------------------
    drop_cols = [
        'category_code',
        'is_software',
        'is_web',
        'is_mobile',
        'is_enterprise',
        'is_advertising',
        'is_gamesvideo',
        'is_ecommerce',
        'is_biotech',
        'is_consulting',
        'is_othercategory',
        'city',
        'labels',
        'zip_code',
        'object_id',
        'name'
    ]

    data = data.drop(columns=[c for c in drop_cols if c in data.columns])

    # ---------------------------------
    # 7. Handle Missing Values
    # ---------------------------------
    data = data.fillna(0)

    print("Final Shape After Preprocessing:", data.shape)

    # ---------------------------------
    # 8. Save Processed Data
    # ---------------------------------
    data.to_csv(PROCESSED_PATH + "startup_processed.csv", index=False)

    print("Processed file saved successfully.")

    return data


if __name__ == "__main__":
    preprocess_data()