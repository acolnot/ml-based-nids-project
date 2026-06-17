import pandas as pd
import numpy as np
import os

def clean_dataset(input_path, output_path):
    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)
    
    initial_rows = len(df)
    print(f"Initial row count: {initial_rows}")

    # 1. Clean column names
    print("Stripping whitespace from column headers...")
    df.columns = df.columns.str.strip()

    # 2. Handle infinite values and NaNs
    print("Removing infinity and NaN values...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 3. Remove duplicates
    print("Dropping duplicate rows...")
    df.drop_duplicates(inplace=True)

    # 4. Encode the target Label
    print("Encoding labels: BENIGN -> 0, Attack -> 1...")
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

    # Print a summary of the cleanup
    final_rows = len(df)
    dropped_rows = initial_rows - final_rows
    print(f"\n--- Cleanup Summary ---")
    print(f"Rows dropped: {dropped_rows}")
    print(f"Final row count: {final_rows}")
    print(f"Class distribution:\n{df['Label'].value_counts(normalize=True) * 100}")

    # 5. Save the cleaned dataset
    print(f"\nSaving cleaned dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Saved !")

if __name__ == "__main__":
    RAW_FILE = "data/Wednesday-workingHours.pcap_ISCX.csv"
    CLEAN_FILE = "data/Cleaned_Wednesday.csv"
    
    if not os.path.exists(RAW_FILE):
        print(f"ERROR: Place the raw CSV inside the 'data/' folder before running.")
    else:
        clean_dataset(RAW_FILE, CLEAN_FILE)