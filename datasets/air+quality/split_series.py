import pandas as pd
import os

# Define file paths
input_file = 'demo_multi_series_exog.csv'
output_dir = os.path.dirname(os.path.abspath(__file__))

# Read the CSV file
print(f"Reading {input_file}...")
df = pd.read_csv(input_file)

# Get unique series IDs
unique_ids = df['series_id'].unique()
print(f"Found {len(unique_ids)} unique series IDs: {sorted(unique_ids)}")

# Split and save each series to a separate file
for series_id in unique_ids:
    # Filter data for this series
    series_df = df[df['series_id'] == series_id]

    # Create output filename
    output_file = os.path.join(output_dir, f'demo_single_{series_id}.csv')

    # Save to CSV with header
    series_df.to_csv(output_file, index=False)
    print(f"Created {os.path.basename(output_file)} with {len(series_df)} rows")

print("\nAll files created successfully!")
