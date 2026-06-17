import pandas as pd

# Read the original CSV file
df = pd.read_csv('arrhythmia.csv')

# Select first 10 columns and the last column (binaryClass)
first_10_cols = df.columns[:10].tolist()
last_col = [df.columns[-1]]

# Combine the column names
selected_columns = first_10_cols + last_col

# Create new dataframe with selected columns
df_transformed = df[selected_columns]

# Save to new CSV file
df_transformed.to_csv('arrhythmia_testing.csv', index=False)

print(f"Transformation complete!")
print(f"Original file had {len(df.columns)} columns and {len(df)} rows")
print(f"New file has {len(df_transformed.columns)} columns and {len(df_transformed)} rows")
print(f"\nSelected columns: {', '.join(selected_columns)}")
