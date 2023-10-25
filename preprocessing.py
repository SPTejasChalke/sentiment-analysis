import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('data.csv')

# Remove <br> and <br/> from the text in the first column
df['review'] = df['review'].str.replace('<br />', '').str.replace('<br/>', '')

# Replace "positive" with 1 and "negative" with 0 in the second column
df['sentiment'] = df['sentiment'].replace({'positive': 1, 'negative': 0})

# Specify the path for the new CSV file
output_csv_path = 'output_file.csv'

# Save the modified DataFrame to a new CSV file
df.to_csv(output_csv_path, index=False)

print(f"Data has been processed and saved to {output_csv_path}")
