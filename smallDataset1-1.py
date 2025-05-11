import pandas as pd

# Path to the original dataset
original_path = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/preprocessed_dblp_FIXED.csv"

# Path to save the smaller dataset
output_path = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/small_preprocessed_dblp.csv"

# How many samples to keep
sample_size = 100000  # Change this to your desired size

# Load and sample the dataset
df = pd.read_csv(original_path)
df_sampled = df.sample(n=sample_size, random_state=42)

# Save the sampled dataset
df_sampled.to_csv(output_path, index=False)
print(f"Saved {sample_size} rows to: {output_path}")
