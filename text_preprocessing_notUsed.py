import pandas as pd
import re
from spacy.lang.en import English
from transformers import AutoTokenizer

# File paths
input_csv = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/cleaned_dblp.csv"
output_csv = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/preprocessed_dblp.csv"

print("Loading cleaned dataset...")
df = pd.read_csv(input_csv)

# Load tokenizer and spaCy tokenizer
nlp = English()
tokenizer = AutoTokenizer.from_pretrained("allenai/specter")

def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)

    doc = nlp(text)
    tokens = [token.text for token in doc]
    text = ' '.join(tokens)

    token_ids = tokenizer.encode(text, truncation=True, max_length=512, add_special_tokens=True)
    return tokenizer.decode(token_ids, skip_special_tokens=True)

# Apply preprocessing
print("Preprocessing 'title' and 'indexed_abstract'...")
df['title'] = df['title'].apply(preprocess_text)
df['indexed_abstract'] = df['indexed_abstract'].apply(preprocess_text)

# Save preprocessed data
df.to_csv(output_csv, index=False)
print(f"Preprocessed dataset saved to: {output_csv}")
print(f"Total records processed: {df.shape[0]}")
