import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv(r'C:\Users\Ajay\cyber_threat\data\raw\SMSSpamCollection', 
                   sep='\t', 
                   header=None, 
                   names=['label', 'text'])

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)

vectorizer = TfidfVectorizer(\
    stop_words='english', 
    max_df= 3000
)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']


model = LogisticRegression(max_iter=1000)
model.fit(X, y)

text_threat_score = model.predict_proba(X)[:, 1]

text_score_df = pd.DataFrame({
    'text': df['text'],
    'text_threat_score': text_threat_score,
    'true_label': y
})

text_score_df.to_csv('data/processed/text_threat_scores.csv', index=False)

print("=== NLP TEXT THREAT SCORE SAMPLE ===")
print(text_score_df.head())
print("\nScore statistics:")
print(text_score_df["text_threat_score"].describe())