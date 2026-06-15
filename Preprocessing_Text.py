import pandas as pd
import emoji
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

file_path = "datasets/"

df = pd.read_excel(f'{file_path}BERT_combined_dataset.xlsx')

print(df.head())
print(df.dtypes)

# Cleanup Data and Preprocessing

# Removing duplicates and data with no Comment
df.drop_duplicates(subset=['text'])

# Case Folding
df['text'] = df['text'].str.lower()

# Remove emojis
df['text'] = df['text'].apply(lambda x: emoji.demojize(x, delimiters=("|", "| ")))

# Remove/modify unneeded text
df['text'] = df['text'].str.replace(r'http\S+', '', regex=True) # http

df['text'] = df['text'].str.replace(r'@\w+', '', regex=True) # mentions

df['text'] = df['text'].str.replace(r'r/\w+', '', regex=True) # subreddit

df['text'] = df['text'].str.replace(r'\d+:\d\d', '', regex=True) # timestamp

df['text'] = df['text'].replace(r'(?<=\d)[.,](?=\d)', 'DECIMAL', regex=True) # ubah rating agar tidak terhapus

df['text'] = df['text'].replace(r'(?<=\d)/(?!\s)(?=\d)', 'SLASH', regex=True) # ubah slash agar tidak terhapus

df['text'] = df['text'].str.replace(r'[^\w\s!?|]', '', regex=True) # gak relevan

df['text'] = df['text'].str.replace(r'[\s+]', ' ', regex=True).str.strip() # whitespace

df['text'] = df['text'].replace('DECIMAL', '.', regex=True) # ubah kembali decimalnya

df['text'] = df['text'].replace('SLASH', '/', regex=True) # ubah kembali decimalnya

# Tokenization, Stopword Removal, and Lemmatization
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    # tokenize teks
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))

    # keep negations
    stop_words = stop_words - {'not', 'no', 'never'}

    # remove stopwords first
    tokens = [w for w in tokens if w not in stop_words]

    # berikan tag part-of-speech kepada token
    tagged = pos_tag(tokens)
    
    # mengubah kata menjadi kata dasar
    lemmas = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged
    ]
    
    return ' '.join(lemmas)

df['text'] = df['text'].apply(lemmatize_text)

print(df.head())
print(df.dtypes)

df.to_excel(f"{file_path}BERT_manual_combined_dataset_preprocessed.xlsx", index=False)