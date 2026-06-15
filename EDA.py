import pandas as pd
import matplotlib.pyplot as plt
import os
from wordcloud import WordCloud, STOPWORDS

# Menggunakan Vader dan manual label
df = pd.read_excel("datasets/BERT_manual_combined_dataset_preprocessed.xlsx")

df = df.drop_duplicates(subset=['text'])
df = df.dropna(subset=['text', 'sentiment'])

df_manual = df[(df['source'] != 'pseudo') & (df['sentiment'] != "")]
df_pseudo = df[(df['source'] == 'pseudo') & (df['sentiment'] != "")]

print("\nOverall Label Distribution: ")
print(df['sentiment'].value_counts())
print("\nManual Label Distribution: ")
print(df_manual['sentiment'].value_counts())
print("\nPseudo (BERT) Label Distribution: ")
print(df_pseudo['sentiment'].value_counts())

all_label = df['sentiment'].value_counts()
manual_label = df_manual['sentiment'].value_counts()
pseudo_label = df_pseudo['sentiment'].value_counts()

# Membuat folder untuk grafik
filepath = "figures/Dataset/ManualBERT/"
if not os.path.exists(filepath):
  os.makedirs(filepath)

# Bar chart
label = ['Negative', 'Neutral', 'Positive']
count = [all_label.loc['negative'], all_label.loc['neutral'], all_label.loc['positive']]
color = ['tab:red', 'tab:orange', 'tab:blue']
plt.figure(figsize=(7, 6))
plt.bar(label, count, label=label, color=color)
plt.title("Distribusi Keseluruhan Label Dataset")
plt.savefig(f"{filepath}DistAllLabel.png")

count = [manual_label.loc['negative'], manual_label.loc['neutral'], manual_label.loc['positive']]
plt.figure(figsize=(7, 6))
plt.bar(label, count, label=label, color=color)
plt.title("Distribusi Label Manual Dataset")
plt.savefig(f"{filepath}DistManualLabel.png")

count = [pseudo_label.loc['negative'], pseudo_label.loc['neutral'], pseudo_label.loc['positive']]
plt.figure(figsize=(7, 6))
plt.bar(label, count, label=label, color=color)
plt.title("Distribusi Pseudo Label Dataset")
plt.savefig(f"{filepath}DistPseudoLabel.png")

# Bikin WordCloud
import re

text = ' '.join(df['text'].astype(str).tolist())

text = re.sub(r'[^A-Za-z\s]', '', text)

text = text.lower()

stopwords = set(STOPWORDS)
text = ' '.join(word for word in text.split() if word not in stopwords)

from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(width=800, height=400, background_color='white', random_state=157).generate(text)

plt.figure(figsize=(10, 5), dpi=500)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  
plt.title("Word Cloud Komentar BFV")
plt.savefig(f"{filepath}WordCloud.png")
