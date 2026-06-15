# Library
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (confusion_matrix, classification_report)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import os


# Configuration
random_state_val = 157
data_input = 'text'
data_output = 'sentiment'
algo = 'NB'

print("Random state seed:", random_state_val)
print("Using Algorithm:", algo)
print("Using BERT: YES")


# Load Dataset
df = pd.read_excel("datasets/BERT_manual_combined_dataset_preprocessed.xlsx")


# Delete duplicates and empty input (Preprocessing dapat menyebabkannya)
df = df.drop_duplicates(subset=['text'])
df = df.dropna(subset=['text', 'sentiment'])

# Separate Manual Labeled and Pseudo Labeled
df_manual = df[(df['source'] == 'manual') & (df[data_output] != "")].copy()
df_pseudo = df[(df['source'] == 'pseudo') & (df[data_output] != "")].copy()

print("Total Row:", len(df))
print(df[data_output].value_counts())


# Splitting data
X_man_train_text, X_test_text, y_man_train, y_test = train_test_split(df_manual[data_input], df_manual[data_output], test_size=0.2, random_state=random_state_val, stratify=df_manual[data_output])


# Preprocessing for Model
# Encoder and Vectorizer
encoder = LabelEncoder()
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 3))

# Encode label
y_man_train = encoder.fit_transform(y_man_train)
y_pse = encoder.transform(df_pseudo[data_output])
y_test = encoder.transform(y_test)

# Combine text to one for training
X_train = pd.concat([X_man_train_text, df_pseudo[data_input]])

# Vectorizing
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test_text)

# Combining label to one for training
y_train = np.concatenate([
    y_man_train,
    y_pse
])


# Build Model
model_base = MultinomialNB()

param_dist = {
    "alpha":[1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 
            0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 
            0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
}

model_search = GridSearchCV(
    estimator=model_base, 
    param_grid=param_dist,
    cv = 5,
    scoring='average_precision',
    n_jobs=-1,
    refit=True,
    verbose=2,
    return_train_score=True
)


# Model Training
model_search.fit(X_train, y_train)

cv_results = pd.DataFrame(model_search.cv_results_)
filepath = f"results/Model/{algo}"
if not os.path.exists(filepath):
    os.makedirs(filepath)
cv_results.to_excel(f"{filepath}/CV_Results_{algo}.xlsx", index=False)

print("Best Estimator:", model_search.best_estimator_)
print("Best Parameter:", model_search.best_params_)
print("Best Score:", model_search.best_score_)
print("Class Log Prior:", model_search.best_estimator_.class_log_prior_)

model = model_search.best_estimator_


# Evaluasi Model dengan set
filepath = f"figures/Model/{algo}"
if not os.path.exists(filepath):
    os.makedirs(filepath)

dataset_name = "Training Set"

filename_suffix = ""

if dataset_name == "Training Set":
    filename_suffix = "train"

# Prediksi kelas
y_pred = model.predict(X_train)

# Confusion matrix (TN, FP, FN, TP)
cm = confusion_matrix(y_train, y_pred)

# Output hasil
print(f"\n=== METRIK EVALUASI {dataset_name.upper()} ===")
print(classification_report(y_train, y_pred))

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7,7))
ax = sns.heatmap(cm, cmap='Blues', annot=True, fmt='d')
# Berikan label sumbu X_train dan nilainya. 
ax.set_xlabel("Predicted", fontsize=14, labelpad=10)
ax.xaxis.set_ticklabels(encoder.classes_)  
# Berikan label sumbu y_test dan nilainya
ax.set_ylabel("Actual", fontsize=14, labelpad=20)
ax.yaxis.set_ticklabels(encoder.classes_)
# Berikan nama ke plot
ax.set_title(f"Confusion Matrix for {dataset_name}", fontsize=14, pad=20)
# Simpan Confusion Matrix
plt.savefig(f"{filepath}/CM_{filename_suffix}.png")


# Metrik evaluasi dengan test set
dataset_name = "Test Set"

if dataset_name == "Test Set":
    filename_suffix = "test"

# Prediksi kelas
y_pred = model.predict(X_test)

# Confusion matrix (TN, FP, FN, TP)
cm = confusion_matrix(y_test, y_pred)

# Output hasil
print(f"\n=== METRIK EVALUASI {dataset_name.upper()} ===")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7,7))
ax = sns.heatmap(cm, cmap='Blues', annot=True, fmt='d')
# Berikan label sumbu X_train dan nilainya. 
ax.set_xlabel("Predicted", fontsize=14, labelpad=10)
ax.xaxis.set_ticklabels(encoder.classes_)  
# Berikan label sumbu y_test dan nilainya
ax.set_ylabel("Actual", fontsize=14, labelpad=20)
ax.yaxis.set_ticklabels(encoder.classes_)
# Berikan nama ke plot
ax.set_title(f"Confusion Matrix for {dataset_name}", fontsize=14, pad=20)
# Simpan Confusion Matrix
plt.savefig(f"{filepath}/CM_{filename_suffix}.png")


# Display sample output
classif_result_samples = pd.DataFrame()
classif_result_samples["Text"] = X_test_text[:20]
classif_result_samples["Label Aktual"] = encoder.inverse_transform(y_test[:20])
classif_result_samples["Label Prediksi"] = encoder.inverse_transform(y_pred[:20])

filepath = f"results/Model/{algo}"
classif_result_samples.to_excel(f"{filepath}/Classification_Result_Samples.xlsx", index=False)

# Saving
filepath = "algorithmBinaries"
if not os.path.exists(filepath):
    os.makedirs(filepath)


pickle.dump(vectorizer, open("algorithmBinaries/vectorizer.pkl", 'wb'))
pickle.dump(model, open(f"{filepath}/{algo}.pkl", 'wb'))