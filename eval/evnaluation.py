import pandas as pd
from sklearn.metrics import accuracy_score, classification_report



df = pd.read_csv('mixed_results - Sheet1.csv')

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)


# replace the values to fit the second dataset.
mapping = {
    'typed': 'safe_for_extraction',
    'mixed': 'high_hallucination_risk',
    'handwritten' : 'high_hallucination_risk'
}
df['classification_gemini'] = df['classification_gemini'].map(mapping)
# df['is_correct'] = df['category'] == df['expected output']

df_clean = df.dropna(subset=['classification_gemini'])
df_clean = df_clean.reset_index(drop=True)
print(df_clean[['classification_gemini', 'classification_idan']])
print(len(df), len(df_clean))

accuracy_idan = accuracy_score(df_clean['expected output'], df_clean['classification_idan'])
accuracy_gemini = accuracy_score(df_clean['expected output'], df_clean['classification_gemini'])
print(f"Precision of Idan's model %: {accuracy_idan}")
print(f"Precision of Gemini %: {accuracy_gemini}")
report_idan = classification_report(df_clean['expected output'], df_clean['classification_idan'])
report_gemini = classification_report(df_clean['expected output'], df_clean['classification_gemini'])

print(print("--- Idan's model Report---"))
print(report_idan)

print(print("--- Gemini's model Report---"))
print(report_gemini)

