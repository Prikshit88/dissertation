import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# 1. Data Loading
path = r"C:\Users\ptush\OneDrive\Documents\Dessertation\Outpatients.csv"
if not os.path.exists(path):
    print(f"Error: The file '{path}' was not found.")
    sys.exit(1)

dtype_dict = {f"ICD9_DGNS_CD_{i+1}": str for i in range(25)}
dtype_dict.update({f"HCPCS_CD_{i+1}": str for i in range(45)})

try:
    data_1 = pd.read_csv(path, low_memory=False, dtype=dtype_dict)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    sys.exit(1)

# 2. Data Cleaning
diag_code_cols = [f"ICD9_DGNS_CD_{i+1}" for i in range(25)]
hcpcs_code_cols = [f"HCPCS_CD_{i+1}" for i in range(45)]

for col_name in diag_code_cols:
    if col_name in data_1.columns:
        data_1[col_name] = data_1[col_name].fillna('').astype(str).str.strip().str.pad(5, side='left', fillchar='0')

for col_name in hcpcs_code_cols:
    if col_name in data_1.columns:
        data_1[col_name] = data_1[col_name].fillna('').astype(str).str.strip().str.pad(5, side='left', fillchar='0')

# 3. Diagnosis Code Categorization
diag_categories = {
    'Infection_&_Parasitic': range(0, 140), 'Neoplasm': range(140, 240),
    'Endocrine_Nutritional_Immunity': range(240, 280), 'Blood': range(280, 290),
    'Mental_&_Behavioral': range(290, 320), 'Nervous': range(320, 390),
    'Circulatory': range(390, 460), 'Respiratory': range(460, 520),
    'Digestive': range(520, 580), 'Genitourinary': range(580, 630),
    'Complications_Pregnancy_Childbirth': range(630, 680), 'Skin': range(680, 710),
    'Musculoskeletal': range(710, 740), 'Congenital_Anomaly': range(740, 760),
    'Perinatal_Conditions': range(760, 780), 'Symptoms_&_Ill-defined': range(780, 800),
    'Injury_&_Poisining': range(800, 1000),
    'Supplementary_V_Codes': 'V', 'Supplementary_E_Codes': 'E'
}

def map_diag_code(code):
    if not code or not isinstance(code, str) or code.strip() in ['00000', '']:
        return None
    if code.startswith('V'):
        return 'Supplementary_V_Codes'
    if code.startswith('E'):
        return 'Supplementary_E_Codes'
    try:
        code_num = int(code[:3])
        for category, code_range in diag_categories.items():
            if isinstance(code_range, range) and code_num in code_range:
                return category
    except (ValueError, TypeError):
        return None
    return 'Other'

for i in range(25):
    col_name = f"ICD9_DGNS_CD_{i+1}"
    diag_col = f"Diag_Cat_{i+1}"
    if col_name in data_1.columns:
        data_1[diag_col] = data_1[col_name].apply(map_diag_code)

# 4. Procedure Code Categorization
def map_hcpcs_code(code):
    if not code or not isinstance(code, str) or code.strip() in ['00000', '']:
        return None
    if code.startswith('0'): return 'Anesthesia'
    if any(code.startswith(s) for s in ['1', '2', '3', '4', '5', '6']): return 'Surgery'
    if code.startswith('7'): return 'Radiology'
    if code.startswith('8'): return 'Pathology_Procedure'
    if any(code.startswith(s) for s in ['992', '993', '994']): return 'E&M'
    if code.startswith('A0'): return 'Ambulance'
    if 'A42' <= code[:3] <= 'A80': return 'Medical_Supplies'
    if code.startswith('A9'): return 'Investigational'
    if 'J0' <= code[:2] <= 'J8': return 'Drugs_other_than_oral'
    if code.startswith('J9'): return 'Chemotherapy'
    if code.startswith('G'): return 'Temp_Codes_Procedures_Services'
    if code.startswith('Q'): return 'Temp_Codes'
    return 'Other_Services'

for i in range(45):
    col_name = f"HCPCS_CD_{i+1}"
    proc_col = f"Proc_Cat_{i+1}"
    if col_name in data_1.columns:
        data_1[proc_col] = data_1[col_name].apply(map_hcpcs_code)

# 5. Feature Engineering
diag_cat_cols = [f'Diag_Cat_{i+1}' for i in range(25) if f'Diag_Cat_{i+1}' in data_1.columns]
proc_cat_cols = [f'Proc_Cat_{i+1}' for i in range(45) if f'Proc_Cat_{i+1}' in data_1.columns]

diag_melt = data_1[['DESYNPUF_ID'] + diag_cat_cols].melt(id_vars='DESYNPUF_ID', value_name='Diag_Category').dropna()
proc_melt = data_1[['DESYNPUF_ID'] + proc_cat_cols].melt(id_vars='DESYNPUF_ID', value_name='Proc_Category').dropna()

patient_diag = diag_melt.groupby('DESYNPUF_ID')['Diag_Category'].unique().apply(list).reset_index()
patient_proc = proc_melt.groupby('DESYNPUF_ID')['Proc_Category'].unique().apply(list).reset_index()

patient_data = pd.merge(patient_diag, patient_proc, on='DESYNPUF_ID', how='outer')
patient_data = patient_data.fillna('').applymap(lambda x: [] if x == '' else x)

mlb_diag = MultiLabelBinarizer()
diag_features = pd.DataFrame(
    mlb_diag.fit_transform(patient_data['Diag_Category']),
    columns=[f"Diag_{c}" for c in mlb_diag.classes_],
    index=patient_data.index
)

mlb_proc = MultiLabelBinarizer()
proc_features = pd.DataFrame(
    mlb_proc.fit_transform(patient_data['Proc_Category']),
    columns=[f"Proc_{c}" for c in mlb_proc.classes_],
    index=patient_data.index
)

features = pd.concat([diag_features, proc_features], axis=1)
features['DESYNPUF_ID'] = patient_data['DESYNPUF_ID']

# 6. Silhouette Analysis (Modified to match your Figure 3.2)
print("\nRunning Silhouette Analysis for K-means...")

# Define the cluster range based on your figure
cluster_range = range(60, 86, 5)  # From 60 to 85 in steps of 5
silhouette_scores = []

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features.drop('DESYNPUF_ID', axis=1))
    silhouette_avg = silhouette_score(features.drop('DESYNPUF_ID', axis=1), cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"n_clusters = {n_clusters}: Silhouette Score = {silhouette_avg:.5f}")

# Find the best number of clusters
best_n = cluster_range[np.argmax(silhouette_scores)]
best_score = max(silhouette_scores)

# Create the plot matching your Figure 3.2
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of clusters', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('Silhouette Analysis for K-means Clustering', fontsize=14, pad=20)

# Set the y-axis limits to match your figure
plt.ylim(0.24, 0.28)

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.7)

# Highlight the best score
plt.axvline(x=best_n, color='r', linestyle='--', linewidth=2)
plt.scatter(best_n, best_score, color='red', s=200, marker='*', 
           label=f'Best: {best_n} clusters\nScore: {best_score:.5f}')

# Add the text annotation like in your figure
plt.text(70, 0.265, f'Best number of clusters: {best_n}\nBest silhouette score: {best_score:.5f}', 
         bbox=dict(facecolor='white', alpha=0.8))

plt.legend()
plt.show()

print(f"\nBest number of clusters: {best_n}")
print(f"Best silhouette score: {best_score:.5f}")