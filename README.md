# data-cleaning-and-preprocessing
task -1
# ============================================
# 📊 CUSTOMER SEGMENTATION USING K-MEANS
# ============================================

# 🔹 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 🔹 2. Create (or load) dataset
# If you have a CSV file, replace this part with:
# df = pd.read_csv('Mall_Customers.csv')

np.random.seed(42)
n = 200
df = pd.DataFrame({
    'CustomerID': np.arange(1, n+1),
    'Gender': np.random.choice(['Male', 'Female'], n),
    'Age': np.random.randint(18, 70, n),
    'Annual_Income_k': np.random.randint(15, 150, n),
    'Spending_Score': np.random.randint(1, 100, n)
})

# Introduce a few missing values for realism
df.loc[np.random.choice(df.index, 3, replace=False), 'Annual_Income_k'] = np.nan
df.loc[np.random.choice(df.index, 3, replace=False), 'Age'] = np.nan

# 🔹 3. Data Cleaning
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Annual_Income_k'].fillna(df['Annual_Income_k'].median(), inplace=True)
df['Gender_Code'] = (df['Gender'] == 'Male').astype(int)

# 🔹 4. Feature Selection and Scaling
features = ['Age', 'Annual_Income_k', 'Spending_Score']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 5. Find optimal number of clusters (k)
sil_scores = {}
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil_scores[k] = silhouette_score(X_scaled, labels)

best_k = max(sil_scores, key=sil_scores.get)
print(f"✅ Best k found: {best_k} (Silhouette Score = {sil_scores[best_k]:.3f})")

# 🔹 6. Final Clustering
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled) + 1

# 🔹 7. Cluster Profiles
profile = df.groupby('Cluster').agg({
    'CustomerID': 'count',
    'Age': 'mean',
    'Annual_Income_k': 'mean',
    'Spending_Score': 'mean',
    'Gender_Code': 'mean'
}).reset_index()

profile.rename(columns={
    'CustomerID': 'Count',
    'Gender_Code': 'Male_%'
}, inplace=True)

profile['Male_%'] = (profile['Male_%'] * 100).round(1)
profile[['Age', 'Annual_Income_k', 'Spending_Score']] = profile[['Age', 'Annual_Income_k', 'Spending_Score']].round(1)

print("\n📈 Cluster Profile Summary:\n", profile)

# 🔹 8. Visualizations
plt.figure(figsize=(6, 4))
plt.plot(list(sil_scores.keys()), list(sil_scores.values()), marker='o')
plt.title('Silhouette Scores for Different k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.show()

plt.figure(figsize=(7, 5))
for c in sorted(df['Cluster'].unique()):
    subset = df[df['Cluster'] == c]
    plt.scatter(subset['Annual_Income_k'], subset['Spending_Score'], label=f'Cluster {c}')
plt.title('Customer Segments (Income vs Spending)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend()
plt.show()

# 🔹 9. Save results
df.to_csv('customer_segmentation_labeled.csv', index=False)
print("\n💾 Saved labeled dataset as 'customer_segmentation_labeled.csv'")
