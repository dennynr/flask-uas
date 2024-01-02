# Mengimpor modul yang dibutuhkan
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# Memuat data iris
iris = datasets.load_iris()
X = iris.data
y = iris.target
print(iris)
# Menentukan jumlah klaster
n_clusters = 3

# Melakukan klastering dengan K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(X)
labels = kmeans.labels_

# Menghitung rata-rata koefisien siluet untuk semua sampel
silhouette_avg = silhouette_score(X, labels)
print("Untuk n_clusters =", n_clusters,
      "Rata-rata koefisien siluet adalah :", silhouette_avg)

# Menghitung koefisien siluet untuk setiap sampel
sample_silhouette_values = silhouette_samples(X, labels)
print("Nilai koefisien siluet untuk setiap sampel adalah :", sample_silhouette_values)