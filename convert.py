import pandas as pd
from sklearn.cluster import KMeans
import pickle

# Membaca data dari file CSV
data = pd.read_csv('data_kecamatan.csv')

# Memilih fitur untuk clustering
X = data[['Jumlah_LakiLaki', 'Jumlah_Perempuan', 'Jumlah_LakiPerempuan']]

# Menentukan jumlah klaster yang sesuai dengan jumlah sampel
jumlah_klaster = min(3, data.shape[0])

# Melakukan KMeans clustering
kmeans = KMeans(n_clusters=jumlah_klaster, random_state=42)
kmeans.fit(X)

# Simpan model ke dalam file dengan pickle
with open('kmeans_model.pkl', 'wb') as model_file:
    pickle.dump(kmeans, model_file)
