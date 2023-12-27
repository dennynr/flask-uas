import pandas as pd
import pickle

# Membaca data dari file CSV
data = pd.read_csv('data_kecamatan.csv')

# Memilih fitur untuk clustering
X = data[['Jumlah_LakiLaki', 'Jumlah_Perempuan', 'Jumlah_LakiPerempuan']]

# Membaca model dari file pickle
with open('kmeans_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Mendapatkan prediksi klaster
predicted_clusters = loaded_model.predict(X)

# Menambahkan kolom Klaster ke DataFrame
data['Klaster'] = predicted_clusters

# Menampilkan hasil klaster
print(data)
