import json
import pickle

# Contoh data untuk disimpan
data = {
    'nama': 'John Doe',
    'umur': 25,
    'pekerjaan': 'Pengembang'
}

# Menggunakan JSON untuk menyimpan data ke file
json_file_path = 'data.json'
with open(json_file_path, 'w') as json_file:
    json.dump(data, json_file)

# Menggunakan Pickle untuk menyimpan data ke file
pickle_file_path = 'data.pkl'
with open(pickle_file_path, 'wb') as pickle_file:
    pickle.dump(data, pickle_file)

# Membaca data dari file JSON
with open(json_file_path, 'r') as json_file:
    json_data = json.load(json_file)
    print("Data dari file JSON:", json_data)

# Membaca data dari file Pickle
with open(pickle_file_path, 'rb') as pickle_file:
    pickle_data = pickle.load(pickle_file)
    print("Data dari file Pickle:", pickle_data)
