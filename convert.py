import json
import pickle

# Membaca data dari file JSON
json_file_path = 'puskesmas.json'
with open(json_file_path, 'r') as json_file:
    json_data = json.load(json_file)

# Menyimpan data JSON ke file menggunakan Pickle
pickle_file_path = 'data.pkl'
with open(pickle_file_path, 'wb') as pickle_file:
    pickle.dump(json_data, pickle_file)

print(f"Data JSON dari {json_file_path} telah disimpan dalam bentuk Pickle di {pickle_file_path}")
