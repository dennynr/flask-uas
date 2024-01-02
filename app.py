from flask import Flask, render_template
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import base64
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    # 1. Membaca data dari file Pickle
    pickle_file_path = 'data.pkl'
    with open(pickle_file_path, 'rb') as pickle_file:
        pickle_data = pickle.load(pickle_file)

    # 2. Mengambil data dari dictionary
    data = pickle_data['data']

    # 3. Mengubah data ke dalam format array NumPy
    np_data = np.array([(d['Tuberkulosis'], d['Hipertensi'], d['Glukosa']) for d in data])

    # 4. Preprocessing: Standarisasi data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(np_data)

    # 5. Menentukan jumlah klaster yang diinginkan
    jumlah_klaster = 4

    # 6. Membuat model K-Means dengan jumlah klaster yang ditentukan
    kmeans = KMeans(n_clusters=jumlah_klaster, n_init=10, random_state=42)

    # 7. Melatih model pada data
    kmeans.fit(data_scaled)

    # 8. Memprediksi klaster untuk setiap data point
    prediksi_klaster = kmeans.predict(data_scaled)

    # 9. Menghitung nilai siluet
    siluet_score = silhouette_score(data_scaled, prediksi_klaster)

    # 10. Menampilkan hasil klaster, pusat klaster, dan nilai siluet
    result_text = f"Hasil Klaster:\n{prediksi_klaster}\n\nPusat Klaster:\n{kmeans.cluster_centers_}\n\nNilai Siluet: {siluet_score}"

    # 11. Visualisasi hasil klaster
    plt.switch_backend('Agg')
    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=prediksi_klaster, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
    plt.title('Hasil K-Means Clustering')
    plt.xlabel('Fitur 1 (Standarisasi)')
    plt.ylabel('Fitur 2 (Standarisasi)')

    # 12. Simpan plot ke dalam BytesIO
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    # 13. Konversi plot ke dalam format base64 untuk disertakan di HTML
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    # 14. Render template HTML dengan hasil dan plot
    return render_template('index.html', result=result_text, plot_url=plot_url)

# 15. Menjalankan aplikasi Flask jika script ini dijalankan secara langsung
if __name__ == '__main__':
    app.run(debug=True)
