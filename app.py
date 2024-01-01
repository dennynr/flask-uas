from flask import Flask, render_template
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/')
def index():
    # Membuat dataset contoh
    data, _ = make_blobs(n_samples=300, centers=4, random_state=42)

    # Preprocessing: Standarisasi data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Menentukan jumlah klaster yang diinginkan
    jumlah_klaster = 4

    # Membuat model K-Means dengan jumlah klaster yang ditentukan
    kmeans = KMeans(n_clusters=jumlah_klaster, random_state=42)

    # Melatih model pada data
    kmeans.fit(data_scaled)

    # Memprediksi klaster untuk setiap data point
    prediksi_klaster = kmeans.predict(data_scaled)

    # Menghitung nilai siluet
    siluet_score = silhouette_score(data_scaled, prediksi_klaster)

    # Menampilkan hasil klaster, pusat klaster, dan nilai siluet
    result_text = f"Hasil Klaster:\n{prediksi_klaster}\n\nPusat Klaster:\n{kmeans.cluster_centers_}\n\nNilai Siluet: {siluet_score}"

    # Visualisasi hasil klaster
    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=prediksi_klaster, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
    plt.title('Hasil K-Means Clustering')
    plt.xlabel('Fitur 1 (Standarisasi)')
    plt.ylabel('Fitur 2 (Standarisasi)')

    # Simpan plot ke dalam BytesIO
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    # Konversi plot ke dalam format base64 untuk disertakan di HTML
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('index.html', result=result_text, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
