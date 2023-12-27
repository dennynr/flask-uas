from flask import Flask, render_template
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/')
def index():
    # Membaca data dari file CSV
    data = pd.read_csv('data_kecamatan.csv')

    # Menampilkan jumlah baris data
    jumlah_baris = data.shape[0]

    # Memilih fitur untuk clustering
    X = data[['Jumlah_LakiLaki', 'Jumlah_Perempuan', 'Jumlah_LakiPerempuan']]

    # Menentukan jumlah klaster yang sesuai dengan jumlah sampel
    jumlah_klaster = min(3, jumlah_baris)

    # Melakukan KMeans clustering
    kmeans = KMeans(n_clusters=jumlah_klaster, random_state=42)
    kmeans.fit(X)

    # Menambahkan kolom Klaster ke DataFrame
    data['Klaster'] = kmeans.labels_

    # Visualisasi hasil klaster
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X['Jumlah_LakiLaki'], X['Jumlah_Perempuan'], X['Jumlah_LakiPerempuan'], c=data['Klaster'], cmap='viridis', s=50)
    ax.set_xlabel('Jumlah Laki-Laki')
    ax.set_ylabel('Jumlah Perempuan')
    ax.set_zlabel('Jumlah Laki-Perempuan')
    plt.title('Hasil K-Means Clustering')

    # Simpan plot ke objek BytesIO
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)

    # Konversi objek BytesIO ke base64
    plot_base64 = base64.b64encode(image_stream.getvalue()).decode('utf-8')

    # Render template dengan jumlah baris data dan plot
    return render_template('index.html', jumlah_baris=jumlah_baris, plot_base64=plot_base64)

if __name__ == '__main__':
    app.run(debug=True)
