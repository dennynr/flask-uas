from flask import Flask, render_template, request
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from io import BytesIO
import pickle
from mpl_toolkits.mplot3d import Axes3D
import base64
import pandas as pd

app = Flask(__name__)

data = {}  # Declare data as a global variable
k = 3  # Declare k as a global variable
max_iter = 200  # Declare max_iter as a global variable

@app.route('/')
def index():
    global data, k, max_iter

    k = 3  # Define the value of k here

    with open('data.pkl', 'rb') as file:
        loaded_data = pickle.load(file)

    data = {item['Puskesmas']: {'Tuberkulosis': item['Tuberkulosis'],
                                'Hipertensi': item['Hipertensi'], 'Glukosa': item['Glukosa']} for item in loaded_data['data']}
    # Rest of your code...
    k = 3

    # Maximum number of iterations
    max_iter = 200  # or any other suitable value

    # Initial centroids
    centroids = {
        1: np.array([89, 23000, 2500]),
        2: np.array([79, 18000, 2000]),
        3: np.array([69, 13000, 1500])
    }

    # Function to calculate Euclidean distance
    def euclidean_distance(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    # Function to extract numerical values from dictionaries
    def extract_values(dictionary):
        return list(dictionary.values())

    # Function to assign data to clusters
    def assign_to_clusters(data, centroids):
        clusters = {}
        for name, values in data.items():
            distances = [euclidean_distance(extract_values(
                values), centroids[i]) for i in range(1, k+1)]
            cluster = np.argmin(distances) + 1
            clusters[name] = cluster
        return clusters

    # Function to update centroids
    def update_centroids(data, clusters):
        new_centroids = {}
        for i in range(1, k + 1):
            cluster_data = [extract_values(
                data[name]) for name, cluster in clusters.items() if cluster == i]
            if cluster_data:  # Check if the cluster is not empty
                # Convert to numerical values
                cluster_data = np.array(cluster_data, dtype=float)
                new_centroids[i] = np.mean(cluster_data, axis=0)
            else:
                # If the cluster is empty, keep the centroid unchanged
                new_centroids[i] = centroids[i]
        return new_centroids

    # Append results for the initial state (iteration 0)
    clusters = assign_to_clusters(data, centroids)
    iteration_numbers = [0]
    iteration_clusters = [clusters.copy()]
    iteration_centroids = [centroids.copy()]

    # K-means algorithm
    for iteration in range(1, max_iter + 1):
        # Assign data to clusters
        clusters = assign_to_clusters(data, centroids)

        # Print clusters and centroids for debugging
        print(f"Iteration {iteration}")
        print("Clusters:", clusters)
        print("Centroids:", centroids)

        # Update centroids
        new_centroids = update_centroids(data, clusters)

        # Check for convergence
        if np.all([np.array_equal(new_centroids[i], centroids[i]) for i in range(1, k+1)]):
            break

        # Update centroids for the next iteration
        centroids = new_centroids

        # Append results to the lists
        iteration_numbers.append(iteration)
        iteration_clusters.append(clusters.copy())
        iteration_centroids.append(centroids.copy())

    # Calculate silhouette score
    data_array = np.array([extract_values(data[name]) for name in data])
    labels = np.array(list(clusters.values()))

    unique_clusters = np.unique(labels)
    if len(unique_clusters) == 1:
        print("Warning: Only one cluster found. Silhouette score cannot be calculated.")
        silhouette_avg = 0.0  # Set silhouette score to 0 in this case
    else:
        silhouette_avg = silhouette_score(data_array, labels)
        print(f"Silhouette Score: {silhouette_avg}")

# Plotting the clusters and centroids in 3D
    plt.figure()
    fig = plt.figure(figsize=(10, 8))  # Adjust the size as needed
    ax = fig.add_subplot(111, projection='3d')


    ax.scatter(data_array[:, 0], data_array[:, 1], data_array[:, 2], c=list(
        clusters.values()), cmap='viridis')
    ax.scatter(np.array(list(centroids.values()))[:, 0], np.array(list(centroids.values()))[
               :, 1], np.array(list(centroids.values()))[:, 2], s=300, c='red', marker='X')

    ax.set_title(
        f'K-Means Clustering\nClusters: {list(clusters.values())}\nSilhouette Score: {silhouette_avg}')
    ax.set_xlabel('Tuberkulosis')
    ax.set_ylabel('Hipertensi')
    ax.set_zlabel('Glukosa')


    # Convert lists to a DataFrame
    iteration_results = pd.DataFrame({
        'Iteration': iteration_numbers,
        'Clusters': iteration_clusters,
        'Centroids': iteration_centroids
    })

    # Save the dataframe to HTML for displaying as a table
    iteration_table_html = iteration_results.to_html(index=False)

    # Save plot to BytesIO
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Convert plot to base64 format for HTML inclusion
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Pass both the table and plot_url to the template
    return render_template('index.html', iteration_table_html=iteration_table_html, plot_url=plot_url, iteration_results=iteration_results)


@app.route('/update_centroids', methods=['POST'])
def update_centroids():
    global iteration_table_html, plot_url, data, k, max_iter  # Add max_iter to global variables

    centroid1 = np.array([
        int(request.form['centroid1_x']),
        int(request.form['centroid1_y']),
        int(request.form['centroid1_z'])
    ])

    centroid2 = np.array([
        int(request.form['centroid2_x']),
        int(request.form['centroid2_y']),
        int(request.form['centroid2_z'])
    ])

    centroid3 = np.array([
        int(request.form['centroid3_x']),
        int(request.form['centroid3_y']),
        int(request.form['centroid3_z'])
    ])

    centroids = {
        1: centroid1,
        2: centroid2,
        3: centroid3
    }

 
    # Function to calculate Euclidean distance
    def euclidean_distance(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    # Function to extract numerical values from dictionaries
    def extract_values(dictionary):
        return list(dictionary.values())

    # Function to assign data to clusters
    def assign_to_clusters(data, centroids):
        clusters = {}
        for name, values in data.items():
            distances = [euclidean_distance(extract_values(
                values), centroids[i]) for i in range(1, k+1)]
            cluster = np.argmin(distances) + 1
            clusters[name] = cluster
        return clusters

    # Function to update centroids
    def update_centroids(data, clusters):
        new_centroids = {}
        for i in range(1, k + 1):
            cluster_data = [extract_values(
                data[name]) for name, cluster in clusters.items() if cluster == i]
            if cluster_data:  # Check if the cluster is not empty
                # Convert to numerical values
                cluster_data = np.array(cluster_data, dtype=float)
                new_centroids[i] = np.mean(cluster_data, axis=0)
            else:
                # If the cluster is empty, keep the centroid unchanged
                new_centroids[i] = centroids[i]
        return new_centroids

    # Append results for the initial state (iteration 0)
    clusters = assign_to_clusters(data, centroids)
    iteration_numbers = [0]
    iteration_clusters = [clusters.copy()]
    iteration_centroids = [centroids.copy()]

    # K-means algorithm
    for iteration in range(1, max_iter + 1):
        # Assign data to clusters
        clusters = assign_to_clusters(data, centroids)

        # Print clusters and centroids for debugging
        print(f"Iteration {iteration}")
        print("Clusters:", clusters)
        print("Centroids:", centroids)

        # Update centroids
        new_centroids = update_centroids(data, clusters)

        # Check for convergence
        if np.all([np.array_equal(new_centroids[i], centroids[i]) for i in range(1, k+1)]):
            break

        # Update centroids for the next iteration
        centroids = new_centroids

        # Append results to the lists
        iteration_numbers.append(iteration)
        iteration_clusters.append(clusters.copy())
        iteration_centroids.append(centroids.copy())

    # Calculate silhouette score
    data_array = np.array([extract_values(data[name]) for name in data])
    labels = np.array(list(clusters.values()))

    unique_clusters = np.unique(labels)
    if len(unique_clusters) == 1:
        print("Warning: Only one cluster found. Silhouette score cannot be calculated.")
        silhouette_avg = 0.0  # Set silhouette score to 0 in this case
    else:
        silhouette_avg = silhouette_score(data_array, labels)
        print(f"Silhouette Score: {silhouette_avg}")

    # Plotting the clusters and centroids in 3D
    fig = plt.figure(figsize=(10, 8))  # Adjust the size as needed
    ax = fig.add_subplot(111, projection='3d')


    ax.scatter(data_array[:, 0], data_array[:, 1], data_array[:, 2], c=list(
        clusters.values()), cmap='viridis')
    ax.scatter(np.array(list(centroids.values()))[:, 0], np.array(list(centroids.values()))[
               :, 1], np.array(list(centroids.values()))[:, 2], s=300, c='red', marker='X')

    ax.set_title(
        f'K-Means Clustering\nClusters: {list(clusters.values())}\nSilhouette Score: {silhouette_avg}')
    ax.set_xlabel('Tuberkulosis')
    ax.set_ylabel('Hipertensi')
    ax.set_zlabel('Glukosa')

    # Convert lists to a DataFrame
    iteration_results = pd.DataFrame({
        'Iteration': iteration_numbers,
        'Clusters': iteration_clusters,
        'Centroids': iteration_centroids
    })

    # Save the dataframe to HTML for displaying as a table
    iteration_table_html = iteration_results.to_html(index=False)

    # Save plot to BytesIO
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Convert plot to base64 format for HTML inclusion
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Pass both the table and plot_url to the template
    return render_template('index.html', iteration_table_html=iteration_table_html, plot_url=plot_url, iteration_results=iteration_results)


if __name__ == '__main__':
    app.run(debug=True)
