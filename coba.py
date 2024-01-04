import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from io import BytesIO
import pickle
from mpl_toolkits.mplot3d import Axes3D
import base64

with open('data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

# Extract 'data' key from loaded data
data = {item['Puskesmas']: {'Tuberkulosis': item['Tuberkulosis'], 'Hipertensi': item['Hipertensi'], 'Glukosa': item['Glukosa']} for item in loaded_data['data']}
# Number of clusters
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
        distances = [euclidean_distance(extract_values(values), centroids[i]) for i in range(1, k+1)]
        cluster = np.argmin(distances) + 1
        clusters[name] = cluster
    return clusters

# Function to update centroids
def update_centroids(data, clusters):
    new_centroids = {}
    for i in range(1, k + 1):
        cluster_data = [extract_values(data[name]) for name, cluster in clusters.items() if cluster == i]
        if cluster_data:  # Check if the cluster is not empty
            cluster_data = np.array(cluster_data, dtype=float)  # Convert to numerical values
            new_centroids[i] = np.mean(cluster_data, axis=0)
        else:
            # If the cluster is empty, keep the centroid unchanged
            new_centroids[i] = centroids[i]
    return new_centroids


# K-means algorithm
for iteration in range(max_iter):
    # Assign data to clusters
    clusters = assign_to_clusters(data, centroids)

    # Print clusters and centroids for debugging
    print(f"Iteration {iteration + 1}")
    print("Clusters:", clusters)
    print("Centroids:", centroids)

    # Update centroids
    new_centroids = update_centroids(data, clusters)

    # Check for convergence
    if np.all([np.array_equal(new_centroids[i], centroids[i]) for i in range(1, k+1)]):
        break

    # Update centroids for the next iteration
    centroids = new_centroids

# Calculate silhouette score
data_array = np.array([extract_values(data[name]) for name in data])
labels = np.array(list(clusters.values()))  # Move this line here to define labels

unique_clusters = np.unique(labels)
if len(unique_clusters) == 1:
    print("Warning: Only one cluster found. Silhouette score cannot be calculated.")
    silhouette_avg = 0.0  # Set silhouette score to 0 in this case
else:
    silhouette_avg = silhouette_score(data_array, labels)
    print(f"Silhouette Score: {silhouette_avg}")
    
# Plotting the clusters and centroids in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data_array[:, 0], data_array[:, 1], data_array[:, 2], c=list(clusters.values()), cmap='viridis')
ax.scatter(np.array(list(centroids.values()))[:, 0], np.array(list(centroids.values()))[:, 1], np.array(list(centroids.values()))[:, 2], s=300, c='red', marker='X')

ax.set_title(f'K-Means Clustering\nClusters: {list(clusters.values())}\nSilhouette Score: {silhouette_avg}')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')

# Save plot to BytesIO
img = BytesIO()
plt.savefig(img, format='png')
img.seek(0)

# Convert plot to base64 format for HTML inclusion
plot_url = base64.b64encode(img.getvalue()).decode('utf8')

# Display or use the plot_url as needed
plt.show()