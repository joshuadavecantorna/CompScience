def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return distance ** 0.5

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = []
    
    def fit(self, X):
        """Train the model using K-Means clustering"""
        self.centroids = X[:self.k]  # Initialize centroids as first k points
        
        for _ in range(self.max_iters):
            clusters = [[] for _ in range(self.k)]
            
            for point in X:
                distances = [euclidean_distance(point, centroid) for centroid in self.centroids]
                cluster_index = distances.index(min(distances))
                clusters[cluster_index].append(point)
            
            new_centroids = []
            for cluster in clusters:
                if cluster:
                    new_centroid = [sum(dim) / len(cluster) for dim in zip(*cluster)]
                    new_centroids.append(new_centroid)
                else:
                    new_centroids.append([0] * len(X[0]))
            
            if new_centroids == self.centroids:
                break
            self.centroids = new_centroids
    
    def predict(self, X_test):
        """Assign each test point to the nearest cluster"""
        predictions = []
        for point in X_test:
            distances = [euclidean_distance(point, centroid) for centroid in self.centroids]
            cluster_index = distances.index(min(distances))
            predictions.append(cluster_index)
        return predictions

# Example usage
data = [[1, 2], [2, 3], [3, 3], [5, 5], [6, 7], [7, 8]]
kmeans = KMeans(k=2)
kmeans.fit(data)
test_data = [[4, 4], [6, 6]]
predictions = kmeans.predict(test_data)
print("Cluster assignments:", predictions)
