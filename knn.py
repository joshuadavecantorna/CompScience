def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return distance ** 0.5

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.data = []
        self.labels = []

    def fit(self, X, y):
        """Store training data"""
        self.data = X
        self.labels = y
    
    def predict(self, X_test):
        """Predict the class for each test sample"""
        predictions = []
        for test_point in X_test:
            distances = []
            for i in range(len(self.data)):
                dist = euclidean_distance(test_point, self.data[i])
                distances.append((dist, self.labels[i]))
            
            # Sort by distance and get top k neighbors
            distances.sort()
            neighbors = [label for _, label in distances[:self.k]]
            
            # Get the most common label
            prediction = max(set(neighbors), key=neighbors.count)
            predictions.append(prediction)
        return predictions

# Example usage
data = [[1, 2], [2, 3], [3, 3], [5, 5], [6, 7], [7, 8]]
labels = ['A', 'A', 'A', 'B', 'B', 'B']

knn = KNN(k=3)
knn.fit(data, labels)
test_data = [[7, 4], [3, 6]]
predictions = knn.predict(test_data)
print("Predictions:", predictions)
