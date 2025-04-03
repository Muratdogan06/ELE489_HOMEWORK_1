import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    # Calculate the Euclidean distance between two points
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance


def manhattan_distance(x1, x2):
    # Calculate the Manhattan (L1) distance between two points
    distance = np.sum(np.abs(x1 - x2))
    return distance


class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        # Initialize k-NN with k (number of neighbors) and distance metric (Euclidean or Manhattan)
        self.k = k

        # Choose the distance metric function based on the user input
        if distance_metric == 'euclidean':
            self.distance_function = euclidean_distance  # Set to Euclidean distance function
        elif distance_metric == 'manhattan':
            self.distance_function = manhattan_distance  # Set to Manhattan distance function
        else:
            # Raise an error if the distance metric is not supported
            raise ValueError("Unsupported distance metric: Choose 'euclidean' or 'manhattan'.")

    def fit(self, X, y):
        # Store the training data (features and labels)
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # Predict the class label for each sample in the input X
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # Compute the distances from the input point to all points in the training set
        distances = [self.distance_function(x, x_train) for x_train in self.X_train]

        # Get the indices of the k-nearest neighbors (smallest distances)
        k_indices = np.argsort(distances)[:self.k]

        # Get the labels of the k-nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Get the most common class label (majority vote)
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]  # Return the label with the highest frequency