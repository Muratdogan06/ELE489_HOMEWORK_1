# K-Nearest-Neighbors-KNN-Algorithm-with-Wine-Dataset-Analysis

## Project Description

This project implements the **K-Nearest Neighbors (KNN)** algorithm to classify data from the **Wine dataset**. The dataset is used to predict the type of wine based on various features such as alcohol content, flavanoids, ash, and magnesium. The project compares a custom KNN implementation with the standard Sklearn KNeighborsClassifier and evaluates their performance using accuracy and confusion matrices for different values of **K** (number of neighbors) and distance metrics (Euclidean and Manhattan).

## Instructions to Run the Code

1. **Clone the repository** or download the project files.
   
   ```bash
   git clone <repository-url>
Install required libraries:

2. You can install the necessary libraries using pip:

 ```bash
 pip install numpy pandas scikit-learn matplotlib seaborn ucimlrepo
deneme

3.Run the Jupyter Notebook:

Open the analysis.ipynb notebook in Jupyter Notebook or JupyterLab. Follow the instructions inside the notebook:

The notebook loads and preprocesses the Wine dataset.

It then trains the KNN model using both the custom and Sklearn implementations for different K values.

Accuracy and confusion matrices are computed to compare the models' performance.

4. View the Results:

Accuracy vs K values: Displays the accuracy for various values of K.

Confusion Matrices: Visualizes the classification performance for both the custom KNN and Sklearn KNN models.

## Conclusion
The project compares custom KNN and Sklearn implementations on the Wine dataset, finding the optimal value of K using different distance metrics. Confusion matrices help assess the model's performance in detail.
