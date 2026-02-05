import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

#set a seed for reproducibility
np.random.seed(42)


# Get sklearn.datasets.make blobs synthetic data.(Well-separated, spherical clusters – good for baseline evaluation.)
# Get sklearn.datasets.make moons synthetic data.(non-linearly separable/non-spherical clusters – used to explore K-Means limitations.)
# To ensure visual clarity and interpretability of your scatter plots, use approximately 100-200 data points when generating each synthetic dataset.
X_make_blobs, _ = make_blobs(n_samples=150)
X_make_moons, _ = make_moons(n_samples=150)

# Part 1: K-Means Clustering

print("K-Means Clustering Results:\n")
# Apply K-Means (init=’random’) clustering on both datasets
for k in [2,3,4,5]:
    # Apply K-Means clustering for make_blobs dataset
    kmeans = KMeans(n_clusters=k, init='random')
    kmeans.fit(X_make_blobs)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Visualize the clusters for make_blobs dataset. Generate 2D scatter plots of the clustering results. Color each cluster distinctly and mark the cluster centers using model.cluster_centers_.
    plt.figure()
    plt.scatter(X_make_blobs[:, 0], X_make_blobs[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X')
    plt.title(f'K-Means Clustering on make_blobs (k={k}, init=random)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    # Use the silhouette score (sklearn.metrics.silhouette_score) to evaluate how well-separated the resulting clusters are
    silhouette_avg = silhouette_score(X_make_blobs, labels)
    print(f'Silhouette Score for make_blobs (k={k}): {silhouette_avg}')

    # Apply K-Means clustering for make_moons dataset
    kmeans = KMeans(n_clusters=k, init='random')
    kmeans.fit(X_make_moons)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Visualize the clusters for make_moons dataset
    plt.figure()
    plt.scatter(X_make_moons[:, 0], X_make_moons[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X')
    plt.title(f'K-Means Clustering on make_moons (k={k}, init=random)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    
    # Use the silhouette score (sklearn.metrics.silhouette_score) to evaluate how well-separated the resulting clusters are.
    silhouette_avg = silhouette_score(X_make_moons, labels)
    print(f'Silhouette Score for make_moons (k={k}): {silhouette_avg}')
    print()

print("\nK-Means++ Clustering Results:\n")
# Apply K-Means++ (init=k-means++) clustering on both datasets
for k in [2,3,4,5]:
    # Apply K-Means clustering for make_blobs dataset
    kmeans_plus = KMeans(n_clusters=k, init='k-means++')
    kmeans_plus.fit(X_make_blobs)
    labels = kmeans_plus.labels_
    centroids = kmeans_plus.cluster_centers_

    # Visualize the clusters for make_blobs dataset
    plt.figure()
    plt.scatter(X_make_blobs[:, 0], X_make_blobs[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X')
    plt.title(f'K-Means Clustering on make_blobs (k={k}, init=k-means++)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    # Use the silhouette score (sklearn.metrics.silhouette_score) to evaluate how well-separated the resulting clusters are
    silhouette_avg = silhouette_score(X_make_blobs, labels)
    print(f'Silhouette Score for make_blobs (k={k}): {silhouette_avg}')

    # Apply K-Means clustering for make_moons dataset
    kmeans_plus = KMeans(n_clusters=k, init='k-means++')
    kmeans_plus.fit(X_make_moons)
    labels = kmeans_plus.labels_
    centroids = kmeans_plus.cluster_centers_

    # Visualize the clusters for make_moons dataset
    plt.figure()
    plt.scatter(X_make_moons[:, 0], X_make_moons[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X')
    plt.title(f'K-Means Clustering on make_moons (k={k}, init=k-means++)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    # Use the silhouette score (sklearn.metrics.silhouette_score) to evaluate how well-separated the resulting clusters are.
    silhouette_avg = silhouette_score(X_make_moons, labels)
    print(f'Silhouette Score for make_moons (k={k}): {silhouette_avg}')
    print()


# Part 2: Expectation-Maximization (Gaussian Mixture Models)

# Use sklearn.mixture.GaussianMixture to implement Expectation-Maximization (EM), a probabilistic clustering algorithm that models data as a mixture of Gaussian distributions.
# Apply EM to the same datasets used in Part 1 (make_blobs and make_moons) with varying numbers of components (k=2, 3, 4, 5) covariance type (full for general ellipses vs. diag for axis-aligned ellipses).

print("\nExpectation-Maximization GMM (covariance_type='full') Results:\n")
for k in [2,3,4,5]:
    # Apply Gaussian Mixture Model for make_blobs dataset
    gmm = GaussianMixture(n_components=k, covariance_type='full')
    gmm.fit(X_make_blobs)
    labels = gmm.predict(X_make_blobs)
    centroids = gmm.means_

    # Visualize the clusters for make_blobs dataset. Use ellipses to represent the Gaussian components (e.g., using matplotlib.patches.Ellipse)
    plt.figure()
    plt.scatter(X_make_blobs[:, 0], X_make_blobs[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X')

    # Add ellipses for Gaussian components
    covariances = gmm.covariances_
    for i in range(k):
        eigenvalues, eigenvectors = np.linalg.eigh(covariances[i])
        #sort eigenvalues and eigenvectors
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        
        width, height = 2 * np.sqrt(eigenvalues)
        ellipse = Ellipse(centroids[i], width, height, angle=angle, edgecolor='black', facecolor='none', linewidth=2)
        plt.gca().add_patch(ellipse)
    plt.title(f'GMM Clustering on make_blobs (k={k}) covariance_type=full')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    # Use the silhouette score to evaluate how well-separated the resulting clusters are.
    silhouette_avg = silhouette_score(X_make_blobs, labels)
    print(f'Silhouette Score for make_blobs GMM (k={k}): {silhouette_avg}')
    # Report the average log-likelihood of the fitted model using GaussianMixture.score(X), which gives the per-sample log-likelihood of the data under the learned mixture model.
    log_likelihood = gmm.score(X_make_blobs)
    print(f'Average Log-Likelihood for make_blobs GMM (k={k}): {log_likelihood}')
    # Compute the Bayesian Information Criterion (BIC) using GaussianMixture.bic(X). Lower BIC values indicate a better balance between model fit and model complexity.
    bic = gmm.bic(X_make_blobs)
    print(f'Bayesian Information Criterion (BIC) for make_blobs GMM (k={k}): {bic}')

    # Apply Gaussian Mixture Model for make_moons dataset
    gmm = GaussianMixture(n_components=k, covariance_type='full')
    gmm.fit(X_make_moons)
    labels = gmm.predict(X_make_moons)
    centroids = gmm.means_

    # Visualize the clusters for make_moons dataset with ellipses representing Gaussian components
    plt.figure()
    plt.scatter(X_make_moons[:, 0], X_make_moons[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X')

    # Add ellipses for Gaussian components
    covariances = gmm.covariances_
    for i in range(k):
        eigenvalues, eigenvectors = np.linalg.eigh(covariances[i])
        #sort eigenvalues and eigenvectors
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        
        width, height = 2 * np.sqrt(eigenvalues)
        ellipse = Ellipse(centroids[i], width, height, angle=angle, edgecolor='black', facecolor='none', linewidth=2)
        plt.gca().add_patch(ellipse)

    plt.title(f'GMM Clustering on make_moons (k={k}, covariance_type=full)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    # Use the silhouette score to evaluate how well-separated the resulting clusters are.
    silhouette_avg = silhouette_score(X_make_moons, labels)
    print(f'Silhouette Score for make_moons GMM (k={k}): {silhouette_avg}')
    # Report the average log-likelihood of the fitted model using GaussianMixture.score(X), which gives the per-sample log-likelihood of the data under the learned mixture model.
    log_likelihood = gmm.score(X_make_moons)
    print(f'Average Log-Likelihood for make_moons GMM (k={k}): {log_likelihood}')
    # Compute the Bayesian Information Criterion (BIC) using GaussianMixture.bic(X). Lower BIC values indicate a better balance between model fit and model complexity.
    bic = gmm.bic(X_make_moons)
    print(f'Bayesian Information Criterion (BIC) for make_moons GMM (k={k}): {bic}')

    print()


print("\nExpectation-Maximization GMM (covariance_type='diag') Results:\n")
for k in [2,3,4,5]:
    # Apply Gaussian Mixture Model for make_blobs dataset
    gmm = GaussianMixture(n_components=k, covariance_type='diag')
    gmm.fit(X_make_blobs)
    labels = gmm.predict(X_make_blobs)
    centroids = gmm.means_

    # Visualize the clusters for make_blobs dataset
    plt.figure()
    plt.scatter(X_make_blobs[:, 0], X_make_blobs[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X')
    # Add ellipses for Gaussian components
    covariances = gmm.covariances_
    for i in range(k):
        eigenvalues, eigenvectors = np.linalg.eigh(np.diag(covariances[i]))
        #sort eigenvalues and eigenvectors
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        width, height = 2 * np.sqrt(eigenvalues)
        ellipse = Ellipse(centroids[i], width, height, angle=0, edgecolor='black', facecolor='none', linewidth=2)
        plt.gca().add_patch(ellipse)

    plt.title(f'GMM Clustering on make_blobs (k={k}, covariance_type=diag)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    
    # Use the silhouette score to evaluate how well-separated the resulting clusters are.
    silhouette_avg = silhouette_score(X_make_blobs, labels)
    print(f'Silhouette Score for make_blobs GMM (k={k}, diag): {silhouette_avg}')
    # Report the average log-likelihood of the fitted model using GaussianMixture.score(X), which gives the per-sample log-likelihood of the data under the learned mixture model.
    log_likelihood = gmm.score(X_make_blobs)
    print(f'Average Log-Likelihood for make_blobs GMM (k={k}, diag): {log_likelihood}')
    # Compute the Bayesian Information Criterion (BIC) using GaussianMixture.bic(X). Lower BIC values indicate a better balance between model fit and model complexity.
    bic = gmm.bic(X_make_blobs)
    print(f'Bayesian Information Criterion (BIC) for make_blobs GMM (k={k}, diag): {bic}')
    
    # Apply Gaussian Mixture Model for make_moons dataset
    gmm = GaussianMixture(n_components=k, covariance_type='diag')
    gmm.fit(X_make_moons)
    labels = gmm.predict(X_make_moons)
    centroids = gmm.means_

    # Visualize the clusters for make_moons dataset
    plt.figure()
    plt.scatter(X_make_moons[:, 0], X_make_moons[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X')
    # Add ellipses for Gaussian components
    covariances = gmm.covariances_
    for i in range(k):
        eigenvalues, eigenvectors = np.linalg.eigh(np.diag(covariances[i]))
        #sort eigenvalues and eigenvectors
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
    
        width, height = 2 * np.sqrt(eigenvalues)
        ellipse = Ellipse(centroids[i], width, height, angle=0, edgecolor='black', facecolor='none', linewidth=2)
        plt.gca().add_patch(ellipse)

    plt.title(f'GMM Clustering on make_moons (k={k}, covariance_type=diag)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    # Use the silhouette score to evaluate how well-separated the resulting clusters are.
    silhouette_avg = silhouette_score(X_make_moons, labels)
    print(f'Silhouette Score for make_moons GMM (k={k}, diag): {silhouette_avg}')
    # Report the average log-likelihood of the fitted model using GaussianMixture.score(X), which gives the per-sample log-likelihood of the data under the learned mixture model.
    log_likelihood = gmm.score(X_make_moons)
    print(f'Average Log-Likelihood for make_moons GMM (k={k}, diag): {log_likelihood}')
    # Compute the Bayesian Information Criterion (BIC) using GaussianMixture.bic(X). Lower BIC values indicate a better balance between model fit and model complexity.
    bic = gmm.bic(X_make_moons)
    print(f'Bayesian Information Criterion (BIC) for make_moons GMM (k={k}, diag): {bic}')

    print()