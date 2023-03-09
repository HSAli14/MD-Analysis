import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pytraj as pt

def parse_args():
    parser = argparse.ArgumentParser(description='Perform K-means clustering on PCA data')
    parser.add_argument('input_traj', help='Input trajectory file in NetCDF format')
    parser.add_argument('input_top', help='Input topology file in Amber format')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()

    # Load data and perform PCA
    traj = pt.load(args.input_traj, top=args.input_top)
    data = pt.pca(traj, '@CA', n_vecs=2)
    data1 = data[0]
    data2 = np.transpose(data1)

    # Perform K-means clustering and plot the results
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data2)
    labels = kmeans.labels_

    # Calculate the center of each cluster and mark it on the plot
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ['blue', 'red', 'green', 'magenta', 'purple', 'yellow'] # set color for each cluster
    centers = kmeans.cluster_centers_ # get the center of each cluster

    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        ax.scatter(data2[cluster_indices, 0], data2[cluster_indices, 1], color=colors[i], s=20)
        # mark the center of each cluster on the plot
        ax.scatter(centers[i, 0], centers[i, 1], color='black', marker='*', s=150)

    plt.xlabel('PC1', fontsize = 14)
    plt.ylabel('PC2', fontsize = 14) 
    plt.xticks(np.arange(-10,11, 5), fontsize = 14)
    plt.yticks(np.arange(-10,11, 5), fontsize = 14)

    plt.rcParams['font.size'] = '14'
    ax.set_title("K-means Clustering")
    plt.savefig('PCA.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
