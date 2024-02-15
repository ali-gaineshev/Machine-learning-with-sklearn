from sklearn.cluster import AgglomerativeClustering
import data
from clusters import *
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from collections import Counter

def main(dataset_num, z_coordinate):
    points = data.get_data(dataset_num = dataset_num)
    
    ks = [2,3,4,5,6,7,8]
    linkage = 'single'

    print("DATASET 1")

    print("\nSingle\n")
    
    max_score = -1
    best_k = -1
    for k in ks:
        if(max_score != -1):
            first = True
        else:
            first = False

        score = clustering(points, k, linkage, z_coordinate,first)

        if(score > max_score):
            best_k = k
            max_score = score

    print("Best k ", best_k, " with score ", max_score)

    
    
    print("\nAverage\n")
    linkage = 'average'
    
    max_score = -1
    best_k = -1
    for k in ks:
        if(max_score != -1):
            first = True
        else:
            first = False
        score = clustering(points,k, linkage, z_coordinate,first)

        if(score > max_score):
            best_k = k
            max_score = score

    print("Best k ", best_k, " with score ", max_score)
    



def clustering(points, k, linkagetype, z_coordinate, first):

    if(first):
        Z = linkage(points, method=linkagetype, metric='euclidean')
        plt.figure(figsize=(10, 6))
        dendrogram(Z, orientation='top', truncate_mode='lastp')

        plt.savefig(f"images/agglomerative/dendogram_{linkagetype}.png")

        Z = linkage(points, method='ward', metric='euclidean')
        dendrogram(Z, orientation='top', truncate_mode='lastp')
        plt.savefig(f"images/agglomerative/dendogram_ward.png")



    cluster = AgglomerativeClustering(n_clusters = k, linkage=linkagetype)
    labels = cluster.fit_predict(points)
    

    occurences(labels)

    clusters = separate_clusters(labels, points)

    if(k == 2 and linkagetype == 'single'):
        visualize_meshlab(clusters)

    visualize(clusters, z_coordinate, linkagetype, k)

    silhouette_avg = silhouette_score(points, labels)
    return silhouette_avg



def occurences(arr):

    counter = Counter(arr)

    for element, occurrences in counter.items():
        print(f"Label: {element}, Occurrences: {occurrences}")

    print("")

def separate_clusters(labels, points):
    clusters = []
    i = 0

    for x in list(set(labels)):
        
        clusters.append([])
    
    for point in points:
        clusters[labels[i]].append(point)    
        i+= 1

    return clusters

def visualize_meshlab(clusters):

    with open("output_cluster1.xyz", 'w') as file:
        for point in clusters[0]:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")

    with open("output_cluster2.xyz", 'w') as file:
        
        for point in clusters[1]:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")


def visualize(clusters, z_coordinate, linkage, k):
    image_path = "./images/agglomerative/"

    if(z_coordinate == False):
        fig, ax = plt.subplots(figsize = (12,12))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Data points with clusters. {linkage} linkage")
        i = 0
        for cluster in clusters:
            x = [point[0] for point in cluster]
            y = [point[1] for point in cluster]

            ax.plot(x, y,marker="o", linestyle = "")
            i += 1
        
        plt.savefig(f"{image_path}2d_visual_{linkage}_{k}.png")
        plt.close(fig)

    else:
        fig, ax = plt.subplots(figsize = (12,12))
        ax = fig.add_subplot(111,projection='3d')
        ax.view_init(elev=10, azim=95)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"Data points with clusters. {linkage} linkage" )

        for cluster in clusters:
            x = [point[0] for point in cluster]
            y = [point[1] for point in cluster]
            z = [point[2] for point in cluster]

            ax.scatter(x, y, z ,marker="o")
            

        plt.savefig(f"{image_path}3d_visual_{linkage}_{k}.png")
        plt.close(fig)

if __name__ == '__main__':
    main(dataset_num=1, z_coordinate=False)
    main(dataset_num=2, z_coordinate=True)