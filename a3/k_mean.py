import data
import random
from clusters import *
import numpy as np

import matplotlib.pyplot as plt


def random_initialization(points, k):
    k = min(k, len(points))
    random_centers = random.sample(points, k)
    
    clusters = []
    for centroid in random_centers:
        
        cluster = Cluster(centroid, points = [centroid], num_points= 1)
        clusters.append(cluster)

    return clusters

def k_means_plusplus(points, k):
    # first is random
    clusters = random_initialization(points, 1)

    for _ in range(k - 1):
        distances_to_center = []

        for point in points:
            min_distance = min(cluster.distance_to_centroid(point) ** 2 for cluster in clusters)#squared euclidian distance

            #points that are already centroids have distance of 0
            distances_to_center.append(min_distance)

        # probability proportional to the squared distance
        total_distance = sum(distances_to_center)
        probabilities = []

        i = 0
        for point in points:
            p = distances_to_center[i]/total_distance
            probabilities.append(p)
            i+=1
            
        #points tuple(float,float,float)
        
        #np.random choice will select assign probability to array [0, len(points)) and then select a number which will be an index for centroid
        centroid_i: list = np.random.choice([index for index in range(len(points))], size = 1, p = probabilities)
    
        centroid = points[centroid_i[0]]
        clusters.append(Cluster(centroid = centroid,points = [centroid], num_points=1))

    return clusters

def clustering(points, dataset_num, init, k):
    #main clustering algoritm
    if(init == "random"):
        clusters = random_initialization(points, k)

    if(init == "++"):
        clusters = k_means_plusplus(points, k)

    changes = 2 * k # 2 iterations of changes for convergence within any cluster
    while(True):
        if(changes == 0): #for 2 iterations, no changes
            break

        for point in points:
            distances_to_center = []

            #assign points
            for cluster in clusters:
                if(point != cluster.centroid):
                    cluster.check_point(point)#remove point if present
                    distance = cluster.distance_to_centroid(point)
                    distances_to_center.append(distance)
            
            #found distances to each cluster, pick min
            min_dist_i = distances_to_center.index(min(distances_to_center))

            clusters[min_dist_i].add_to_cluster(point)

        first = True
        for cluster in clusters:
            #get new centroid
            cluster.remove_duplicates()
            new_centroid = cluster.get_mean_of_points()
            cluster.update_centroid(new_centroid)

            #check for any changes, if so reset number of changes value
            if(first == True):
                if(cluster.prev_iter_points != []):
                    any_changes = cluster.check_cluster_changes()
                    if(any_changes == False):
                        changes -= 1
                    else:
                        first = False
                        changes = 2 * k #reset changes 
                else:
                    changes = 2 * k

            cluster.set_prev_points(cluster.get_points())

    #calculate cost
    total_cost = 0
    for cluster in clusters:
        total_cost += cluster.get_cost()


    if(total_cost == 0):
        print("Something went wrong, cost is 0")

    return clusters, total_cost


def experiment(points, k, init, dataset_num):
    """
    perform clustering on the same k value 5 times and pick the best one with lowest cost
    """
    best_clusters = []
    min_cost = 9999999999
    


    if(dataset_num == 1):
        for i in range(5):
            clusters, cost = clustering(points, dataset_num , init = init, k = k)
            if(cost < min_cost):
                best_clusters = clusters
                min_cost = cost

        visualize(best_clusters, False,f"{init}_k_{k}", min_cost)

        return best_clusters, min_cost
    
    else:
        clusters, cost = clustering(points, dataset_num , init = init, k = k)

        visualize(best_clusters, z_coordinate=True,name = f"{init}_k_{k}", total_cost=min_cost)

        return clusters, cost
    

def main(init):
    ks = [2,3,4,5,6,7,8]
    dataset_num = 1
    points = data.get_data(dataset_num)
    print("DATASET 1", end= ". ")
    z_coordinate = False


    all_costs = []
    best_cluster = None
    min_cost = 9999999999

    for k in ks:
        clusters, cost = experiment(points, k, init, dataset_num)
        all_costs.append(cost)

        #get the best cluster with min cost out of all experiment with k
        if(cost < min_cost): # always max k :()
            best_cluster = clusters


    visualize_cost(ks,all_costs, init)
    print("Done!")
    
    """-----------------3d points----------------"""
    
    
    print("\nDATASET 2", end =". ")
    z_coordinate = True
    all_costs = []
    best_cluster = None
    min_cost = 9999999999
    dataset_num = 2
    points = data.get_data(dataset_num)

    for k in ks:
        print("DOING ", k)
        clusters, cost = experiment(points, k, init, dataset_num)
        all_costs.append(cost)

        if(cost < min_cost):
            best_cluster = clusters
            min_cost = cost

    
    visualize_cost(ks,all_costs, init, z_coordinate)
    print("Done!")
    

def visualize_cost(k, cost, name, z_coor = False):
    
    two_d = "2d" if not z_coor else "3d"
    image_path = "./images/k_means_random/" +two_d + "_"
    fig, ax = plt.subplots()
    ax.set_xlabel("k")
    ax.set_ylabel("cost")
    ax.set_title(f"k vs cost")
    
    ax.plot(k, cost,marker="o", linestyle = "-")
    
    
    plt.savefig(f"{image_path}k_vs_cost_{name}.png")

def visualize(clusters, z_coordinate, name, total_cost = None):
    image_path = "./images/k_means_random/"

    if(z_coordinate == False):
        fig, ax = plt.subplots(figsize = (12,12))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Data points with clusters with total cost {total_cost}")
        i = 0
        for cluster in clusters:
            x = [point.x for point in cluster.points]
            y = [point.y for point in cluster.points]

            ax.plot(x, y,marker="o", linestyle = "")
            i += 1
        
        plt.savefig(f"{image_path}2d_visual_{name}.png")
        plt.close(fig)

    else:
        fig, ax = plt.subplots(figsize = (12,12))
        ax = fig.add_subplot(111,projection='3d')

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"Data points with clusters" )

        for cluster in clusters:
            x = [point.x for point in cluster.points]
            y = [point.y for point in cluster.points]
            z = [point.z for point in cluster.points]

            ax.scatter(x, y, z ,marker="o")
            
        #ax.view_init(elev=20, azim=95)
        plt.savefig(f"{image_path}3d_visual_{name}.png")
        plt.close(fig)

if __name__ == "__main__":

    print("Random initialization\n")
    main(init = "random")

    print("\nK means ++")
    main(init = "++")