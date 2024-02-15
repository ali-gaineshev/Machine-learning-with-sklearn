import math
from collections import namedtuple

class Cluster:

    def __init__(self, centroid, points, num_points = 0):
        #actuall points as centroid must be saved in the points array!
        self.centroid = centroid
        self.points = points 
        self.num_points = num_points #initially centroid must be added right away!
        self.prev_iter_points = []
        self.cost = 0


    def get_cost(self):
        wcss = 0 #within-cluster sum of square
        for point in self.points:
            wcss += self.distance_to_centroid(point) ** 2
            if(self.distance_to_centroid(point) == 0):
                print("0!")
        self.cost = wcss
        return wcss
    
    #euclidian distance 
    def distance_to_centroid(self, other_point):
        return math.sqrt((self.centroid.x - other_point.x) ** 2 + (self.centroid.y - other_point.y) ** 2 + (self.centroid.z - other_point.z) ** 2)
    

    def add_to_cluster(self, point):
        self.points.append(point)
        self.num_points += 1

    def get_points(self):
        return self.points.copy()

    def update_centroid(self, new_centroid):
        self.centroid = new_centroid
        
    def set_prev_points(self, points):
        self.prev_iter_points = points

    def check_cluster_changes(self):
        if(set(self.points) == set(self.prev_iter_points)):
            return False
        else:
            return True


    def get_mean_of_points(self):
        Point = namedtuple('Point', ['x','y','z'])
        x = 0
        y = 0
        z = 0

        for coordinate in self.points:

            x += coordinate.x
            y += coordinate.y
            z += coordinate.z
            
        x = x/self.num_points
        y = y/self.num_points
        z = z/self.num_points

        new_centroid = Point(x,y,z)
        return new_centroid
    

    def check_point(self, point):
        if(point in self.points):
            self.points.remove(point)
            self.num_points -= 1

    def remove_duplicates(self):
        self.points = list(set(self.points))
        self.num_points = len(self.points)

    def __str__(self):
        centroid = f"({self.centroid.x:.3f}, {self.centroid.y:.3f}, {self.centroid.z:.3f})"
        centroid_in = "No"
        if self.centroid in self.points:
            centroid_in = "Yes"

        points = ""
        for p in self.points:
            point = f"({p.x:.3f}, {p.y:.3f}, {p.z:.3f}), "
            points += point

        print_statement = f"Centroid: {centroid}. Centroid in points array: {centroid_in} \nNum points: {self.num_points} \nPoints: {points}\n"
        return print_statement
        