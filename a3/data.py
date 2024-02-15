import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from collections import namedtuple

def extract_file_name(file_path):
    file_name = os.path.basename(file_path)
    return file_name


def read_data(file, visualize = False):
    Point = namedtuple('Point', ['x','y','z'])
    data = []

    x_data = []
    y_data = []
    z_data = []

    x_less = []
    y_less = []
    z_less = []

    columns = 0
    with open(file, 'r') as f:
        
        csv_reader = csv.reader(f, delimiter=',')
        line_c = 0
        
        for row in csv_reader: 
            columns = len(row)
            
            x = float(row[0])
            y = float(row[1])
            if(columns == 3):
                z = float(row[2])

                if((z > -7 and z < 7) and (x < 7 and x > -7)):
                    x_less.append(x)
                    y_less.append(y)
                    z_less.append(z)
                
            else:
                z = 0

            point = Point(x,y,z)
            data.append(point)   

            x_data.append(x)
            y_data.append(y)
            z_data.append(z)

            line_c += 1

    file = extract_file_name(file)

    if(file == ""):
        print("Error with the file")
        exit(1)


    if(visualize == True):
        print(f'File path: "{file}". Processed {line_c} lines.')

        
        visualize_data(file, x_data, y_data, z_data, columns)
        
        if(x_less != []):
            visualize_data("inner_" + file, x_less, y_less, z_less, columns)

    else:
        return data



def visualize_data(file, x, y, z, columns):

    image_path = "./images/"

    if(columns == 2):
        fig, ax = plt.subplots(figsize = (12,8))

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Data points" )
    
        ax.plot(x, y, marker="o", linestyle = "")

        plt.savefig(f"{image_path}{file}_visual.png")


    else:
        fig, ax = plt.subplots(figsize = (12,8))
        ax = fig.add_subplot(111,projection='3d')

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"Data points" )

    
        ax.scatter(x, y, z, c = [i for i in range(len(x))],marker="o")

        plt.savefig(f"{image_path}{file}_visual.png")

def get_data(dataset_num):
    return read_data(f"./data/dataset{dataset_num}.csv", visualize=False)

if __name__ == '__main__':
    read_data("./data/dataset1.csv", visualize=True)
    read_data("./data/dataset2.csv", visualize=True)

