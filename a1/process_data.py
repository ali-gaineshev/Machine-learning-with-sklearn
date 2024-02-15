import random

def process(file = './data/spambase.data'):
    lines = []
    features_max_value = get_max_of_features()
    with open(file,"r") as f:
        for line in f:
            lines.append(line.strip())
    
    random.shuffle(lines)

    data_x = []
    data_y = []
    for line in lines:
        dataset = line.split(',')
        if(len(dataset) > 2):
            x = [float(value) for value in dataset[:-1]]
            y = int(dataset[-1])
        
            x = [x[i] / features_max_value[i] for i in range(len(x))]

            data_x.append(x)
            data_y.append(y)

    return data_x,data_y

def cut_training_data(data_x,data_y, percentage):

    new_size = int(len(data_x) * percentage / 100)
    new_x_train = []
    new_y_train = []
    counter = 0

    for i in range(len(data_x)):
        if counter != new_size:
            new_x_train.append(data_x[i])
            new_y_train.append(data_y[i])
            counter += 1
        else:
            break
    return new_x_train, new_y_train

def get_max_of_features(file = './data/max_values.txt'):
    max_values = []
    feature_info = []
    with open(file,"r") as f:
        for line in f:
            data = line.split(" ")
            for value in data:
                if value != '':
                    feature_info.append(value)
            max_values.append(float(feature_info[2]))
            feature_info = []
    return max_values

def get_features(file = './data/feature_names.txt'):
    lines = []
    with open(file,"r") as f:
        for line in f:
            lines.append(line.strip())

    return lines
    