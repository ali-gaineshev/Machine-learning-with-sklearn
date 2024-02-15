import utils.mnist_reader as mnist_reader
import numpy as np
import random 

def flip(num):
    if num == 0:
        return 1
    
    if num == 1:
        return 0
    
    return -1

def make_train_noisy(y):
    new_y = []
    flip_p = 0.20
    for i in range(len(y)):
        rnd_num = random.random()
        value = y[i]
        if(flip_p > rnd_num): #within 0.2 -> flip
            value = flip(value)
        new_y.append(value)

    counter = check_noise(new_y, y)
    return new_y, counter

def check_noise(new_y, y):
    counter = 0
    for i in range(len(new_y)):
        if(new_y[i] != y[i]):
            counter += 1
    
    return counter

def get_file_data():
    x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    five_c = 0
    seven_c = 0
    found = False
    new_x_train, new_y_train, new_x_test, new_y_test = [], [], [], []
    for train_i in range(x_train.shape[0]):
        if y_train[train_i] == 5: 
            new_y_train.append(0)
            found = True
            five_c += 1
        elif y_train[train_i] == 7:
            new_y_train.append(1)
            found = True
            seven_c += 1

        if(found == True):
            x_data = []
            for x in x_train[train_i]:
                x_data.append(x/255)
            new_x_train.append(x_data)
        found = False

    print("Train set y = 0 - ", five_c)
    print("Train set y = 1 - ", seven_c)

    
    five_c = 0
    seven_c = 0
    for test_i in range(x_test.shape[0]):
        if y_test[test_i] == 5: 
            new_y_test.append(0)
            found = True
            five_c += 1
        elif y_test[test_i] == 7:
            new_y_test.append(1)
            found = True
            seven_c += 1

        if(found == True):
            x_data = []
            for x in x_test[test_i]:
                x_data.append(x/255)
            new_x_test.append(x_data)

        found = False
    
    print("Test set y = 0 - ", five_c)
    print("Test set y = 1 - ", seven_c)

    print("Old train size ", len(new_x_train))
    new_x_train = new_x_train[::2]
    new_y_train = new_y_train[::2]
    print("New train size ",len(new_x_train))

    new_y_train,noise_counter = make_train_noisy(new_y_train)
    print("Flipped ", noise_counter, " values")
    print("Number of attributes: ", len(x_train[0]))


    np.savez("data/arrays_data.npz",new_x_train = new_x_train,new_y_train = new_y_train, new_x_test = new_x_test, new_y_test = new_y_test)


if __name__ == "__main__":
    get_file_data()