import numpy as np

def split_data(data_x, data_y, k):
    #Splits data in to k folds!
    fold_len = len(data_x) // k
    split_data_x = []
    split_data_y = []

    num_elements = 0
    cur_fold = 0
    fold_x = []
    fold_y = []
    for i in range(len(data_x)):
        if(num_elements == fold_len):
            if(cur_fold != k - 1):#not last array
                split_data_x.append(fold_x)
                split_data_y.append(fold_y)
                num_elements = 0
                cur_fold += 1
                fold_x = []
                fold_y = []
    
        fold_x.append(data_x[i])
        fold_y.append(data_y[i])
        num_elements +=1
    
    split_data_x.append(fold_x)
    split_data_y.append(fold_y)

    
    return np.array(split_data_x),np.array(split_data_y)
        

def get_sets(data_x, data_y, cur_fold):
    #gives current train and test sets depending on the fold
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for fold in range(len(data_x)):
        if(cur_fold == fold):
            x_test = data_x[fold]
            y_test = data_y[fold]

        else:
            x_train.extend(data_x[fold])
            y_train.extend(data_y[fold])

    return np.array(x_train),np.array(y_train), np.array(x_test), np.array(y_test)
        
