import process_data
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def k_random_forest(data_x, data_y, k = 5):
    split_data_x, split_data_y = split_data(data_x, data_y, k)
    ensemble_sizes = [5, 15, 35, 35, 55, 65, 75, 85, 95, 110, 125, 140, 150, 170, 200, 225, 250, 300, 350]
    criterion = 'gini'
    max_features = int(np.sqrt(len(data_x[0])))
    validation_error_avg = []
    for n_estimator in ensemble_sizes:
        #print("-----------------------------")
        #print("\nEnsemble size is ", n_estimator)
        avg_validation_error = 0
        for fold in range(k):
            x_train,y_train, x_test, y_test = get_sets(split_data_x, split_data_y, fold)
            
            rf = RandomForestClassifier(n_estimators = n_estimator, max_features = max_features, criterion = criterion, random_state=0, max_depth= None)
            rf.fit(x_train, y_train)
            

            y_test_pred = rf.predict(x_test)
            
            fold_validation_error = 1 - accuracy_score(y_test, y_test_pred)

            avg_validation_error += fold_validation_error

            #print(f"Fold {fold+1}: validation error of fold = {fold_validation_error}")

        #print(f"\nAverage validation error = {avg_validation_error/k} \n")
        validation_error_avg.append(avg_validation_error/k)

    lowest_error_i = validation_error_avg.index(min(validation_error_avg))
    name = "Random Forest"
    print("\n**********")
    print(f"{name} with {k} folds")
    print(f"Ensemble size with lowest avg error: {ensemble_sizes[lowest_error_i]}")
    print(f"Lowest average error: {validation_error_avg[lowest_error_i]}")
    print("**********\n") 
    visualize(name, validation_error_avg, ensemble_sizes, k, 1)

    return ensemble_sizes[lowest_error_i]

def k_adaboost(data_x, data_y, k):
    split_data_x, split_data_y = split_data(data_x, data_y, k)
    ensemble_sizes = [5, 15, 35, 35, 55, 65, 75, 85, 95, 110, 125, 140, 150, 170, 200, 225, 250, 300, 350]
    validation_error_avg = []
    for n_estimator in ensemble_sizes:
        #print("-----------------------------")
        #print("\nEnsemble size is ", n_estimator)

        avg_validation_error = 0
        for fold in range(k):
            x_train,y_train, x_test, y_test = get_sets(split_data_x, split_data_y, fold)
            
            clf = AdaBoostClassifier(estimator = DecisionTreeClassifier(max_depth = 1), n_estimators = n_estimator, random_state = 0)
            clf.fit(x_train, y_train)

            y_test_pred = clf.predict(x_test)
            
            fold_validation_error = 1 - accuracy_score(y_test, y_test_pred)
            
            avg_validation_error += fold_validation_error

            #print(f"Fold {fold+1}: validation error of fold = {fold_validation_error}")

        #print(f"\nAverage validation ersror = {avg_validation_error/k} \n")
        validation_error_avg.append(avg_validation_error/k)

    lowest_error_i = validation_error_avg.index(min(validation_error_avg))
    name = "Boosted Decision Tree"
    print("\n**********")
    print(f"{name} with {k} folds")
    print(f"Ensemble size with lowest avg error: {ensemble_sizes[lowest_error_i]}")
    print(f"Lowest average error: {validation_error_avg[lowest_error_i]}")
    print("**********\n") 

    visualize(name, validation_error_avg, ensemble_sizes, k,2)
    return ensemble_sizes[lowest_error_i]


def visualize(name, validation_error_avg, ensemble_sizes, k, n):
    image_path = "./images/k_fold_cross_validation/"

    fig, ax = plt.subplots()
    ax.set_xlabel("ensemble size")
    ax.set_ylabel("Average validation error")
    ax.set_title(f"Avg validation error vs ensemble size for {name} with {k} folds" )
    ax.plot(ensemble_sizes, validation_error_avg, marker="o", drawstyle="steps-post")
    plt.savefig(f"{image_path}validation_err_vs_ensemble_size_{k}_folds_{n}.png")

def main():
    k = None
    if(len(sys.argv) == 2):
        try:
            k = int(sys.argv[1])
        except:
            print("Couldn't convert k to integer")
            sys.exit(1)
    else:
        print("Usage: python k_fold.py k")
        sys.exit(1)

    data_x, data_y = process_data.process()
    full_x_train, final_x_test, full_y_train, final_y_test = train_test_split(data_x, data_y, train_size = 0.8, random_state = 0)# this simply cuts data for training and test

    rf_ensemble_size = k_random_forest(full_x_train, full_y_train,k)
    bdt_ensemble_size = k_adaboost(full_x_train, full_y_train,k)

    final_test(rf_ensemble_size, bdt_ensemble_size, full_x_train, full_y_train, final_x_test, final_y_test)

def final_test(rf_ensemble_size, bdt_ensemble_size, x_train, y_train, x_test, y_test):
    criterion = 'gini'
    max_features = int(np.sqrt(len(x_train[0])))

    clf = AdaBoostClassifier(estimator = DecisionTreeClassifier(max_depth = 1), n_estimators = bdt_ensemble_size, random_state = 0)
    clf.fit(x_train, y_train)
    bdt_y_test_pred = clf.predict(x_test)
    bdt_test_error = 1 - accuracy_score(y_test, bdt_y_test_pred)

    rf = RandomForestClassifier(n_estimators = rf_ensemble_size, max_features = max_features, criterion = criterion, random_state=0, max_depth= None)
    rf.fit(x_train, y_train)
    rf_y_test_pred = rf.predict(x_test)    
    rf_test_error = 1 - accuracy_score(y_test, rf_y_test_pred)

    print("Final test")
    print(f"Boosted decision tree with {bdt_ensemble_size} ensemble size")
    print(f"Test error: {bdt_test_error} \n")

    print(f"Random Forest with {rf_ensemble_size} ensemble size")
    print(f"Test error: {rf_test_error} \n")

    difference = abs(rf_test_error - bdt_test_error)
    if(rf_test_error < bdt_test_error):
        print(f"Random forest has lowest test error with difference {difference}")
    else:
        print(f"Boosted decision tree has lowest test error with difference {difference}")

def split_data(data_x, data_y, k):
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

    return split_data_x, split_data_y
        

def get_sets(data_x, data_y, cur_fold):
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for fold in range(len(data_x)):#k = 5 always!
        if(cur_fold == fold):
            x_test = data_x[fold]
            y_test = data_y[fold]

        else:
            x_train.extend(data_x[fold])
            y_train.extend(data_y[fold])

    return x_train,y_train, x_test, y_test
        


if __name__ == '__main__':
    main()