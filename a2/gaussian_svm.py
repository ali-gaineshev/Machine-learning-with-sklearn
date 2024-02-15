import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import k_fold

def svc_rbf():
    try:
        data = np.load("data/arrays_data.npz")
    except:
        print("Please run process_data.py first!")
        exit(1)

    x_train = data["new_x_train"]
    y_train = data["new_y_train"]
    x_test = data["new_x_test"]
    y_test = data["new_y_test"]

    k = 5
    regul_param = get_paramater(0.0001, 3, 13)
    gamma_values = [0.0001, 0.001]
    more_gamma_values = get_paramater(0.01, 3, 7)
    gamma_values.extend(more_gamma_values)
    avg_vald_errs = []

    split_x_train, split_y_train = k_fold.split_data(x_train,y_train,k)
    for param in regul_param:
        print("Training...")
        vald_error = 0
        for fold in range(k):
            new_x_train, new_y_train, x_vald, y_vald = k_fold.get_sets(split_x_train, split_y_train, fold)
            clf = svm.SVC(kernel = 'rbf', C = param, random_state=0)
            clf.fit(new_x_train, new_y_train)

            vald_error += (1 - accuracy_score(y_vald, clf.predict(x_vald)))
        
        avg_vald_errs.append(vald_error/k)


    vald_err_i = avg_vald_errs.index(min(avg_vald_errs))
    tuned_c_param = regul_param[vald_err_i]

    print("\n\nParams, ", regul_param)
    print("LOWEST with 5 folds! validation sets ")
    print("Valdidation Error: ",avg_vald_errs[vald_err_i])
    print("Accuracy vald set", (1 - avg_vald_errs[vald_err_i]))
    print("Tuned C value: ", tuned_c_param)
    print("Default gamma: \"scale\". "
         +"Note from documentation: if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma")

    visualize(avg_vald_errs, [],[], regul_param, None ,1)

    avg_vald_errs = []
    for gamma in gamma_values:
        vald_error = 0
        for fold in range(k):
            new_x_train, new_y_train, x_vald, y_vald = k_fold.get_sets(split_x_train, split_y_train, fold)
            clf = svm.SVC(kernel = 'rbf', C = tuned_c_param, gamma= gamma, random_state=0)
            clf.fit(new_x_train, new_y_train)

            vald_error += (1 - accuracy_score(y_vald, clf.predict(x_vald)))
        
        avg_vald_errs.append(vald_error/k)


    vald_err_i = avg_vald_errs.index(min(avg_vald_errs))
    tuned_gamma_param = gamma_values[vald_err_i]

    print("\n\n Gamma Params, ", gamma_values)
    print("LOWEST with 5 folds! tuned C. validation sets ")
    print("Valdidation Error: ",avg_vald_errs[vald_err_i])
    print("Accuracy vald set", (1 - avg_vald_errs[vald_err_i]))
    print("Tuned C value: ", tuned_c_param)
    print("Tuned gamma: ", tuned_gamma_param)

    visualize(avg_vald_errs, [], gamma_values, regul_param, None , 2)

    test_errs = []
    train_errs = []

    for gamma in gamma_values:
        clf = svm.SVC(kernel = 'rbf', C = tuned_c_param, gamma = gamma , random_state=0)
        clf.fit(x_train, y_train)

        test_error = 1 - accuracy_score(y_test, clf.predict(x_test))
        train_error = 1 - accuracy_score(y_train, clf.predict(x_train))

        test_errs.append(test_error)
        train_errs.append(train_error)

    test_err_i = test_errs.index(min(test_errs))
    optimal_gamma_param = gamma_values[test_err_i]


    print(f"\nLOWEST with test set! Match with previous tuned gamma = {tuned_gamma_param == optimal_gamma_param}")
    print("Tuned C param: ", tuned_c_param)
    print("Previous gamma: ", tuned_gamma_param)
    print("Test err with prev gamma: ", test_errs[vald_err_i])
    print("Train err with prev gamma: ", train_errs[vald_err_i])
    print("Test acc with prev gamma: ", (1 - test_errs[vald_err_i]), "\n")

    print("Lowest Test Error: ",test_errs[test_err_i])
    print("Lowest Train error: ", train_errs[test_err_i])
    print("Highest Accuracy test set", (1 - test_errs[test_err_i]))
    print("Optimal gamma param from test setL ", optimal_gamma_param)

    visualize(test_errs, train_errs, gamma_values, regul_param, tuned_c_param, 3)

    # This is initial plan, but it took too much time 
    """
    min_test_error = 1.0
    min_c_i = 0
    min_y_i = 0
    all_test_errs = []
    i = 0
    for gamma in gamma_values:
        test_errs = []
        for c in regul_param:
            clf = svm.SVC(kernel = 'rbf', C = c, gamma = gamma , random_state=0)
            clf.fit(x_train, y_train)

            test_error = 1 - accuracy_score(y_test, clf.predict(x_test))
            #train_error = 1 - accuracy_score(y_train, clf.predict(x_train))

            test_errs.append(test_error)

        cur_min_i = test_errs.index(min(test_errs))
        if(min_test_error > test_errs[cur_min_i]):
            min_test_error = test_errs[cur_min_i]
            min_c_i = cur_min_i
            min_y_i = i

        i += 1

        all_test_errs.append(test_errs)

    print("\nActual lowest. ")
    print("Test set error ", all_test_errs[min_y_i][min_c_i])
    print("C value: ", regul_param[min_c_i])
    print("Gamma values: ", gamma_values[min_y_i])
    """
    
def visualize(vald_test_error, train_err, gamma_values, c_values, tuned_C ,n):
    image_path = "./images/gaussian_svm/"
    small_s_test_error = get_first_fifteen(vald_test_error)
    small_s_train_error = get_first_fifteen(train_err)
    small_c_values = get_first_fifteen(c_values)
    if( n == 1):
        fig, ax = plt.subplots(figsize = (12,6))
        ax.set_xlabel("C values")
        ax.set_ylabel("Average validation errors")
        ax.set_title(f"Average validation error vs C values" )
        ax.plot(c_values,vald_test_error, marker="o", label = 'validation', drawstyle="steps-post", linestyle = '-')
        ax.legend()
        fig.tight_layout()
        plt.savefig(f"{image_path}avg_validation_err_vs_c_values.png")
        
        fig, ax = plt.subplots()
        ax.set_xlabel("C values (first 15)")
        ax.set_ylabel("Average validations error")
        ax.plot(small_c_values,small_s_test_error, marker="o", label = 'validation', drawstyle="steps-post", linestyle = '-')
        ax.legend()
        fig.tight_layout()
        plt.savefig(f"{image_path}avg_validation_err_vs_c_values_smaller.png")

    if (n == 2):
        fig, ax = plt.subplots(figsize = (12,6))
        ax.set_xlabel("Gamma values")
        ax.set_ylabel("Average validation errors")
        ax.set_title(f"Average validation error vs gamma values with C = {tuned_C}" )
        ax.plot(gamma_values,vald_test_error, marker="o", label = 'validation', drawstyle="steps-post", linestyle = '-')
        ax.legend()
        fig.tight_layout()
        plt.savefig(f"{image_path}avg_validation_err_vs_gamma_values.png")
        

    if (n == 3):
        fig, ax = plt.subplots(figsize = (12,6))
        ax.set_xlabel("Gamma values")
        ax.set_ylabel("Test/Train errors")
        ax.set_title(f"Test/Train error vs gamma values with C = {tuned_C}" )
        ax.plot(gamma_values,vald_test_error, marker="o", label = 'test', drawstyle="steps-post", linestyle = '-')
        ax.plot(gamma_values,train_err, marker="o", label = 'train', drawstyle="steps-post", linestyle = '-')
        ax.legend()
        fig.tight_layout()
        plt.savefig(f"{image_path}test_train_errs_vs_gamma_values.png")



def get_first_fifteen(array):
    counter = 14
    new_array = []
    for i in range(len(array)):
        if(counter == i):
            break
        new_array.append(array[i])

    return new_array

def get_paramater(C, B, iters):
    result = []
    for exp in range(iters):
        result.append(C * pow(B, exp))
    return result

if __name__ == '__main__':
    svc_rbf()