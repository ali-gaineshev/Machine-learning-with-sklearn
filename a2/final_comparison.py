import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def main():
    try:
        data = np.load("data/arrays_data.npz")
    except:
        print("Please run process_data.py first!")
        exit(1)

    x_train = data["new_x_train"]
    y_train = data["new_y_train"]
    x_test = data["new_x_test"]
    y_test = data["new_y_test"]


    models = ["Linear SVM", "Gaussian SVM", "Neural Network"]
    test_errs_1 = []
    test_errs_2 = []
    test_errs_3 = []

    C_linear_tuned = 0.024300000000000002
    C_linear_best = 0.0081

    C_gaus_tuned = 1.9683000000000002
    
    gamma_gaus_tuned = 0.03
    gamma_gaus_best = 0.01

    layer_tuned = (250,70)
    layer_best = (235, 55)

    alpha_best = 0.00001

    for i in range(len(models)):
        if(i == 0):
            #tuned
            clf = svm.SVC(kernel = 'linear', C = C_linear_tuned, random_state=0)
            clf.fit(x_train, y_train)
            test_error = 1 - accuracy_score(y_test, clf.predict(x_test))
            test_errs_1.append(test_error)

            #optimal
            clf = svm.SVC(kernel = 'linear', C = C_linear_best, random_state=0)
            clf.fit(x_train, y_train)
            test_error = 1 - accuracy_score(y_test, clf.predict(x_test))
            test_errs_2.append(test_error)

#           #default
            clf = svm.SVC(kernel = 'linear', random_state=0)
            clf.fit(x_train, y_train)
            test_error = 1 - accuracy_score(y_test, clf.predict(x_test))
            test_errs_3.append(test_error)


    
        if(i == 1):
                #tuned
                clf = svm.SVC(kernel = 'rbf', C = C_gaus_tuned, gamma = gamma_gaus_tuned, random_state=0)
                clf.fit(x_train, y_train)
                test_error = 1 - accuracy_score(y_test, clf.predict(x_test))
                test_errs_1.append(test_error)

                #optimal
                clf = svm.SVC(kernel = 'rbf', C = C_gaus_tuned, gamma = gamma_gaus_best, random_state=0)
                clf.fit(x_train, y_train)
                test_error = 1 - accuracy_score(y_test, clf.predict(x_test))
                test_errs_2.append(test_error)

    #           #default
                clf = svm.SVC(kernel = 'rbf', random_state=0)
                clf.fit(x_train, y_train)

                test_error = 1 - accuracy_score(y_test, clf.predict(x_test))
                test_errs_3.append(test_error)


        if(i == 2):
            mlp = MLPClassifier(hidden_layer_sizes= layer_tuned, max_iter=200, activation = 'relu',
                                    solver='adam',random_state=0, early_stopping= True, n_iter_no_change=5)
            mlp.fit(x_train, y_train)
            test_error = 1 - accuracy_score(y_test, mlp.predict(x_test))
            test_errs_1.append(test_error)

            mlp = MLPClassifier(hidden_layer_sizes= layer_best, max_iter=200, activation = 'relu',
                                    solver='adam',random_state=0, early_stopping= True, n_iter_no_change=5, alpha= alpha_best)
            mlp.fit(x_train, y_train)
            test_error = 1 - accuracy_score(y_test, mlp.predict(x_test))
            test_errs_2.append(test_error)

            mlp = MLPClassifier(max_iter=200, activation = 'relu',
                                    solver='adam',random_state=0, early_stopping= True, n_iter_no_change=5)
            mlp.fit(x_train, y_train)
            test_error = 1 - accuracy_score(y_test, mlp.predict(x_test))
            test_errs_3.append(test_error)
    
    print(test_errs_1)
    print(test_errs_2)
    print(test_errs_3)
    visualize(test_errs_1, test_errs_2, test_errs_3, models)

def visualize(err1, err2, err3, models):
    image_path = "./images/"

    fig, ax = plt.subplots()
    ax.set_xlabel("Models")
    ax.set_ylabel("Test errors")
    ax.set_title(f"Test errors vs model" )
    ax.plot(models, err1, marker="o", label = 'Tuned', linestyle = '-')
    ax.plot(models, err2, marker="o", label = 'Lowest Test Error', linestyle = '-')
    ax.plot(models, err3, marker="o", label = 'Default', linestyle = '-')
    ax.legend()
    fig.tight_layout()
    plt.savefig(f"{image_path}final_comparison.png")
    

if __name__ == '__main__':
    main()
