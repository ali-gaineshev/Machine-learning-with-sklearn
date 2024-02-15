import numpy as np
import k_fold
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def visualize(validation_error_avg, train_err, layer_size, alphas, max_iters, tuned_layer_size, tuned_alpha, n):
    image_path = "./images/neural_network/"

    if( n == 1):
        x_values = range(len(layer_size))

        fig, ax = plt.subplots(figsize = (12,6))
        ax.set_xlabel("Layer size")
        ax.set_ylabel("Average validation error")
        ax.set_title(f"Avg validation error vs layer size" )
        ax.plot(x_values, validation_error_avg, marker="o", label = 'vald', drawstyle="steps-post", linestyle = '-')
        x_labels = [str(t) for t in layer_size]
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels, fontsize = 6)

        plt.savefig(f"{image_path}validation_err_vs_layer_size.png")

    if(n == 2):
        fig, ax = plt.subplots()
        ax.set_xlabel("Alphas")
        ax.set_ylabel("Error")
        ax.set_title(f"Error vs alphas with {tuned_layer_size} layer size" )
        
        ax.plot(range(len(alphas)), validation_error_avg, label = 'test', marker="o", linestyle='-')
        ax.plot(range(len(alphas)), train_err, label = 'train', marker="o", linestyle='-')
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels(alphas, fontsize=8)

        ax.legend()
        plt.savefig(f"{image_path}err_vs_alphas_{tuned_layer_size}_layer_size.png")

    if(n == 3):
        fig, ax = plt.subplots()
        ax.set_xlabel("Max iterations")
        ax.set_ylabel("Error")
        ax.set_title(f"Error vs max epochs with {tuned_layer_size} l.s and {tuned_alpha} alpha" )
        ax.plot(max_iters, validation_error_avg, marker="o", drawstyle="steps-post", label = 'test', linestyle = '-')
        ax.plot(max_iters, train_err, marker="o", drawstyle="steps-post", label = 'train', linestyle = '-')

        ax.legend()
        plt.savefig(f"{image_path}err_vs_iters_{tuned_layer_size}_layer_size.png")

def visualize_last(xy1,xy2,xy3,xy4, err1, err2, err3, err4, t_err1, t_err2, t_err3, t_err4):

    image_path = "./images/neural_network/"
    layer_size = [xy1,xy2,xy3,xy4]
    err = [err1,err2,err3,err4]
    t_err = [t_err1, t_err2, t_err3, t_err4]#train error
    for i in range(4):
        x_values = range(len(layer_size[i]))
        fig, ax = plt.subplots(figsize = (8,6))
        ax.set_xlabel("Layer size")
        ax.set_ylabel("Error")
        ax.set_title(f"Error vs layer size" )
        ax.plot(x_values, err[i], marker="o", drawstyle="steps-post", label = 'test', linestyle = '-')
        ax.plot(x_values, t_err[i], marker="o", drawstyle="steps-post", label = 'train', linestyle = '-')
        x_labels = [str(t) for t in layer_size[i]]
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels, fontsize = 8, rotation = 45)

        plt.tight_layout()
        ax.legend()
        plt.savefig(f"{image_path}err_vs_layer_size_{i+1}.png")

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

    #LAYER SIZE EXPERIMENT
    error_avg = []
    layer_sizes = [(65, 15), (90, 80), (100, 50), (150, 100), (170, 70),  (190, 65), 
                   (200,200), (200, 50), (203, 53), (225,50), (225,100), (250, 70), (280,70),(300, 100)]

    split_x_train, split_y_train = k_fold.split_data(x_train,y_train,5)
    for layer_size in layer_sizes:
        validation_error = 0
        for fold in range(5):
            new_x_train, new_y_train, x_vald, y_vald = k_fold.get_sets(split_x_train, split_y_train, fold)
            mlp = MLPClassifier(hidden_layer_sizes=layer_size, max_iter=200, activation = 'relu', 
                                solver='adam',random_state=0, early_stopping=True, n_iter_no_change=5)
            mlp.fit(new_x_train, new_y_train)

            y_pred = mlp.predict(x_vald)
            error = 1 - accuracy_score(y_vald, y_pred)
            validation_error += error
        error_avg.append(validation_error/5)
        

    lowest_error_i = error_avg.index(min(error_avg))
    tuned_lay_size = layer_sizes[lowest_error_i]
    print("Lowest layer size tuning!")
    print("Hidden layer: ", layer_sizes[lowest_error_i])
    print("Average valid error: ",error_avg[lowest_error_i])
    print("Accuracy ", (1 - error_avg[lowest_error_i]))

    visualize(error_avg, [], layer_sizes, [], [], tuned_lay_size, None, 1)

    #REGULARIZATION EXPERIMENT
    alpha_values = [0.0000001,0.000001, 0.00001, 0.0001, 0.0001, 0.001, 0.01, 1.0]
    error_alphas_avg = []
    train_err = []
    for alpha in alpha_values:
        mlp = MLPClassifier(hidden_layer_sizes = tuned_lay_size, max_iter=200, activation = 'relu',
                            solver='adam',random_state=0, alpha=alpha, early_stopping=True, n_iter_no_change=5)
        mlp.fit(x_train, y_train)

        test_error = 1 - accuracy_score(y_test, mlp.predict(x_test))
        train_error = 1 - accuracy_score(y_train, mlp.predict(x_train))
        error_alphas_avg.append(test_error)
        train_err.append(train_error)

    lowest_error_i = error_alphas_avg.index(min(error_alphas_avg))
    lowest_t_err_i = train_err.index(min(train_err))
    tuned_alpha = alpha_values[lowest_error_i]

    print("\nLowest with alpha!")
    print("Layer size: ",tuned_lay_size)
    print("Alpha: ", tuned_alpha)
    print("Test Error: ",error_alphas_avg[lowest_error_i])
    print("Train error: ", train_err[lowest_t_err_i])
    print("Accuracy test set", (1 - error_alphas_avg[lowest_error_i]))

    visualize(error_alphas_avg, train_err, [], alpha_values,[], tuned_lay_size, tuned_alpha, 2)
    
    #ITERATION Experiment
    max_iters = [x for x in range(50, 610, 50)]
    test_error_iter = []
    train_err_iter = []
    for max_iter in max_iters: 
        mlp = MLPClassifier(hidden_layer_sizes= tuned_lay_size, max_iter=max_iter, activation = 'relu',
                                solver='adam',random_state=0, early_stopping= True, n_iter_no_change=5, alpha=tuned_alpha)
        mlp.fit(x_train, y_train)

        test_error = 1 - accuracy_score(y_test, mlp.predict(x_test))
        train_error = 1 - accuracy_score(y_train, mlp.predict(x_train))
        test_error_iter.append(test_error)
        train_err_iter.append(train_error)
    
    lowest_error_i = test_error_iter.index(min(test_error_iter))
    lowest_t_err_i = train_err_iter.index(min(train_err))
    tuned_iter = max_iters[lowest_error_i]
    if(tuned_iter == 50):
        tuned_iter = 200
    

    print("\nLowest with alpha and max iters!!")
    print("Layer size: ",tuned_lay_size)
    print("Alpha: ", tuned_alpha)   
    print("Max iters: ", tuned_iter)
    print("Test error: ",test_error_iter[lowest_error_i])
    print("Train error: ", train_err_iter[lowest_t_err_i])
    print("Accuracy test set", (1 - test_error_iter[lowest_error_i]))

    visualize(test_error_iter, train_err_iter, [], [], max_iters, tuned_lay_size, tuned_alpha, 3)

    #ANOTHER LAYER SIZE EXPERIMENT
    x = tuned_lay_size[0]
    y = tuned_lay_size[1]

    x1 = [x for x in range(x, x + 35, 3)]
    y1 = [y for y in range(y,y + 35, 3)]

    x2 = [x for x in range(x, x - 35, -3)]
    y2 = [y for y in range(y,y - 35, -3)]

    x3 = [x for x in range(x, x - 35, -3)]
    y3 = [y for y in range(y,y + 35, 3)]

    x4 = [x for x in range(x, x + 35, 3)]
    y4 = [y for y in range(y,y - 35, -3)]

    xy1 = []
    xy2 = []
    xy3 = []
    xy4 = []

    for_len = min(len(x1), len(y1), len(x2), len(y2))

    for i in range(for_len):
        a = (x1[i],y1[i])
        b = (x2[i],y2[i])
        c = (x3[i],y3[i])
        d = (x4[i],y4[i])
        xy1.append(a)
        xy2.append(b)
        xy3.append(c)
        xy4.append(d)

    print("\nModel so far")
    print("Tuned alpha: ", tuned_alpha)
    print("Tuned Layer size: ", tuned_lay_size)
    print("Tuned max epochs: ", tuned_iter)
    print("\nTUNING MORE\n")
    err1 = []
    err2 = []
    err3 = []
    err4 = []
    t_err1 = []
    t_err2 = []
    t_err3 = []
    t_err4 = []

    for i in range(for_len):
        mlp = MLPClassifier(hidden_layer_sizes = xy1[i], max_iter=tuned_iter, activation = 'relu',
                        solver='adam',random_state=0, early_stopping=True, n_iter_no_change=5, alpha=tuned_alpha)
        mlp.fit(x_train, y_train)
        test_error = 1 - accuracy_score(y_test, mlp.predict(x_test))
        train_err = 1 - accuracy_score(y_train, mlp.predict(x_train))
        err1.append(test_error)
        t_err1.append(train_err)

    for i in range(for_len):
        mlp = MLPClassifier(hidden_layer_sizes = xy2[i], max_iter=tuned_iter, activation = 'relu',
                        solver='adam',random_state=0, early_stopping=True, n_iter_no_change=5, alpha=tuned_alpha)
        mlp.fit(x_train, y_train)
        test_error = 1 - accuracy_score(y_test, mlp.predict(x_test))
        train_err = 1 - accuracy_score(y_train, mlp.predict(x_train))
        err2.append(test_error)
        t_err2.append(train_err)

    for i in range(for_len):
        mlp = MLPClassifier(hidden_layer_sizes = xy3[i], max_iter=tuned_iter, activation = 'relu',
                        solver='adam',random_state=0, early_stopping=True, n_iter_no_change=5, alpha=tuned_alpha)
        mlp.fit(x_train, y_train)
        test_error = 1 - accuracy_score(y_test, mlp.predict(x_test))
        train_err = 1 - accuracy_score(y_train, mlp.predict(x_train))
        err3.append(test_error)
        t_err3.append(train_err)

    for i in range(for_len):
        mlp = MLPClassifier(hidden_layer_sizes = xy4[i], max_iter=tuned_iter, activation = 'relu',
                        solver='adam',random_state=0, early_stopping=True, n_iter_no_change=5, alpha=tuned_alpha)
        mlp.fit(x_train, y_train)
        test_error = 1 - accuracy_score(y_test, mlp.predict(x_test))
        train_err = 1 - accuracy_score(y_train, mlp.predict(x_train))
        err4.append(test_error)
        t_err4.append(train_err)    

    l_xy1 = err1.index(min(err1))
    l_xy2 = err2.index(min(err2))
    l_xy3 = err3.index(min(err3))
    l_xy4 = err4.index(min(err4))

    print("\nTest accuracies:")
    print(f"Accuracy: {1 - err1[l_xy1]} with {xy1[l_xy1]}")
    print(f"Accuracy: {1 - err2[l_xy2]} with {xy2[l_xy2]}")
    print(f"Accuracy: {1 - err3[l_xy3]} with {xy3[l_xy3]}")
    print(f"Accuracy: {1 - err4[l_xy4]} with {xy4[l_xy4]}")

    visualize_last(xy1,xy2,xy3,xy4, err1, err2, err3, err4, t_err1, t_err2, t_err3, t_err4)


if __name__ == '__main__':
    main()