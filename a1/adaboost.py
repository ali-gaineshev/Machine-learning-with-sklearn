from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import process_data

def adaboost():
    x_data, y_data = process_data.process()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)

    n_estimators = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    train_scores = [[],[]]
    test_scores = [[],[]]

    depths = [3,5]
    best_f_i = [0,0]
    best_s_i = [0,0]

    highest_test_score = [0,0]
    cur_depth_i = 0
    cur_estim_i = 0
    for depth in [3, 5]:
        local_difference = 0
        best_difference = 999999999
        cur_estim_i = 0
        for estimator in n_estimators:
            base_estimator = DecisionTreeClassifier(max_depth = depth, random_state=0)
            clf = AdaBoostClassifier(estimator = base_estimator, n_estimators = estimator,  random_state = 0)
            
            clf.fit(x_train, y_train)
            test_score = clf.score(x_test, y_test)
            train_score = clf.score(x_train, y_train)

            local_difference = abs(train_score - test_score)
            if(local_difference < best_difference):
                best_difference = local_difference
                best_f_i[cur_depth_i] = cur_estim_i
                

            if(test_score > highest_test_score[cur_depth_i]):
                highest_test_score[cur_depth_i] = test_score
                best_s_i[cur_depth_i] = cur_estim_i
                

            train_scores[cur_depth_i].append(train_score)
            test_scores[cur_depth_i].append(test_score)

            cur_estim_i += 1
        cur_depth_i += 1
    
    make_plots(depths[0], n_estimators, [], train_scores[0], test_scores[0], 1)
    make_plots(depths[1], n_estimators, [], train_scores[1], test_scores[1], 1)

    print("Depth: ", depths[0])
    print("Training scores: ",train_scores[0])
    print("Test scores: ", test_scores[0])
    print("\n----------------------------")
    print("Frist optimal tree")
    print("Estimator: ", n_estimators[best_f_i[0]])
    print("Training score: ", train_scores[0][best_f_i[0]])
    print("Test score: ", test_scores[0][best_f_i[0]])
    print("----------------------------")
    print("Second optimal tree")
    print("Estimator: ", n_estimators[best_s_i[0]])
    print("Training score: ", train_scores[0][best_s_i[0]])
    print("Test score: ", test_scores[0][best_s_i[0]])

    print("\n")
    print("Depth: ", depths[1])
    print("Training scores: ",train_scores[1])
    print("Test scores: ", test_scores[1])
    print("\n----------------------------")
    print("Frist optimal tree")
    print("Estimator: ", n_estimators[best_f_i[1]])
    print("Training score: ", train_scores[1][best_f_i[1]])
    print("Test score: ", test_scores[1][best_f_i[1]])
    print("----------------------------")
    print("Second optimal tree")
    print("Estimator: ", n_estimators[best_s_i[1]])
    print("Training score: ", train_scores[1][best_s_i[1]])
    print("Test score: ", test_scores[1][best_s_i[1]])

    if(test_scores[0][best_s_i[0]] > test_scores[1][best_s_i[1]]):
        test_with_data_size(depths[0], n_estimators[best_s_i[0]], x_train, y_train, x_test, y_test)
    else:
        test_with_data_size(depths[1], n_estimators[best_s_i[1]], x_train, y_train, x_test, y_test)


def test_with_data_size(depth, n_estimator, x_train, y_train, x_test, y_test):

    train_scores = []
    test_scores = []
    percentages = []
    for perc in range(10, 110, 10):
        new_x_train, new_y_train = process_data.cut_training_data(x_train,y_train, perc)
        base_estimator = DecisionTreeClassifier(max_depth = depth)
        clf = AdaBoostClassifier(estimator = base_estimator, n_estimators = n_estimator, random_state = 0)
        clf.fit(new_x_train, new_y_train)
        test_score = clf.score(x_test, y_test)
        train_score = clf.score(x_train, y_train)

        train_scores.append(train_score)
        test_scores.append(test_score)
        percentages.append(perc)

    make_plots(depth, [n_estimator], percentages, train_scores, test_scores, 2)

def make_plots(depth, estimators, perc, train_scores, test_scores, n = 3):
    image_path = "./images/adaboost/depth_" + str(depth)

    if(n == 1):
        fig, ax = plt.subplots()
        ax.set_xlabel("number of iterations(week hypothesis)")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs number of iterations with depth " + str(depth))
        ax.plot(estimators, train_scores, marker="o", label="train", drawstyle="steps-post")
        ax.plot(estimators, test_scores, marker="o", label="test", drawstyle="steps-post")
        ax.legend()
        plt.savefig(image_path + "_accr_vs_num_of_iterations.png")

    elif(n == 2):
        fig, ax = plt.subplots()
        ax.set_xlabel("percent of training data used")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs % of train data with depth " + str(depth) + " and # of iterations " + str(estimators[0]))
        ax.plot(perc, train_scores, marker="o", label="train", drawstyle="steps-post")
        ax.plot(perc, test_scores, marker="o", label="test", drawstyle="steps-post")
        ax.legend()
        plt.savefig(image_path + "_accr_vs_perc_of_train_data_used.png")

if __name__ == '__main__':
    adaboost()