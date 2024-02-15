import process_data
import math
import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def random_forest(criterion):

    data_x, data_y = process_data.process()
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size = 0.8, random_state = 0)
    num_trees = [2, 5, 6, 8, 12, 13, 15, 17, 20, 22, 25, 27, 30, 32, 35, 39, 40, 42, 45, 50, 55, 60, 62, 68, 72, 85, 93, 100]


    train_scores = []
    test_scores = []
    for estimator in num_trees:
        rf = RandomForestClassifier(criterion = criterion, n_estimators = estimator, random_state = 0, max_features = 'sqrt', bootstrap= True)
        rf.fit(x_train, y_train)
        test_score = rf.score(x_test, y_test)
        train_score = rf.score(x_train, y_train)
        train_scores.append(train_score)
        test_scores.append(test_score)
        if(len(sys.argv) > 1 and sys.argv[1] == "."):
            print("\nAccuracy with ", estimator, " # of trees")
            print("Validation set: ",test_score)
            print("Training set ", train_score)

    visualize_data([], num_trees, [], train_scores, test_scores, criterion, None, plot_num = 1, n = None)

    first_best_i = 0
    best_difference = 999999999999
    for cur_i in range(len(train_scores)):
        local_difference = abs(train_scores[cur_i] - test_scores[cur_i])
        if(local_difference < best_difference):
            best_difference = local_difference
            first_best_i = cur_i

    print("\nFirst way optimal forest with ", num_trees[first_best_i], " # of trees")
    print("Training accuracy: ", train_scores[first_best_i])
    print("Testing accuracy: ",test_scores[first_best_i])
    
    sec_best_i = test_scores.index(max(test_scores))
    print("\nSecond optimal forest with ", num_trees[sec_best_i], " # of trees")
    print("Training accuracy: ", train_scores[sec_best_i])
    print("Testing accuracy: ",test_scores[sec_best_i])

    if(criterion == 'gini'):
        test_features(num_trees[first_best_i], x_train, y_train, x_test, y_test, 1)
        test_features(num_trees[sec_best_i], x_train, y_train, x_test, y_test, 2)
        test_perc_of_data(num_trees[first_best_i], x_train, y_train, x_test, y_test, 1)
        test_perc_of_data(num_trees[sec_best_i], x_train, y_train, x_test, y_test, 2)

def test_features(num_tree, x_train, y_train, x_test, y_test, n):
    train_scores = []
    test_scores = []
    num_features = get_random_features()

    for num_feature in num_features:
        rf = RandomForestClassifier(n_estimators = num_tree, random_state = 0, max_features = num_feature, bootstrap= True)
        rf.fit(x_train, y_train)
        test_score = rf.score(x_test, y_test)
        train_score = rf.score(x_train, y_train)
        train_scores.append(train_score)
        test_scores.append(test_score)

    visualize_data(num_features,[],[], train_scores, test_scores, 'gini', num_tree, 2, n)

def test_perc_of_data(num_tree, x_train, y_train, x_test, y_test, n):
    train_scores = []
    test_scores = []
    percentages = []
    for perc in range(10,110, 10):
        new_x_train, new_y_train = process_data.cut_training_data(x_train, y_train, perc)
        rf = RandomForestClassifier(n_estimators = num_tree, random_state = 0, max_features = 'sqrt', bootstrap= True)
        rf.fit(new_x_train, new_y_train)
        test_score = rf.score(x_test, y_test)
        train_score = rf.score(x_train, y_train)

        train_scores.append(train_score)
        test_scores.append(test_score)
        percentages.append(perc)

    visualize_data([] , [], percentages, train_scores, test_scores, 'gini', num_tree, 3, n)  
    
def visualize_data(num_features ,num_trees, percentages, train_scores, test_scores, criterion, forest_size,plot_num, n = 1):
    image_path = "./images/random_forest/" + criterion

    if(plot_num == 1):#n is made for the plot selection only, to have everything in one function   
        fig, ax = plt.subplots()
        ax.set_xlabel("number of trees")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs forest size(number of trees) for training and test sets")
        ax.plot(num_trees, train_scores, marker="o", label="train", drawstyle="steps-post")
        ax.plot(num_trees, test_scores, marker="o", label="test", drawstyle="steps-post")
        ax.legend()
        plt.savefig(image_path + "_accr_vs_forest_size.png")
    
    elif(plot_num == 2):
        fig, ax = plt.subplots()
        ax.set_xlabel("number of features")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs number of features for training and test sets with " + str(forest_size) + " trees")
        ax.plot(num_features, train_scores, marker="o", label="train", drawstyle="steps-post")
        ax.plot(num_features, test_scores, marker="o", label="test", drawstyle="steps-post")
        ax.legend()
        plt.savefig(image_path + "_accr_vs_num_of_features_" + str(n) + ".png")

    elif(plot_num == 3):
        fig, ax = plt.subplots()
        ax.set_xlabel("percentage of original data used")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs % of data used for training and test sets with " + str(forest_size) + " trees")
        ax.plot(percentages, train_scores, marker="o", label="train", drawstyle="steps-post")
        ax.plot(percentages, test_scores, marker="o", label="test", drawstyle="steps-post")
        ax.legend()
        plt.savefig(image_path + "_accr_vs_percentage_of_data_" + str(n) + ".png")


def get_random_features():
    features = process_data.get_features()
    num_features = []
    for num in range(2, int(math.sqrt(len(features)) * 2)):
        num_features.append(num)

    return num_features

def main():
    print("Entropy")
    random_forest(criterion= "entropy")
    print("\n\n\nGini")
    random_forest(criterion= "gini")

if __name__ == '__main__':
    main()