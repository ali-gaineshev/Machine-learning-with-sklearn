from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

import process_data

def decision_tree(criterion):
    
    #clf = DecisionTreeClassifier(criterion= "entropy")
    data_x, data_y = process_data.process()
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size = 0.8)
    
    #https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py
    init_clf = DecisionTreeClassifier(criterion = criterion, random_state=0)
    

    #pre_prunning(init_clf, criterion, x_train, y_train)

    path = init_clf.cost_complexity_pruning_path(x_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    clf = DecisionTreeClassifier()
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(criterion = criterion, random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(x_train, y_train)
        clfs.append(clf)

    #last tree has only 1 node 
    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]
    impurities = impurities[:-1]

    train_scores = [clf.score(x_train, y_train) for clf in clfs]
    test_scores = [clf.score(x_test, y_test) for clf in clfs]
    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]

    cur_i = 0
    best_i = 0
    best_difference = 999999999999
    for i in range(len(train_scores)):
        local_difference = abs(train_scores[cur_i] - test_scores[cur_i])
        if(local_difference < best_difference):
            best_difference = local_difference
            best_i = cur_i
        cur_i += 1

    best_alpha = ccp_alphas[best_i]
    print("\nSummary:")
    print("\n----------------Best tree number 1 with lowest differences between scores\n")
    print("First Best alpha: ", best_alpha)
    print("Difference in accuracy: ",best_difference)
    print("Training accuracy: ", train_scores[best_i])
    print("Testing accuracy: ",test_scores[best_i])
    print("Depth: ",depth[best_i])
    print("Node count: ",node_counts[best_i]) 
    print("Impurity: ",impurities[best_i]) 

    sec_best_i = test_scores.index(max(test_scores))
    sec_best_alpha = ccp_alphas[sec_best_i]
    print("\n----------------Best tree number 2 with highest score from testing validation set\n")
    print("Second Best alpha: ", sec_best_alpha)
    print("Training accuracy: ", train_scores[sec_best_i])
    print("Testing score: ", test_scores[sec_best_i])
    print("Depth: ",depth[sec_best_i])
    print("Node count: ",node_counts[sec_best_i]) 

    make_figures(train_scores, test_scores, ccp_alphas, node_counts, depth, criterion)
    visualize_tree(clfs[best_i], criterion, 1)
    visualize_tree(clfs[sec_best_i],criterion, 2)
    best_tree_experiment(x_train, y_train, x_test, y_test, criterion, best_alpha, 1)
    best_tree_experiment(x_train, y_train, x_test, y_test, criterion, sec_best_alpha, 2)

def best_tree_experiment(x_train, y_train, x_test, y_test, criterion, best_alpha, n):
    if(n == 1):
        print("\nTesting with different sample size for best tree number 1\n")
    else:
        print("\nTesting with different sample size for best tree number 2\n")
    train_scores = []
    test_scores = []
    percentages = []
    
    for perc in range(10,110, 10):
        new_x_train, new_y_train = process_data.cut_training_data(x_train, y_train, perc)
        new_clf = DecisionTreeClassifier(criterion = criterion, random_state=0, ccp_alpha= best_alpha)
        new_clf.fit(new_x_train,new_y_train)
        test_score = new_clf.score(x_test, y_test)
        train_score = new_clf.score(x_train, y_train)
        print("Accuracy: " + str(test_score) + " from ",len(new_x_train), " samples or ", perc, " %  of original sample size:")
        
        percentages.append(perc)
        train_scores.append(train_score)
        test_scores.append(test_score)

    image_path = "./images/decision_tree/" + criterion

    fig, ax = plt.subplots()
    ax.set_xlabel("original training size % used")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs training size with alpha = " + str(round(best_alpha, 4)) +" ." + criterion)
    ax.plot(percentages, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(percentages, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.savefig(image_path + "_accr_vs_training_size_" + str(n) + ".png")




def visualize_tree(clf, criterion, n):
    if(clf.tree_.node_count > 50):
        dpi = 1000
        figsize = (8,8)
    else:
        dpi = 300
        figsize = (5,5)
    features = process_data.get_features()
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize= figsize, dpi=dpi)
    tree.plot_tree(clf, filled=True, feature_names=features,class_names = ["Spam", "Not spam"])
    fig.savefig("./images/decision_tree/" + criterion + "_optimal_tree_" + str(n) + ".png")
    plt.close(fig)


def make_figures(train_scores, test_scores, ccp_alphas, node_counts, depth, criterion):
    image_path = "./images/decision_tree/" + criterion

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and test sets. " + criterion)
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.savefig(image_path + "_accr_vs_alpha.png")

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("number of nodes")
    ax.set_title("Number of nodes vs alpha. " + criterion)
    ax.plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
    ax.plot()
    plt.savefig(image_path + "_nodes_vs_alpha.png")


    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("depth")
    ax.set_title("Depth vs alpha. " + criterion)
    ax.plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
    plt.savefig(image_path + "_depth_vs_alpha.png")



def main():
    print("Entropy")
    decision_tree(criterion= "entropy")
    print("\n\n\nGini")
    decision_tree(criterion= "gini")

if __name__ == '__main__':
    main()