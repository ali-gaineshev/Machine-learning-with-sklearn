This is made for assignment 1 for SENG474 A1. It consists of 4 main files to run:
1. Decision Tree : decision_tree.py
2. Random Forest: random_forest.py
3. Boosted Decision Tree (using Adaboost): adaboost.py
4. K-fold cross validation for random forest and boosted decision tree: k_fold.py

Python version that was used to implement: 3.10.7

1. To run Decision tree file:
python decision_tree.py

2. To run Random Forest file:
python random_forest.py .
Note: last argument is optional, it just adds extra print statements. Running "python random_forest.py" is considered cleaner

3. To run boosted decision tree
python adaboost.py

4. To run k fold cross validation:
python k_fold.py k
where k is the fold number

All of the images that will be generated should be in "images/" folder with corresponding model as a sub directory

Additionally, "process_data.py" should be in the same folder and leave "data" folder as well. All of the images that were used in the report are in in "images" folder, except for random forests. The program was accidentally ran again and images from the report were lost. 

Libraries/packages used:
sklearn
matplotlib.pyplot
random
math
numpy
sys

