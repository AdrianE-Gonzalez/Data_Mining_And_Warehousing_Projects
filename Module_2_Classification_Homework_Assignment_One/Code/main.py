# You shall use the training data set to build your decision tree
# and then use the testing data set to evaluate your decision tree. 
# You need to report classification accuracy using a bar chart
# and compare it with AdaBoost classification which is given in Part II. 

# Part I
# Binary Tree: 0='nothing in hand'; (1-9)='something in hand'

# Part II
# classifier: AdaBoost
# use a perceptron to build a weak classifier at each iteration of AdaBoost
# Run 15 iterations
# Then, you need to apply the final classifier, 
# which is a boosted combination of these 15 perceptron classifiers to the testing data set 
# to evaluate your final classifier

# You shall record the classification accuracy and compare it in a bar chart with that of the decision tree model 


import Load_Dataset

dataset=Load_Dataset.get_dataset()