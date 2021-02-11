from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Imports For Graphviz
import graphviz
import pydotplus
from io import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz


# Splits Dataset Into X and Y;
#   X=Predictive Attributes
#   Y=Goal Attribute
# Returns X,Y Variables
def split_variables(dataset):    
    # split into input (X) and output (y) variables
    Y=dataset.iloc[:,-1:]
    X=dataset.drop(dataset.columns[-1],axis=1)

    return X,Y

# Replaces Numbers 2-9 To A One From The Y Dataframe


# Trains train_dataset Using DecisionTreeClassifier
# Returns The Trained Model
def train_model(train_dataset,max_depth,filename):
    X,Y=split_variables(train_dataset)

    # By Default, Criterion Is Set To Gini
    model=tree.DecisionTreeClassifier(max_depth=max_depth)
    model=model.fit(X,Y.values.ravel())

    # print_model(model,Y,filename)

    return model

# Predicts The test_dataset To The model Trained
# returns pred And pre_acc
#       pred = (The List Of Predicted Results)
#       pred_acc = (The Predicted Accuracy)
def predict_model(model,test_dataset):
    X,Y=split_variables(test_dataset)

    pred =  model.predict(X)
    pred_acc= accuracy_score(Y, pred)
    print(classification_report(Y,pred))
    return pred,pred_acc

# #Creates Creates and Saves a .png Image to selected Folder
# def print_model(model,Y,filename):
#     dot_data = StringIO()
#     export_graphviz(model, out_file=dot_data, 
#                     #feature_names=Y.iloc[:, 0].unique(),
#                     class_names=list(map(str, Y.iloc[:, 0].unique())), 
#                     filled=True, rounded=True, 
#                     special_characters=True)
#     graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#     graph.write_png(filename)
#     Image(graph.create_png())