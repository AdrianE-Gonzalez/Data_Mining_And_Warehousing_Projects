from sklearn import tree
from sklearn.metrics import accuracy_score
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
def update_to_binary(Y):
    Y=Y.replace(to_replace=[2,3,4,5,6,7,8,9], value=1)
    return Y

# Trains train_dataset Using DecisionTreeClassifier
# Returns The Trained Model
def train_model(train_dataset,filename):
    X,Y=split_variables(train_dataset)

    Y=update_to_binary(Y)
    # By Default, Criterion Is Set To Gini
    model=tree.DecisionTreeClassifier(max_depth=2)
    model=model.fit(X,Y)

    print_model(model,Y,filename)

    return model

# Predicts The test_dataset To The model Trained
# returns pred And pre_acc
#       pred = (The List Of Predicted Results)
#       pred_acc = (The Predicted Accuracy)
def predict_model(model,test_dataset):
    X,Y=split_variables(test_dataset)
    Y=update_to_binary(Y)

    pred =  model.predict(X)
    pred_acc= accuracy_score(Y, pred)

    return pred,pred_acc

#Creates Creates and Saves a .png Image to selected Folder
def print_model(model,Y,filename):
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data, 
                    #feature_names=Y.iloc[:, 0].unique(),
                    class_names=list(map(str, Y.iloc[:, 0].unique())), 
                    filled=True, rounded=True, 
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(filename)
    Image(graph.create_png())