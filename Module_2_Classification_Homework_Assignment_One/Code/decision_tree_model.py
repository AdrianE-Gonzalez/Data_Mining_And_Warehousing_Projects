from sklearn import tree

def split_variables(dataset):    
    # split into input (X) and output (y) variables
    Y=dataset.iloc[:,length-1]
    X=dataset.drop(dataset.columns[:,length-1],axis=1)

return X,Y

def train_model():
    model=tree.DecisionTreeClassifier()
    model= decision_tree.fit(X,Y)
   
    return model