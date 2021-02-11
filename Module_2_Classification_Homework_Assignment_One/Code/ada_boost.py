from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report

# Returns X and Y
# Splits Dataset Into X And Y
#   -X= Predictive Attributes
#   -Y= Goal Attribute
def split_variables(dataset):    
    Y=dataset.iloc[:,-1:]
    X=dataset.drop(dataset.columns[-1],axis=1)

    return X,Y

# Returns model
# Used To Create A New AdaBoost Classifier With A New Decision Tree Classifier
def boost(train_dataset,max_depth):
    
    X,Y=split_variables(train_dataset)

    model = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=max_depth),n_estimators=15)
    model = model.fit(X,Y.values.ravel())

    return model

# Returns model
# Used To Create AdaBoost Based On Previously Created Model
def created_tree_boost(train_dataset,max_depth,model_tree):
    
    X,Y=split_variables(train_dataset)

    model = AdaBoostClassifier(base_estimator=model_tree,n_estimators=15)
    model = model.fit(X,Y.values.ravel())

    return model

# Returns pred and pred_acc
# Predicts testing_dataset With Model Built
def predict_model(model,test_dataset):
    X,Y=split_variables(test_dataset)

    pred =  model.predict(X)
    pred_acc= accuracy_score(Y, pred)
    print(classification_report(Y,pred))

    return pred,pred_acc