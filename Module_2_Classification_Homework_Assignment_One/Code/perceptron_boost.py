# Added This Python File To Check Whether I Could Apply A Perceptron To An AdaBoost Classifier
from sklearn import linear_model
from sklearn.metrics import accuracy_score, classification_report

def split_variables(dataset):    
    # split into input (X) and output (y) variables
    Y=dataset.iloc[:,-1:]
    X=dataset.drop(dataset.columns[-1],axis=1)

    return X,Y

def get_perceptron(dataset,max_iter):

    X,Y= split_variables(dataset)

    perceptron_ = linear_model.Perceptron(max_iter=max_iter*15)
    perceptron_= perceptron_.fit(X,Y.values.ravel())
    return perceptron_

def predict_model(model,test_dataset):
    X,Y=split_variables(test_dataset)

    pred =  model.predict(X)
    pred_acc= accuracy_score(Y, pred)
    print(classification_report(Y,pred))
    return pred,pred_acc