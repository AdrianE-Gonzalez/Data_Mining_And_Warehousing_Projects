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

# You shall record the classification accuracy and compare it in a bar chart with that of the decpwdsision tree model 


import Load_Dataset
import decision_tree_model
import ada_boost
import perceptron_boost
import bar_chart

train_dataset=Load_Dataset.get_train_dataset()
test_dataset=Load_Dataset.get_test_dataset()
filename=Load_Dataset.get_results()+'/Trees/Training_Results.png'

def run_multi_test():
    dt= []
    ada= []
    ada_dt= []
    percept= []
    ada_p= []

    for x in range(1,16):

        

        print('##################################################################################################################')
        model=decision_tree_model.train_model(train_dataset,x,filename)
        pred,pred_acc=decision_tree_model.predict_model(model,test_dataset)
        print('Decision_Tree_Depth: '+str(x)+' = '+str(pred_acc))
        dt.append((x,pred_acc))

        print()
        
        model_ada= ada_boost.created_tree_boost(train_dataset,x,model)
        pred_ada,pred_acc_ada= ada_boost.predict_model(model_ada,test_dataset)
        print('Ada_Boost_Tree_Depth: '+str(x)+' = '+str(pred_acc_ada))
        ada.append((x,pred_acc_ada))

        print()

        # perceptron_model= perceptron_boost.get_perceptron(train_dataset,x)
        # pred_boosted,pred_acc_boosted= perceptron_boost.predict_model(perceptron_model,test_dataset)
        # print('Perceptron: '+str(x)+' = '+str(pred_acc_boosted))
        # percept.append((x,pred_acc_boosted))

        # print()
        
        model_boosted= ada_boost.created_tree_boost(train_dataset,x,model_ada)
        pred_boosted,pred_acc_boosted= ada_boost.predict_model(model_boosted,test_dataset)
        print('Created_Tree_Ada_Boost_Tree_Depth: '+str(x)+' = '+str(pred_acc_boosted))
        ada_dt.append((x,pred_acc_boosted))

        print()

        # model_boosted_perceptron= ada_boost.created_tree_boost(train_dataset,x,model_boosted)
        # pred_boosted_perceptron,pred_acc_boosted_perceptron= ada_boost.predict_model(model_boosted_perceptron,test_dataset)
        # print('Created_Perceptron_Ada_Boost_Tree_Depth: '+str(x)+' = '+str(pred_acc_boosted_perceptron))
        # ada_p.append((x,pred_acc_boosted_perceptron))

        print()
    bar_chart.create_bar_chart(dt, ada, ada_dt)


def tree_vs_ada_max():
    model_tree= decision_tree_model.train_model(train_dataset,None,filename)
    pred,pred_acc=decision_tree_model.predict_model(model_tree,test_dataset)
    print('Decision_Tree: '+str(pred_acc))

    model_ada= ada_boost.boost(train_dataset,None)
    pred_ada,pred_acc_ada= ada_boost.predict_model(model_ada,test_dataset)
    print('Ada_Boost_Tree_Depth: '+str(pred_acc_ada))
    
    bar_chart.create_tree_vs_ada_bar_chart(pred_acc,pred_acc_ada, 'Accuracy vs Algorithms: Maximum Depth')

def tree_vs_ada_default():
    model_tree= decision_tree_model.train_model(train_dataset,None,filename)
    pred,pred_acc=decision_tree_model.predict_model(model_tree,test_dataset)
    print('Decision_Tree: '+str(pred_acc))

    model_ada= ada_boost.boost(train_dataset,1)
    pred_ada,pred_acc_ada= ada_boost.predict_model(model_ada,test_dataset)
    print('Ada_Boost_Tree_Depth: '+str(pred_acc_ada))
    
    bar_chart.create_tree_vs_ada_bar_chart(pred_acc,pred_acc_ada,'Accuracy vs Algorithms: Default Max Depth')

run_multi_test()
tree_vs_ada_max()
tree_vs_ada_default()