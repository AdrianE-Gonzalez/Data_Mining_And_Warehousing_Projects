import matplotlib.pyplot as plt 
import matplotlib.pyplot
import numpy as np

# Plots Each Classifier Accuracies Into A Bar Chart
def create_bar_chart(dt, ada, ada_dt):

    width=0.2
    fig, ax = plt.subplots()

    # Loops Through Each Test Run And Polts Them Into The Bar Chart
    for acc in range(0,len(dt)): 

        bar_1= ax.bar((dt[acc][0]-width-.1), round(dt[acc][1]*100,1), width, color='red', label='Decision Tree')
        bar_2= ax.bar(ada[acc][0], round(ada[acc][1]*100,1), width, color='blue', label='Ada Decision Tree')
        bar_3= ax.bar(ada_dt[acc][0]+width+.1, round(ada_dt[acc][1]*100,1), width, color='green', label='Ada Plus Created Decision Tree')
        
        autolabel(bar_1,ax)
        autolabel(bar_2,ax)
        autolabel(bar_3,ax)

    # Set Label Names
    ax.set_ylabel('Accuracy')
    ax.set_xlabel("Tree Depth")
    ax.set_title('Accuracy vs Tree Depth')

    # Creates A Legend According To Classifier According To Their Respective Color
    colors = {'Decision Tree':'red', 
        'AdaBoost Plus Decision Tree Model':'blue', 
        'AdaBoost Plus AdaBoost Model':'green'}         

    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)

    x_ticks = np.arange(0, 16, 1)
    plt.xticks(x_ticks)

    plt.show()

# Labels Each Bar In The Bar Chart
def autolabel(bar,ax):
    for b in bar:
        height = b.get_height()
        ax.annotate('{}%'.format(height),
                    xy=(b.get_x() + b.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Plots Each Classifier Accuracies Into A Bar Chart
def create_tree_vs_ada_bar_chart(dt, ada, chart_name):

    width=0.35
    fig, ax = plt.subplots()

    bar_1= ax.bar(0, round(dt*100,3), width, color='red', label='Decision Tree')
    bar_2= ax.bar(1, round(ada*100,3), width, color='blue', label='Ada Decision Tree')
        
    autolabel(bar_1,ax)
    autolabel(bar_2,ax)

    # Set Label Names
    ax.set_ylabel('Accuracy')
    ax.set_xlabel("Alogrithms")
    ax.set_title(chart_name)
    
    # Creates A Legend According To Classifier According To Their Respective Color
    colors = {'Decision Tree':'red', 
        'AdaBoost':'blue'}         

    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)

    plt.xticks(np.arange(2), ('Decision Tree', 'Ada Boost'))

    plt.show()