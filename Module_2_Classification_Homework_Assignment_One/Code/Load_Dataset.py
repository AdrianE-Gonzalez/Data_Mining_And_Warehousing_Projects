# I Created an .env file in order to set my file pathings
#           -FILE_PATH= 'direct path to the folder containing the Dataset'

import os
from dotenv import load_dotenv

import pandas as pd

#This is Used To Access Dataset In Specified File Path
load_dotenv()
FILE_PATH=os.getenv('FILE_PATH')
RESULTS_PATH=os.getenv('RESULTS_PATH')


# Returns dataset 
# Dataset Contains 10 Predictive Attributes; 1 Goal Attribute
# 10 Predictive Attributes Are In Ordered By (Suit, Card #),(Suit, Card #), etc...
# 1 Goal Attribute:
#      0: Nothing in hand; not a recognized poker hand 
#      1: One pair; one pair of equal ranks within five cards
#      2: Two pairs; two pairs of equal ranks within five cards
#      3: Three of a kind; three equal ranks within five cards
#      4: Straight; five cards, sequentially ranked with no gaps
#      5: Flush; five cards with the same suit
#      6: Full house; pair + different rank three of a kind
#      7: Four of a kind; four equal ranks within five cards
#      8: Straight flush; straight + flush
#      9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush

def change_outputs(dataset):
    dataset.iloc[:,-1:] = dataset.iloc[:,-1:].replace(to_replace=[2,3,4,5,6,7,8,9], value=1)
    return dataset

# Returns Training Dataset
def get_train_dataset():
    # Saves Dataset To a Pandas Dataframe
    # Set Header To None Since, poker-hand-training-true.data File Does Not Include Header
    # If Header Not Set To None, The First Row Will Be Counted As The Column Names (Header)
    train_dataset= pd.read_csv(FILE_PATH+'/poker-hand-training-true.data', header=None)

    train_dataset=change_outputs(train_dataset)
    return train_dataset

# Returns Testing Dataset
def get_test_dataset():
    test_dataset= pd.read_csv(FILE_PATH+'/poker-hand-testing.data', header=None)
    test_dataset=change_outputs(test_dataset)

    return test_dataset

# Returns The File Path To Store Results
def get_results():
    return RESULTS_PATH