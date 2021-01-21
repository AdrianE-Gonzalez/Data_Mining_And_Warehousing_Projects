# I Created an .env file in order to set my file pathings
#           -FILE_PATH= 'direct path to the folder containing the Dataset'

import os
from dotenv import load_dotenv

import pandas as pd

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


def get_dataset():
    #This is Used To Access Dataset In Specified File Path
    load_dotenv()
    FILE_PATH=os.getenv('FILE_PATH')
    
    # Saves Dataset To a Pandas Dataframe
    dataset= pd.read_csv(FILE_PATH, delimiter='\n')

    return dataset