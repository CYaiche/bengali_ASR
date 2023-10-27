import os 
import json
import pandas as pd 
from common.common_params import EXTRACT_DIR, MAX_TEXT_OUTPUT_MINUS_START_CHAR


def remove_punctuation(test_str):

    # initializing punctuations string
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~”“’‘–—।'''
    res=" "

    for ele in test_str:
        if ele not in punc:
            res+=ele

    return res 

def print_characters(text):
    used_characters = set(list(text))
    c_list = list(used_characters)
    print(f"len : {len(c_list)}")
    print(sorted(c_list, reverse=True))
    
# Step 1: Tokenization (Character-level)
def tokenize_text(text):
    return list(text)

# Step 2: Vocabulary Creation
def create_vocabulary(texts):
    # Get unique characters
    unique_chars = set(char for text in texts for char in tokenize_text(text))
    
    # Create a vocabulary mapping characters to IDs
    char_to_id = {char: idx for idx, char in enumerate(unique_chars)}
    
    return char_to_id
    
if __name__ == "__main__" : 
    data_loc = os.path.join(EXTRACT_DIR,"metadata","life.csv")
    data_clean =  os.path.join(EXTRACT_DIR,"metadata","life_clean.csv")
    vocabulary_location = os.path.join(EXTRACT_DIR,"metadata","life_clean_vocabulary.json")
    # load train and test metadata 

    df = pd.read_csv(data_loc)
    print(df.head())

    text = ' '.join(tuple(df["sentence"].explode()))
    print_characters(text)


    df["sentence_clean"] = df["sentence"].apply(remove_punctuation)
    
    text = ' '.join(tuple(df["sentence_clean"].explode()))
    print_characters(text)
    # create a alphabet with chars map to id 
    vocabulary = create_vocabulary(text)
    print(vocabulary)

    # Assuming char_to_id is your vocabulary dictionary
    with open(vocabulary_location, 'w') as f:
        json.dump(vocabulary, f)
        
        

    # save sentence without punctuation 
    df_clean = df.drop("sentence",axis=1 )

    
    # remove label that have less then 20 characters and more than 120  (outliers )
    df_clean["len"] = df_clean["sentence_clean"].apply(len)
    mask = (df_clean["len"] >= 20) & (df_clean["len"] <= MAX_TEXT_OUTPUT_MINUS_START_CHAR)
    df_clean = df_clean[mask]
    
    df_clean["sentence"] = df_clean["sentence_clean"]
    df_clean.drop(columns=["len","sentence_clean"],inplace=True)
    print(df_clean.head())
    df_clean.to_csv(data_clean)
    

    
