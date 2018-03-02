import re 
import os
import argparse
import dateutil.parser as dparser
import gensim as gs
from keras.models import model_from_json
from keras.preprocessing import sequence

REG_CLEANER = re.compile(r'[\s\#\!\.\?,\(\)\{\}\[\]\:\;"]')
lstm_model = None
loaded_dict = None
dict_keys = None 

def parseDate( date_string):
    return dparser.parse(date_string, fuzzy=True)

def clean_word(word):
    return REG_CLEANER.sub(" ", word.lower()).strip()

def model_loading_from_json_and_weights(saved_json_file,saved_weights):
    model_json_file=open(saved_json_file,"r")
    loaded_json_model=model_json_file.read()
    model_json_file.close()
    ml_model=model_from_json(loaded_json_model)
    ml_model.load_weights(saved_weights)
    print "Loaded model from disk"
    return ml_model
    
class Item(object):
    def __init__(self,date,text):
        self.date = date
        self.text = text

    def __lt__(self,other):
        return self.date < other.date

    def __eq__(self,other):
        return self.date == other.date

    def __ge__(self,other):
        return self.date >= other.date

    def __str__(self):
        return str(self.date) + " " + self.text

    def get_text(self):
        return self.text

    def get_date(self):
        return self.date


date_regex =re.compile('\s*\"created_at\":\s*\"([a-zA-Z\s0-9:+]+)\"')
text_regex =re.compile('\s*\"text\":\s*\"([^"]+)\"')


def getfiles(directory):
    files  = os.listdir(directory)
    files.sort()
    return files

def prepare(model_files):
    global lstm_model, loaded_dict, dict_keys
    lstm_model=model_loading_from_json_and_weights(model_files+"/model.json",model_files+"/model.h5")
    loaded_dict=gs.corpora.Dictionary().load(model_files+"/sentiment_classifier_dictionary_model.dict")
    dict_keys = loaded_dict.token2id

def predict(text):
    text= text.split()
    
    text_as_token_ids=[loaded_dict.token2id[token] for token in text if token in dict_keys]
    if len(text_as_token_ids) > 0:
        text_sequence=sequence.pad_sequences([text_as_token_ids],maxlen=100)
        prob_1=lstm_model.predict_proba(text_sequence,verbose=2)[0][0]
    else:    
        prob_1=0.5
    return prob_1



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                description='UPDATE TWEET WITH SENTIMENT AND POLITICAL AFFILIATION')

    parser.add_argument(
        '-i', '--input', type=str, help='directory ', required=True)
    parser.add_argument(
        '-o', '--output', type=str, help='OUTPUT TWEETS FILE', default='output.txt', required=False)

    parser.add_argument(
        '-l', '--logtimer', type=int, help='every l seconds it will log the surge result', default=10, required=False)
    parser.add_argument(
        '-v','--verbose', help='verbose at DEBUG level ', default= False, required=False, action='store_true')

    parser.add_argument('-m', '--model_files', type=str, help='directory ', required=True)
    args = parser.parse_args()
    lstm_model=model_loading_from_json_and_weights(args.model_files+"/model.json",args.model_files+"/model.h5")
    loaded_dict=gs.corpora.Dictionary().load(args.model_files+"/sentiment_classifier_dictionary_model.dict")
    files = getfiles(args.input)
    dict_keys = loaded_dict.token2id
    for file in files:
        with open(args.input +"/"+ file, 'r') as readfile:
            items = []
            for line in readfile:
                #date = re.findall(date_regex,line)
                text = re.findall(text_regex,line)
                text = clean_word(text[0])
                text= text.split()
                
                text_as_token_ids=[loaded_dict.token2id[token] for token in text if token in dict_keys]
                if len(text_as_token_ids) > 0:
                    text_sequence=sequence.pad_sequences([text_as_token_ids],maxlen=100)
                    prob_1=lstm_model.predict_proba(text_sequence,verbose=2)[0][0]
                    prob_0=1-prob_1
                else:    
                    prob_1=0.5
                    prob_0=0.5
                last_occ_ind=line.rfind("}")
                changed_line=line[:last_occ_ind]+',"prob_positive":'+str(prob_1)+',"prob_negative":'+str(prob_0)+'}'
                print changed_line            

