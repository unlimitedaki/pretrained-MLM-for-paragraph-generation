from nltk.tokenize import sent_tokenize
from allennlp.predictors.predictor import Predictor
import pdb
import json
from tqdm import tqdm
import nltk
nltk.download('punkt')

def find_labels(node,POS_label="RB",CON_label="ADVP",result = []):
    if POS_label in node['attributes'] or CON_label in node["attributes"]:
        result.append(node['word'])
        return 
    elif "children" in node.keys():
        for child in node["children"]:
            find_labels(child,POS_label,CON_label,result)
        return 
    return

def split_data(file_name):
    sentence_data = []
    with open(file_name,'r',encoding='utf8') as f:
        data = f.readlines()
    sentence_data = [sent_tokenize(line) for line in data] 
    return sentence_data

def pos_analyze(sentence_data):
    # pdb.set_trace()
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")
    blank_documents = {}
    for i,document in tqdm(enumerate(sentence_data)):
        blank_sentence = {}
        for sentence in document:
            blank_sentence[sentence] = []
            sentence_labels = predictor.predict(sentence=sentence)
            hierplane_tree = sentence_labels['hierplane_tree']
            ADV_result = []
            find_labels(hierplane_tree['root'],"RB","ADVP",ADV_result)
            # blank_sentence[sentence]["ADV"] = []
            for item in ADV_result:
                new_sentence = sentence.replace(item,"[BLANK]")
                blank_sentence[sentence].append(tuple([new_sentence,item,"ADV"]))
            PP_result = []
            find_labels(hierplane_tree['root'],"IGNORE","PP",PP_result)
            # blank_sentence[sentence]["PP"] = []
            for item in PP_result:
                new_sentence = sentence.replace(item,"[BLANK]")
                blank_sentence[sentence].append(tuple([new_sentence,item,"PP"]))
            # print(blank_sentence[sentence])
        blank_documents[i] = blank_sentence
    
    with open("sentences_with_blank.json",'w',encoding="utf8") as f:
        json.dump(blank_documents,f,indent=2,ensure_ascii = False)
    return blank_documents
    
    