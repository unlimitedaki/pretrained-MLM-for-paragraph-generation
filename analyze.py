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

def simplify(doc,pos_predictor):
    pos = pos_predictor.predict(sentence=doc)
    
    # s.1 递归遍历，找出要删的字符串列表[SBAR | IN+S]
    def traverse(dict, result=[]):               
        if dict["attributes"] == ["SBAR"]:
            result.append(dict["word"])
            return True

        if "children" in dict.keys():
            flag=0
            for i in range(len(dict["children"])):
                if i<len(dict["children"])-1 and dict["children"][i]["attributes"]==["IN"] and dict["children"][i+1]["attributes"]==["S"]:
                    string = dict["children"][i]["word"] + " " + dict["children"][i+1]["word"]
                    result.append(string)
                    flag = 1
                elif dict["children"][i]["attributes"]==["S"] and flag==1:
                    flag = 0
                    continue
                else:  
                    traverse(dict["children"][i], result)
        else:
            return True
    # s.2 模式匹配[PP|ADVP] + , 的开头字符串
    def judgement(dict, result=[]):
        while "children" in dict.keys():
            if dict["children"][0]["attributes"][0] not in ["PP", "ADVP"]:
                dict = dict["children"][0]
            else:
                if dict["children"][1]["attributes"] == [","]:
                    string = dict["children"][0]["word"] + " " + dict["children"][1]["word"]+" "
                    result.append(string)
                break
    delete_list = []
    traverse(pos["hierplane_tree"]["root"], delete_list)
    judgement(pos["hierplane_tree"]["root"], delete_list)
    # print(doc,delete_list)

    for st in delete_list:
        doc = doc.replace(st, "")
    
    doc_list = doc.split(",")
    if len(doc_list)==3:
        doc = doc_list[0] + " , " + doc_list[1]
    # s.2 删除两个逗号之间的部分
    # print(doc)
    return doc


def pos_analyze(sentence_data):
    # pdb.set_trace()
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")
    blank_documents = {}
    for i,document in tqdm(enumerate(sentence_data)):
        blank_sentence = {}
        for sentence in document:
            sentence = simplify(sentence,predictor)
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
    
    