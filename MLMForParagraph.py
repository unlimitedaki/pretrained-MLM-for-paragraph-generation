
import torch
import torch.nn as nn
from transformers import *
import json
import argparse
from tqdm import tqdm
import pdb

def load_data(file_name):
    with open(file_name,'r',encoding="utf8") as f:
        data = f.readlines()
    return data

def load_pretrained_model(model_name):
    if 'bert' in model_name:
        return BertForMaskedLM.from_pretrained(model_name)
    elif 'roberta' in model_name:
        return RobertaForMaskedLM.from_pretrained(model_name)

def load_tokenizer(model_name):
    if 'bert' in model_name:
        return BertTokenizer.from_pretrained(model_name)
    elif 'roberta' in model_name:
        return RobertaTokenizer.from_pretrained(model_name)

class MLMForParagraph(nn.Module):
    def __init__(self,model_name,k=5):
        super().__init__()
        self.pretraindModel = load_pretrained_model(model_name)
        self.tokenizer = load_tokenizer(model_name)
        self.k = k

    def forward(self,input_sentence):
        input_ids = self.tokenizer(input_sentence,return_tensors="pt")["input_ids"].cuda()
        # pdb.set_trace()
        masked_indexs = torch.eq(input_ids,103)[0]
        prediction_scores = self.pretraindModel(input_ids)[0][0]
        predictions = {}
        for index,i in enumerate(masked_indexs):
            if i == True:
                topk,topk_index = torch.topk(prediction_scores[index],self.k)
                topk_words = self.tokenizer.convert_ids_to_tokens(topk_index)
                predictions[index] = {}
                predictions[index]['words'] = topk_words
                predictions[index]['scores'] = topk.cpu().detach().numpy().tolist() 
                # predictions[index]['sentences'] = [self.tokenizer.convert_tokens_to_string(input_ids[:index]+torch.tensor([i]).cuda()+input_ids[index+1:]) for i in topk_index]               
        return predictions

def gen_paragraph(args):
    data = load_data(args.source_file)
    model = MLMForParagraph(args.model_name,k=args.topk)
    model.cuda()
    result = {}
    for input_sentence in tqdm(data):
        predictions = model(input_sentence)
        result[input_sentence] = predictions
    with open(args.result_file,'w',encoding='utf8') as f:
        json.dump(result,f,indent=2,ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str,default="bert-base-uncased")
    parser.add_argument('--source_file',type=str)
    parser.add_argument('--result_file',type=str)
    parser.add_argument('--topk',type = int)
    args = parser.parse_args()

    gen_paragraph(args)

            