import os,sys
if os.path.exists("external_libraries"):
    sys.path.append('external_libraries')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import *
import json
import argparse
from tqdm import tqdm
import pdb
from analyze import *
import logging


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
    def __init__(self,model_name,k=3,beam_nums = 3,mask_range = 5):
        super().__init__()
        self.pretraindModel = load_pretrained_model(model_name)
        self.tokenizer = load_tokenizer(model_name)
        self.k = k
        self.beam_nums = beam_nums
        self.mask_range =mask_range
    
    def beam_search(self,scores,beam_nums,max_length):
        beam_list = []
        position = 0
        while(position<max_length):
            topk_scores,topk_index = torch.topk(scores[position],beam_nums) #(1,beam_nums)
            if position == 0:
                prev_scores = topk_scores
                prev_index = topk_index.view(beam_nums,1)
            else:
                cand_scores = torch.matmul(prev_scores.view(beam_nums,-1),topk_scores.view(-1,beam_nums)).view(-1,beam_nums*beam_nums)
                
                topk_cand_scores,topk_cand_index = torch.topk(cand_scores,beam_nums)
                first_index = (topk_cand_index//beam_nums).view(beam_nums,-1)
                cur_index = (topk_cand_index%beam_nums).view(beam_nums,-1)
                first_index = first_index.expand(beam_nums,prev_index.shape[-1])
                first_index = prev_index.gather(0,first_index)
                cur_index = topk_index.view(beam_nums,-1).gather(0,cur_index)
                prev_index = torch.cat((first_index,cur_index),dim=1)
                prev_scores = topk_cand_scores
            position += 1
        return torch.pow(prev_scores.view(-1,beam_nums),-1/float(max_length)),prev_index
            

    def forward(self,input_ids):
        batch_size = input_ids.shape[0]
        outputs  = self.pretraindModel(input_ids)[0]
        outputs = F.softmax(outputs,dim=-1)
        # masked_indexs  = torch.nonzero(input_ids == 103,as_tuple=None)
        all_scores = []
        all_index = []
        for i,input_id in enumerate(input_ids):
            masked_index  = torch.nonzero(input_id==103)
            # pdb.set_trace()
            reshaped_masked_index = masked_index.expand(masked_index.shape[0],outputs.shape[-1])
            masked_logits = outputs[i].gather(0,reshaped_masked_index)
            beam_scores,beam_index = self.beam_search(masked_logits,self.beam_nums,masked_index.shape[0])
            for index in beam_index:
                all_index.append(index)
            all_scores.append(beam_scores)
        
        all_scores = torch.cat(all_scores,dim = -1).view(-1,self.beam_nums*self.mask_range)
        topk_scores,topk_index = torch.topk(all_scores,k = self.k,largest = False)
        cand_tokens = []
        for i,index in enumerate(topk_index):
            tokens = [all_index[i*all_scores.shape[-1]+j].cpu().detach().numpy().tolist() for j in index]
            cand_tokens.append(tokens)
        # pdb.set_trace()
        return topk_scores.cpu().detach().numpy().tolist(), cand_tokens
        
            # pdb.set_trace()
        # input_ids = self.tokenizer(input_sentence,return_tensors="pt")["input_ids"].cuda()
        # # pdb.set_trace()
        # masked_indexs = torch.eq(input_ids,103)[0]
        # prediction_scores = self.pretraindModel(input_ids)[0]
        # pdb.set_trace()
        # predictions = {}
        # for index,i in enumerate(masked_indexs):
        #     if i == True:
        #         topk,topk_index = torch.topk(prediction_scores[index],self.k)
        #         topk_words = self.tokenizer.convert_ids_to_tokens(topk_index)
        #         predictions[index] = {}
        #         predictions[index]['words'] = topk_words
        #         predictions[index]['scores'] = topk.topk.cpu().detach().numpy().tolist()  
                # predictions[index]['sentences'] = [self.tokenizer.convert_tokens_to_string(input_ids[:index]+torch.tensor([i]).cuda()+input_ids[index+1:]) for i in topk_index]               
        return predictions

def pn_context(statement,passage,sentence_id):
    if sentence_id == 0:
        pre = ""
    else:
        pre = passage[sentence_id-1]
    if sentence_id == len(passage)-1:
        next = ""
    else:
        next = passage[sentence_id+1]
    return pre+statement+next
    

def mask_sentence(args,data):
    masked_data = []
    for doc_id,document in data.items():
        passage = list(document.keys())
        for sentence_id,raw in enumerate(document.keys()):
            for item in document[raw]:
                PN_masked_data = []
                C_masked_data = []
                if "[BLANK]" not in item[0]:
                    continue
                for i in range(1,args.mask_range+1):
                    mask_tokens = " ".join(["[MASK]"]*i)
                    # pdb.set_trace()
                    statement = item[0].replace("[BLANK]",mask_tokens)
                    PN_statement = pn_context(statement,passage,sentence_id)
                    C_statement = passage[sentence_id]+statement
                    PN_masked_data.append(tuple([PN_statement,item[1],i,doc_id,sentence_id,item[2],"PN"]))
                    C_masked_data.append(tuple([C_statement,item[1],i,doc_id,sentence_id,item[2],"C"]))
                masked_data += PN_masked_data
                masked_data += C_masked_data
                    # masked = (PN_statement,i,doc_id,sentence_id,item[1],"PN")
    with open("masked_data.json","w",encoding="utf8") as f:
        json.dump(masked_data,f,indent = 2,ensure_ascii=False)
    return masked_data
                

def preprocess(args,tokenizer,masked_data):
    input_ids = torch.tensor([tokenizer(
        data[0],
        max_length = args.max_seq_length,
        padding="max_length"
        )["input_ids"] for data in masked_data
    ])
    # others = [data[1:] for data in masked_data]
    # others = [data[1:] for data in masked_data]
    dataset = TensorDataset(input_ids)
    sampler = SequentialSampler(dataset)
    dataloader= DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)
    return dataloader


def gen_paragraph(args):
    source_data = split_data(args.source_file)
    if (not args.overwrite_cache) and os.path.exists("masked_data.json"):
        logger.info("load cached masked data from masked_data.json")
        with open("masked_data.json",'r',encoding='utf8') as f:
            masked_data = json.load(f)
    else:
        if (not args.overwrite_cache) and os.path.exists("sentences_with_blank.json"):
            logger.info("loading cached sentence from sentences_with_blank.json")
            with open("sentences_with_blank.json",'r',encoding = 'utf8') as f:
                analyze_result = json.load(f)
        else:
            logger.info("analyzing sentence from {}".format(args.source_file))
            
            analyze_result = pos_analyze(source_data)
        logger.info("masking sentence")
        masked_data = mask_sentence(args,analyze_result)

    result_file = open(args.result_file,'w',encoding = 'utf8')
    
    # model.cuda()

    tokenizer = load_tokenizer(args.model_name)
    
    dataloader = preprocess(args,tokenizer,masked_data)
    model = MLMForParagraph(args.model_name,k=args.topk)
    model.cuda()
    model.eval()
    num_examples = 0
    last_passage_id = -1
    last_sentence_id = -1
    result_data = {}
    logger.info("generating paragraph")
    for step,batch in tqdm(enumerate(dataloader),total=len(dataloader)):
        input_ids = batch[0].cuda()
        scores,cand_tokens = model(input_ids)
        last_sentence_id = -1
        for i in range(len(cand_tokens)):
            origin_data = masked_data[num_examples*args.mask_range]

            masked_part = origin_data[1]
            masked_sentence = origin_data[0]
            # origin_sentence = masked_sentence.replace("[MASK]",masked_part)
            pos_label = origin_data[5]
            context_label = origin_data[6]
            passage_id = int(origin_data[3])
            sentence_id = origin_data[4]
            origin_sentence = source_data[passage_id][sentence_id]
            if passage_id != last_passage_id:
                result_file.write("Passage {}:\n".format(str(passage_id)))
                result_data[passage_id] = {}
                last_passage_id = passage_id
            if sentence_id != last_sentence_id:
                result_file.write("[{}] {}\n".format(str(sentence_id),source_data[passage_id][sentence_id]))
                result_data[passage_id][sentence_id] = {}
                result_data[passage_id][sentence_id]["statements"] = []
                result_data[passage_id][sentence_id]['raw'] = origin_sentence
                last_sentence_id = sentence_id
            for j in range(len(cand_tokens[i])):
                prediction_tokens = tokenizer.convert_ids_to_tokens(cand_tokens[i][j])
                statement = origin_sentence.replace(masked_part," ".join(prediction_tokens))
                result_file.write("[{}][{}] [{}] {} [{}]\n".format(pos_label,context_label,masked_part,statement,str(scores[i][j])))
                result_data[passage_id][sentence_id]["statements"].append({"pos_label":pos_label,"context_label":context_label,"masked_words":masked_part,'statement':statement,'score':scores[i][j]})    
            num_examples += 1
    with open("mlm_result.json",'w',encoding='utf8') as f:
        json.dump(result_data,f,indent = 4,ensure_ascii=False)
    return result_data

def mlm_paragraph(
    source_file,
    result_file,
    topk = 3,
    mask_range = 5,
    batch_size = 10,
    overwrite_cache = True
):
    args.source_file = source_file
    args.result_file = result_file
    args.topk = topk
    args.mask_range = mask_range
    args.batch_size = batch_size
    args.overwrite_cache = overwrite_cache
    return gen_paragraph(args)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name',type=str,default="bert-base-uncased")
parser.add_argument('--source_file',type=str,default = "CompreOE-passage.txt")
parser.add_argument('--result_file',type=str,default = 'result.txt')
parser.add_argument('--topk',type = int,default=3)
parser.add_argument('--batch_size',type = int,default = 10)
parser.add_argument('--mask_range',type = int, default=5)
parser.add_argument('--max_seq_length',type = int, default=120)
parser.add_argument('--overwrite_cache',type = bool,default= False)
try:
    args = parser.parse_args()
except:
    args = parser.parse_args(args = [])

if __name__ == "__main__":
    gen_paragraph(args)

            