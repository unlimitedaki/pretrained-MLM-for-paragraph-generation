from MLMForParagraph import *
import json
result_data = mlm_paragraph("CompreOE-passage.txt","result.txt",topk = 3,mask_range = 5,batch_size = 10)
print(result_data)
with open("result.json",'w',encoding='utf8') as f:
    json.dump(result_data,f,indent=2,ensure_ascii=False)