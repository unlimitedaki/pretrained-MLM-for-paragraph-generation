# pretrained-MLM-for-paragraph-generation

## 环境配置

allnnlp-models需要最后一个安装

```shell
pip install transformers==3.0.2
pip install allennlp
pip install --pre allennlp-models
```

## 接口函数：

```python
def mlm_paragraph(
    source_file,
    result_file,
    topk = 3,
    mask_range = 5, # mask数量范围
    batch_size = 10, # 需要取mask_range的倍数
    overwrite_cache = True #覆盖中间文件(blank_selection,mask结果)
)
文本格式的结果会被写入result_file，同时也返回一个json格式的结果
{
  "0": { #文档编号
    "0": [ # 句子编号
      { #复述句
        "pos_label": "PP", # blank 类型
        "context_label": "PN", # 上下文类型
        "masked_words": "in the United States of America",
        "statement": "Yellowstone National Park is nearby.It became the first National Park in 1872.",
        "score": 1.3861298561096191 # 分数采用困惑度，越小越好
      }]}}
```

## 直接运行

```shell
!python MLMForParagraph.py \
    --source_file "CompreOE-passage.txt" \
    --result_file "result.txt" \
    --topk 3 \
    --mask_range 5 \
```



