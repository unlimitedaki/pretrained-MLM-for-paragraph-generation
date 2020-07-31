# pretrained-MLM-for-paragraph-generation

## 环境配置

```shell
pip install transformers==3.0.2
```

## 运行

```shell
python MLMForParagraph.py --source_file "examples.txt" \
    --result_file "result.json" \
    --topk 5 \
    --model_name "bert-base-uncased" \
```

source_file 为数据文件

result_file 为预测结果，为json格式

topk 取mask位置的前k个预测结果

model_name 预训练模型，现在仅支持BERT

## 数据文件格式

```
I have a dog, my [MASK] is cute.
My dog is [MASK].
My dog is [MASK], my dog is cute.
```

## 结果文件格式

```json
{
  "I have a dog, my [MASK] is cute.\n": {
    "7": {
      "words": [
        "dad",
        "mom",
        "husband",
        "cat",
        "boyfriend"
      ],
      "scores": [
        9.344655990600586,
        9.24299430847168,
        8.592103958129883,
        8.381598472595215,
        8.314824104309082
      ]
    }
}
```



