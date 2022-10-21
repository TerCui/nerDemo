import torch.nn
from transformers import BertTokenizerFast
from transformers import AutoModelForTokenClassification
import numpy as np


model = AutoModelForTokenClassification.from_pretrained('./output')

tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
print(model)

def get_token(input):
    english = 'abcdefghijklmnopqrstuvwxyz'
    output=[]
    buffer=''
    for s in input:
        if s in english or s in english.upper():
            buffer+=s
        else:
            if buffer:output.append(buffer)
            buffer=''
            output.append(s)
    if buffer: output.append(buffer)
    return output

if __name__ == '__main__':
    input_str = '四川发现3878亿立方米页岩气储量'
    input_char = get_token(input_str)
    input_tensor = tokenizer(input_char,is_split_into_words=True,padding=True,max_length=512,truncation=True,return_offsets_mapping=True,return_tensors='pt')
    input_tokens = input_tensor.tokens()
    offsets = input_tensor['offset_mapping']
    ignore_mask = offsets[0,:,1]==0
    input_tensor.pop("offset_mapping")

    outputs = model(**input_tensor)
    probabilities = torch.nn.functional.softmax(outputs.logits,dim=-1)[0].tolist()
    prediction = outputs.logits.argmax(dim=-1)[0].tolist()

    results=[]
    tokens = input_tensor.tokens()
    idx = 0
    while idx < len(prediction):
        if ignore_mask[idx]:
            idx+=1
            continue
        pred = prediction[idx]
        label = model.config.id2label[pred]
        if label !="O":
            label = label[2:]
            start = idx
            end = start +1
            all_scores = []
            all_scores.append(probabilities[start][prediction[start]])
            while(
                end < len(prediction)
                and model.config.id2label[prediction[end]] == f'I-{label}'
            ):
                all_scores.append(probabilities[end][prediction[end]])
                end +=1
                idx +=1
            score = np.mean(all_scores).item()
            word = input_tokens[start:end]
            results.append({"entity_group":label,
                            "score":score,
                            "word":word,
                            "start":start,
                            "end":end})
        idx += 1
    print(results)