from pathlib import Path
import re
import numpy as np
import torch
from torch.utils.data import Dataset

data_dir = './data'

def read_data(data_path):
    file_path = Path(data_path)
    raw_text = file_path.read_text(encoding='UTF-8').strip()
    raw_docs = re.split('\n\t?\n',raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token,tag = line.split(' ')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs,tag_docs

token_train,tag_train = read_data(data_dir+'/tain.txt')
token_val,tag_val = read_data(data_dir+'/val.txt')
token_test,tag_test = read_data(data_dir+'/test.txt')

unique_tags = set(tag for doc in tag_train for tag in doc)
tag2id = {tag:id for id,tag in enumerate(unique_tags)}
id2tag = {id:tag for tag,id in tag2id.items()}

label_list = list(unique_tags)

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
train_encodings = tokenizer(token_train,is_split_into_words=True,padding=True,max_length=512,truncation=True,return_offsets_mapping=True)
val_encodings = tokenizer(token_val,is_split_into_words=True,padding=True,max_length=512,truncation=True,return_offsets_mapping=True)

def encodeing_labels(tags,encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoding_labels = []
    for doc_labels,doc_offset in zip(labels,encodings.offset_mapping):
        doc_enc_labels = np.ones(len(doc_offset),dtype=int)*-100
        arr_offset = np.array(doc_offset)
        if len(doc_labels) >= 510:
            doc_labels = doc_labels[:510]
        doc_enc_labels[(arr_offset[:,0]==0) & (arr_offset[:,1]!=0)] = doc_labels
        encoding_labels.append(doc_enc_labels)
    return encoding_labels

train_labels = encodeing_labels(tag_train,train_encodings)
val_labels = encodeing_labels(tag_val,val_encodings)

class NerDataset(Dataset):

    def __init__(self,encodings,labels):
        print('__init__ called.')
        self.encodings=encodings
        self.labels=labels

    def __getitem__(self,idx):
        item = {key:torch.tensor(val[idx]) for key,val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_encodings.pop('offset_mapping')
val_encodings.pop('offset_mapping')
train_dataset = NerDataset(train_encodings,train_labels)
val_dataset = NerDataset(val_encodings,val_labels)

from transformers import AutoModelForTokenClassification,Trainer,TrainingArguments
model = AutoModelForTokenClassification.from_pretrained('ckiplab/albert-base-chinese-ner',num_labels=7,
                                                        ignore_mismatched_sizes=True,
                                                        id2label=id2tag,
                                                        label2id=tag2id)
print(model)

from datasets import load_metric
metric = load_metric('seqeval')

def compute_metric(p):
    predictions,labels = p
    predictions = np.argmax(predictions,axis=2)

    true_predictions = [
        [label_list[p] for (p,l) in zip(prediction,label) if l!=-100]
        for prediction,label in zip(predictions,labels)
    ]

    true_labels = [
        [label_list[l] for (p,l) in zip(prediction,label)if l!=-100]
        for prediction,label in zip(predictions,labels)
    ]

    results = metric.compute(predictions=true_predictions,references = true_labels)
    return {"precision":results["overall_precision"],
            "recall":results["overall_recall"],
            "f1":results["overall_f1"],
            "accuracy":results["overall_accuracy"]}

checkpoint="bert-base-chinese"
num_train_epochs=100
per_device_train_batch_size=8
per_device_eval_batch_size=8

train_args = TrainingArguments(output_dir='./output',num_train_epochs=num_train_epochs,per_device_train_batch_size=per_device_train_batch_size,
                               per_device_eval_batch_size=per_device_eval_batch_size,warmup_steps=500,weight_decay=0.01,logging_dir='./logs',logging_steps=10,
                               save_steps=1000,save_strategy='steps',save_total_limit=1,evaluation_strategy='steps',eval_steps=1000)

trainer = Trainer(model=model,args=train_args,train_dataset=train_dataset,eval_dataset=val_dataset,compute_metrics=compute_metric)

trainer.train()
trainer.evaluate()

model.save_pretrained("./checkpoint/model/%s-%sepoch"%(checkpoint,num_train_epochs))