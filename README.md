# nerDemo
NLP网络bert-base ner简单的demo项目

数据标注用开源的doccano进行标注
然后用convert中的代码转成BIO格式的数据
用huggingface的预训练模型加载bert模型
训练并保存成自己的模型
demoner中进行预测并解析预测结果
