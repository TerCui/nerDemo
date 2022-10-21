from torch.utils.data import Dataset

class NerDataset():

    def  __init__(self,encodings,labels):
        print('__init__ called.')
        self.encodings=encodings
        self.labels=labels

rect = NerDataset(3,4)

print(rect.__dict__)
