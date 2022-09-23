from sklearn import metrics
import numpy as np
import pandas as pd
import torch
import mil_pytorch.mil as mil
from mil_pytorch.mil import BagModel, MilDataset
from torch.utils.data import DataLoader
import time
from torch.utils import data



class MyDataset(data.Dataset):
    def __init__(self, csv_path, chunkSize, train_data_size):
        self.chunksize = chunkSize
        self.train_data_size = train_data_size
        self.reader = pd.read_csv(csv_path, iterator=True, header=None, chunksize=self.chunksize)

    def __len__(self):
        return self.train_data_size

    def __getitem__(self, index):
        data = self.reader.get_chunk()
        tensorData = torch.as_tensor(data.values, dtype=torch.float64)
        return tensorData

def eer(pred, labels):
    fpr, tpr, threshold = metrics.roc_curve(labels.detach(), pred.detach(), pos_label=1)
    fnr = 1 - tpr
    EER_fpr = fpr[numpy.nanargmin(numpy.absolute((fnr - fpr)))]
    EER_fnr = fnr[numpy.nanargmin(numpy.absolute((fnr - fpr)))]
    return EER_fpr, EER_fnr

def accuracy(pred, target, threshold = 0):
    pred = pred.detach().numpy()
    target = target.detach().numpy()

    pred[pred >= threshold] = 1
    pred[pred < threshold] = -1

    return numpy.sum(target == pred)/target.shape[0]

#read trained model


print('EVALUATION:')
chunk_size = 1
batch_size = 5
kwargs = {}
# Train dataloader for evaluation
custom_data_pos = MyDataset('./data/instance/val/59pos_instance.csv', chunk_size, train_data_size=1000)
custom_data_neg = MyDataset('./data/instance/val/59neg_instance.csv', chunk_size, train_data_size=1000)

test_loader1 = DataLoader(dataset=custom_data_pos, batch_size=batch_size, num_workers=0, shuffle=False, **kwargs)
test_loader2 = DataLoader(dataset=custom_data_neg, batch_size=batch_size, num_workers=0, shuffle=False, **kwargs)


for x, y in zip(test_loader1, test_loader2):
    instance = torch.cat((x, y), 0)
    input_len = instance.size()[2]
    # print('****', instance.size()[2])

    # exit()

    prepNN = torch.nn.Sequential(
        torch.nn.Linear(input_len, 10, bias=True),
        torch.nn.ReLU(),
    )

    afterNN = torch.nn.Sequential(
        torch.nn.Linear(10, 1, bias=True),
        torch.nn.Tanh(),
    )

    model = mil.BagModel(prepNN, afterNN, aggregation_func=torch.mean).double()

    model.load_state_dict(torch.load('model.pt')['model_state_dict'])

    id1 = torch.tensor(np.ones(len(instance) // 2), dtype=torch.long)
    id2 = torch.tensor(np.zeros(len(instance) // 2), dtype=torch.long)
    ids = torch.cat((id1, id2), 0)

    label1 = torch.tensor(np.ones(len(instance) // 2), dtype=torch.long)
    label2 = torch.tensor(np.zeros(len(instance) // 2), dtype=torch.long)
    label2[label2 == 0] = -1
    labels = torch.cat((label1, label2), 0)
    # print(instance.size())
    dataset = MilDataset(instance, ids, labels)
    # print(len(dataset.data))
    test_dl = DataLoader(dataset, batch_size=len(dataset.data), collate_fn=mil.collate)

    for data, ids, labels in test_dl:
        pred = model((data, ids))
        exit()
        loss = criterion(pred[:,0], labels)
        acc = accuracy(pred[:,0], labels)
        eer_fpr, eer_fnr = eer(pred[:,0], labels)

    print('TEST DATA')
    print('Loss: {:6}'.format(loss.item()))
    print('Accuracy: {:.2%}'.format(acc))
    print('Equal error rate approximation using false positive rate: {:.3}'.format(eer_fpr))
    print('Equal error rate approximation using false negative rate: {:.3}'.format(eer_fnr))