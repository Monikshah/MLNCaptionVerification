import numpy as np
import pandas as pd
import torch
import mil_pytorch.mil as mil
from mil_pytorch.mil import BagModel, MilDataset
from torch.utils.data import DataLoader
import time
from torch.utils import data
import csv
import pickle

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

def create_model(input_len):
    prepNN = torch.nn.Sequential(
        torch.nn.Linear(input_len, 50, bias=True),
        torch.nn.ReLU(),
    )
    #
    # Define custom afterNN
    afterNN = torch.nn.Sequential(
        torch.nn.Linear(50, 1, bias=True),
        torch.nn.Tanh(),
    )
    #
    # # Define model with prepNN, afterNN and torch.mean as aggregation function
    # model = mil.BagModel(prepNN, afterNN, torch.mean)
    model = mil.BagModel(prepNN, afterNN, aggregation_func=torch.mean).double()

    return model

def read_input_data():
    # Positive and negative instance generation
    tmp = open("./data/coco_dicts.json")  # label_to_idx and predicate_to_idx
    coco_dicts = json.load(tmp)
    pred_labels = coco_dicts['predicate_to_idx']  # loads the labels of the predicate
    tmp1 = open("./data/aligned_triplets_final.json")     # loads predicates and its object-pairs
    triplets = json.load(tmp1)
    count = 1
    imgs = json.load(open('./data/cocotalk_final.json', 'r'))['images']    # loads image name, split and image id

    return pred_labels, triplets, imgs

def generate_instances(pred_label, triplets, imgs):
    pos_bag_index = neg_bag_index = {}
    key = 'obj_pair'
    pos_bag_index.setdefault(key, [])
    neg_bag_index.setdefault(key, [])
    dir_path = './data/instance/train/' + str(pred_label - 1)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    for entry in imgs:
        pos_instance = neg_instance = []
        if entry['split'] in ['train', 'restval']:
            image_id = entry['id']
            triplets_keys = [key for key in triplets[str(entry['id'])]]
            if not triplets_keys:
                continue
            if str(pred_label - 1) not in triplets_keys:
                continue
            name = str(image_id) + '.npz'
            sg_img_file = os.path.join(sg_img_dir, name)
            obj_feat = np.load(sg_img_file)['feat']  #use dataloader to load this file
            for key in triplets[str(entry['id'])]:   #use batch size for the length
                    if int(key) == pred_label - 1:
                        for obj_pair in triplets[str(entry['id'])][key]:
                            temp = np.concatenate([obj_feat[obj_pair[0]], obj_feat[obj_pair[1]]])
                            temp = temp.tolist()
                            pos_instance.append(temp)
                    else:
                        for obj_pair in triplets[str(entry['id'])][key]:
                            temp = np.concatenate([obj_feat[obj_pair[0]], obj_feat[obj_pair[1]]])
                            temp = temp.tolist()
                            neg_instance.append(temp)
            if 'obj_pair' in pos_bag_index:
                pos_bag_index['obj_pair'].append(pos_instance)
            if 'obj_pair' in neg_bag_index:
                neg_bag_index['obj_pair'].append(neg_instance)
    return pos_bag_index['obj_pair'], neg_bag_index['obj_pair']

def main():
    batch_size = 10
    kwargs = {}
    chunk_size = 1
    n_feat = 4096

    sg_img_dir = './data/cocobu_att/'
    pred_labels, triplets, imgs = read_input_data()


    for predicate, pred_label in pred_labels.items():
        pos_bag_feat, neg_bag_feature = generate_instances(pred_label, triplets, imgs)

        model = create_model(n_feat)
        criterion = mil.MyHingeLoss()

        learning_rate = 1e-4
        weight_decay = 1e-6
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        train_losses = torch.empty(0)
        start = time.time()
        print('TRAINING:')

        # obj_feature = torch.tensor(obj_feature, dtype=torch.double)
        # train_data = DataLoader(dataset=obj_feature, batch_size=batch_size)
        for data1, data2 in zip(pos_obj_feature, neg_obj_feature):
            pos_instance = torch.tensor(data1, dtype=torch.double)
            neg_instance = torch.tensor(data2, dtype=torch.double)
            instance = torch.cat((pos_instance, neg_instance))
            pos_id = torch.tensor(np.ones(len(pos_instance)), dtype=torch.long)
            neg_id = torch.tensor(np.zeros(len(neg_instance)), dtype=torch.long)
            ids = torch.cat((pos_id, neg_id))
            labels = torch.tensor([1, -1], dtype=torch.long)
            dataset = MilDataset(instance, ids, labels)
            # train_data = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=mil.collate)
            # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            # Training parameters
            epochs = 1

            # Tensor for collecting losses over batches
            # train_losses = torch.empty(0)

            for epoch in range(epochs):
                for data, ids, labels in dataset:
                    inst = torch.zeros(len(data), n_feat, dtype = torch.double)
                    inst[:, :len(data[0])] = data
                    pred = model((inst, ids))
                    loss = criterion(pred[:, 0], labels)

                    # Update weights
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Save loss on this batch
                    train_losses = torch.cat((train_losses, loss.float()))
                # Compute averega loss on this epoch
                train_loss = torch.mean(train_losses, dim=0, keepdim=True)

                # Clear tensor for saving losses over batches
                train_losses = torch.empty(0)

                # Print info about learning every 100 epochs
                if (epoch + 1) % 1 == 0:
                    print('[{}/{}] | train_loss: {}'.format(epoch + 1, epochs, train_loss.item()))
                # exit()

        print('Finished training - elapsed time: {}'.format(time.time() - start))
        exit()
            # torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'loss': train_loss,
            #         }, "model.pt")
            # # torch.save(model.state_dict(), "bag_model.pt")
            # exit()

            ## Evaluation

        from sklearn import metrics

        def eer(pred, labels):
            fpr, tpr, threshold = metrics.roc_curve(labels.detach(), pred.detach(), pos_label=1)
            fnr = 1 - tpr
            EER_fpr = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
            EER_fnr = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
            return EER_fpr, EER_fnr

        def accuracy(pred, target, threshold = 0):
            pred = pred.detach().numpy()
            target = target.detach().numpy()

            pred[pred >= threshold] = 1
            pred[pred < threshold] = -1

            return np.sum(target == pred)/target.shape[0]

        print('EVALUATION:')

        batch_size = 10
        chunk_size = 1
        # Train dataloader for evaluation
        custom_data_pos = MyDataset('./data/instance/val/59pos_instance.csv', chunk_size, train_data_size=100)
        custom_data_neg = MyDataset('./data/instance/val/59neg_instance.csv', chunk_size, train_data_size=100)

        test_loader1 = DataLoader(dataset=custom_data_pos, batch_size=batch_size, num_workers=0, shuffle=False, **kwargs)
        test_loader2 = DataLoader(dataset=custom_data_neg, batch_size=batch_size, num_workers=0, shuffle=False, **kwargs)

        for test_x, test_y in zip(test_loader1, test_loader2):
            instance = torch.cat((test_x, test_y), 0)
            id1 = torch.tensor(np.ones(len(instance) // 2), dtype=torch.long)
            id2 = torch.tensor(np.zeros(len(instance) // 2), dtype=torch.long)
            ids = torch.cat((id1, id2), 0)
            labels = torch.tensor([1, -1], dtype=torch.long)
            dataset = MilDataset(instance, ids, labels)
            test_dl = DataLoader(dataset, batch_size=batch_size, collate_fn=mil.collate)

            for data, ids, labels in test_dl:
                target = torch.zeros(len(data), n_feat, dtype=torch.double)
                target[:, :len(data[0])] = data
                pred = model((target, ids))
                loss = criterion(pred[:,0], labels)
                print(pred[:,0], labels)
                acc = accuracy(pred[:,0], labels)
                # eer_fpr, eer_fnr = eer(pred[:,0], labels)

            print('TEST DATA')
            print('Loss: {:6}'.format(loss.item()))
            print('Accuracy: {:.2%}'.format(acc))
            # print('Equal error rate approximation using false positive rate: {:.3}'.format(eer_fpr))
            # print('Equal error rate approximation using false negative rate: {:.3}'.format(eer_fnr))
        print('<<<<<<<<<<<<<')


if __name__ == '__main__':
    main()

# i = 0

# for train_x, train_y in zip(train_loader1, train_loader2):
#     print('iteration:', i)
#     i = i + 1
#     instance = torch.cat((train_x, train_y), 0)
#     id1 = torch.tensor(np.ones(len(instance) // 2), dtype=torch.long)
#     id2 = torch.tensor(np.zeros(len(instance) // 2), dtype=torch.long)
#     ids = torch.cat((id1, id2), 0)
#     labels = torch.tensor([1, -1], dtype=torch.long)
# print(len(obj_feature))
# exit()





