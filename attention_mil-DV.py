import numpy as np
import torch
#import mil_pytorch.mil as mil
#from mil_pytorch.mil import BagModel, MilDataset
from torch.utils.data import DataLoader
import time
import os
import json
from model_new import GatedAttention
from torch.autograd import Variable

def read_input_data():
    # Positive and negative instance generation
    tmp = open("./data/coco_dicts.json")  # label_to_idx and predicate_to_idx
    coco_dicts = json.load(tmp)
    pred_labels = coco_dicts['predicate_to_idx']  # loads the labels of the predicate
    tmp1 = open("./data/aligned_triplets_final.json")     # loads predicates and its object-pairs
    triplets = json.load(tmp1)
    imgs = json.load(open('./data/cocotalk_final.json', 'r'))['images']    # loads image name, split and image id

    return pred_labels, triplets, imgs

sg_img_dir = './data/cocobu_att/'
obj_box_dir = './data/cocobu_box/'

# model = Attention()
def train_instances(pred_label, triplets, imgs):
    pos_bag_index = neg_bag_index = {}
    key = 'obj_pair'
    pos_bag_index.setdefault(key, [])
    neg_bag_index.setdefault(key, [])
    model = GatedAttention()
    learning_rate = 0.00005
    weight_decay = 1e-6
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
    n_epoch = 10
    start = time.time()
    for epoch in range(n_epoch):
        train_loss = 0
        train_error = 0
        cnt = 0
        model.train()
        for entry in imgs:
            if entry['split'] in ['train', 'restval']:
                image_id = entry['id']
                triplets_keys = [key for key in triplets[str(entry['id'])]]
                if not triplets_keys:
                    continue
                if str(pred_label - 1) not in triplets_keys:
                    continue
                name = str(image_id) + '.npz'
                name_box = str(image_id) + '.npy'
                obj_box_file = os.path.join(obj_box_dir, name_box)
                box_feat = np.load(obj_box_file)
                sg_img_file = os.path.join(sg_img_dir, name)
                obj_feat = np.load(sg_img_file)['feat']  #use dataloader to load this file
                pcnt = 0
                ncnt = 0
                for key in triplets[str(entry['id'])]:   #use batch size for the length
                    if int(key) == pred_label - 1:
                        for obj_pair in triplets[str(entry['id'])][key]:
                            temp = np.concatenate([obj_feat[obj_pair[0]], obj_feat[obj_pair[1]],box_feat[obj_pair[0]], box_feat[obj_pair[1]]])
                            x_np = torch.from_numpy(temp).reshape(1,4104)
                            if pcnt==0:
                                xPos = x_np
                                pcnt = 1
                            else:
                                xPos = torch.cat((x_np,xPos), 0)

                    else:
                        for obj_pair in triplets[str(entry['id'])][key]:
                            temp = np.concatenate([obj_feat[obj_pair[0]], obj_feat[obj_pair[1]], box_feat[obj_pair[0]], box_feat[obj_pair[1]]])
                            x_np = torch.from_numpy(temp).reshape(1,4104)
                            if ncnt==0:
                                xNeg = x_np
                                ncnt = 1
                            else:
                                xNeg = torch.cat((x_np,xNeg), 0)
                if pcnt==1:
                    pos_label = torch.tensor(np.ones(1), dtype=torch.bool)
                    data_s, labels = Variable(xPos), Variable(pos_label)
                    optimizer.zero_grad()
                    loss, _ = model.calculate_objective(data_s, labels)
                    train_loss += loss.data[0]
                    loss.backward()
                    optimizer.step()
                    error, _ = model.calculate_classification_error(data_s, labels)
                    train_error += error 
                    cnt = cnt + 1
                if ncnt==1:
                    neg_label = torch.tensor(np.zeros(1), dtype=torch.bool)
                    data_s, labels = Variable(xNeg), Variable(neg_label)
                    optimizer.zero_grad()
                    loss, _ = model.calculate_objective(data_s, labels)
                    train_loss += loss.data[0]
                    loss.backward()
                    optimizer.step()

                    error, _ = model.calculate_classification_error(data_s, labels)
                    train_error += error
                    cnt = cnt + 1
        train_loss /= cnt
        train_error /= cnt
        if (epoch % 1 == 0):
            print('Epoch: {}, Loss: {}, Train error: {:.4f}'.format(epoch, train_loss.detach().numpy()[0], train_error))
    print('Finished training - elapsed time: {}'.format(time.time() - start))
    path = './data/model-feat-box-GatedAttention/' + str(pred_label) + '.pt'
    torch.save(model, path)

    return model
    
def generate_instances_test(pred_label, triplets, imgs, model):
    model.eval()
    test_error = 0
    cnt = 0
    for entry in imgs:
        if entry['split'] in ['test']:
            image_id = entry['id']
            triplets_keys = [key for key in triplets[str(entry['id'])]]
            if not triplets_keys:
                continue
            if str(pred_label - 1) not in triplets_keys:
                continue
            name = str(image_id) + '.npz'
            name_box = str(image_id) + '.npy'
            obj_box_file = os.path.join(obj_box_dir, name_box)
            box_feat = np.load(obj_box_file)
            sg_img_file = os.path.join(sg_img_dir, name)
            obj_feat = np.load(sg_img_file)['feat']  #use dataloader to load this file
            pcnt = 0
            ncnt = 0
            for key in triplets[str(entry['id'])]:   #use batch size for the length
                if int(key) == pred_label - 1:
                    for obj_pair in triplets[str(entry['id'])][key]:
                        temp = np.concatenate([obj_feat[obj_pair[0]], obj_feat[obj_pair[1]],box_feat[obj_pair[0]], box_feat[obj_pair[1]]])
                        x_np = torch.from_numpy(temp).reshape(1,4104)
                        if pcnt==0:
                            xPos = x_np
                            pcnt = 1
                        else:
                            xPos = torch.cat((x_np,xPos), 0)

                else:
                    for obj_pair in triplets[str(entry['id'])][key]:
                        temp = np.concatenate([obj_feat[obj_pair[0]], obj_feat[obj_pair[1]],box_feat[obj_pair[0]], box_feat[obj_pair[1]]])
                        x_np = torch.from_numpy(temp).reshape(1,4104)
                        if ncnt==0:
                            xNeg = x_np
                            ncnt = 1
                        else:
                            xNeg = torch.cat((x_np,xNeg), 0)
                if pcnt==1:
                    pos_label = torch.tensor(np.ones(1), dtype=torch.bool)
                    data_s, labels = Variable(xPos), Variable(pos_label)
                    # loss, _ = model.calculate_objective(data_s, labels)
                    # test_loss += loss.data[0]
                    error, _ = model.calculate_classification_error(data_s, labels)
                    test_error += error 
                    cnt = cnt + 1
                if ncnt==1:
                    neg_label = torch.tensor(np.zeros(1), dtype=torch.bool)
                    data_s, labels = Variable(xNeg), Variable(neg_label)
                    # loss, _ = model.calculate_objective(data_s, labels)
                    # test_loss += loss.data[0]
                    error, _ = model.calculate_classification_error(data_s, labels)
                    test_error += error
                    cnt = cnt + 1
    test_error /= cnt
    print('total test error:', test_error)


def main():
    pred_labels, triplets, imgs = read_input_data()
    itr = 0
    for predicate, pred_label in pred_labels.items():
        print('iteration:', itr)
        if itr < 159:
            itr += 1
            continue
        model = train_instances(pred_label, triplets, imgs)
        # print(model)
        #generate_instances_test(pred_label, triplets, imgs, model)
        itr += 1
        # exit()

if __name__ == '__main__':
    main()



