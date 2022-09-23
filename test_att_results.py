import glob
import numpy as np
import torch
import os
import json
from torch.autograd import Variable

#read test input
def read_input_data():
    # Positive and negative instance generation
    tmp = open("./data/coco_dicts.json")  # label_to_idx and predicate_to_idx
    coco_dicts = json.load(tmp)
    obj_labels = coco_dicts['label_to_idx']
    pred_labels = coco_dicts['predicate_to_idx']  # loads the labels of the predicate
    tmp1 = open("./data/aligned_triplets_final.json")     # loads predicates and its object-pairs
    triplets = json.load(tmp1)
    imgs = json.load(open('./data/cocotalk_final.json', 'r'))['images']    # loads image name, split and image id
    return pred_labels, obj_labels, triplets, imgs

sg_img_dir = './data/cocobu_att/'
sg_img_obj_dir = './data/coco_img_sg/'
obj_box_dir = './data/cocobu_box/'
sg_dict = np.load('./data/coco_pred_sg_rela.npy', allow_pickle=True)[()]
sg_dict = sg_dict['i2w']
tmp = open("./data/coco_dicts.json")  # label_to_idx and predicate_to_idx
coco_dicts = json.load(tmp)
pred_labels = coco_dicts['predicate_to_idx']  # loads the labels of the predicate

def sub_pred_obj(predicted, triplets):
    predicted = sum(predicted, [])
    itr = 0
    out_triplet = []
    for i in triplets:
        object_pair = triplets[i]
        for value in object_pair:
            value.insert(2, int(predicted[itr]))
            out_triplet.append(value)
        itr += 1
    return out_triplet

def generate_instances(entry, triplets, files):
    image_id = entry['id']
    name = str(image_id) + '.npz'
    name_box = str(image_id) + '.npy'
    obj_box_file = os.path.join(obj_box_dir, name_box)
    box_feat = np.load(obj_box_file)
    sg_img_file = os.path.join(sg_img_dir, name)
    obj_feat = np.load(sg_img_file)['feat']  # use dataloader to load this file
    #obj = np.load(sg_img_obj_dir)['obj_attr']  # use dataloader to load this file
    # sg_path = sg_img_obj_dir + name_box
    # sg_use = np.load(sg_path, allow_pickle=True, encoding='latin1')[()]  #contains the objects
    # obj_attr = sg_use['obj_attr']

    # triplets_keys = [key for key in triplets[str(entry['id'])]]
    triplet_result = []
    for key in triplets[str(image_id)]:
        obj_pairs = triplets[str(entry['id'])][key]
        pcnt = 0
        for obj_pair in obj_pairs:
            temp = np.concatenate([obj_feat[obj_pair[0]], obj_feat[obj_pair[1]],box_feat[obj_pair[0]], box_feat[obj_pair[1]]])
            x_np = torch.from_numpy(temp).reshape(1,4104)
            if pcnt==0:
                xPos = x_np
                pcnt = 1
            else:
                xPos = torch.cat((xPos,x_np), 0)
        att_weight = []
        predicted = []
        for file in files:
            model = torch.load(file)
            model.eval()
            if pcnt==1:
                pos_label = torch.tensor(np.ones(1), dtype=torch.bool)
                data_s, labels = Variable(xPos), Variable(pos_label)
                prob, pred, attention_wts = model(data_s)
            att_weight.append(attention_wts)
            predicted.append(prob)
        predicted_ = torch.tensor(predicted)
        predicted_final = torch.topk(predicted_, 1) #select indices of first 5 top probabilitities
        predicted_final1 = [int(x) for x in predicted_final[1]]  #index of predicted final in files
        pred_objpair = obj_pairs[int(torch.argmax(att_weight[predicted_final1[0]]))]
        predicted_final_rela = [os.path.split(files[i])[1].split('.')[0] for i in predicted_final1] # top five predicates

        if int(predicted_final_rela[0]) > 200:
            print('exception:', int(predicted_final_rela[0]))
        triplet_res = [pred_objpair[0], pred_objpair[1], int(predicted_final_rela[0])]
        triplet_result.append(triplet_res)

    return triplet_result

def main():
    pred_labels, obj_labels, triplets, imgs = read_input_data()
    path = "/home/monika/Documents/PhD-Project/WeakVRD-Captioning/data/model-feat-box-GatedAttention/*"
    files = glob.glob(path)
    folder = "./data/vrg_gated_max_pred_att/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    itr = 0
    for img in imgs:
        vrg_use = dict(np.load(os.path.join("./data/coco_cmb_vrg_new/", str(img['id']) + '.npz')))
        triplets_keys = [key for key in triplets[str(img['id'])]]
        print(itr)
        itr += 1
        if not triplets_keys:
            np.savez_compressed(folder + str(img['id']) + '.npz', prela=vrg_use['prela'],
                                wrela=vrg_use['wrela'], obj=vrg_use['obj'])
            continue
        output_triplet = generate_instances(img, triplets, files)
        output_triplet = np.array(output_triplet)

        np.savez_compressed(folder + str(img['id']) + '.npz', prela=vrg_use['prela'], wrela=output_triplet, obj=vrg_use['obj'])


if __name__ == '__main__':
    main()