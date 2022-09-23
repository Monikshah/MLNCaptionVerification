import numpy as np
import os
import json

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
#obj_labels = coco_dicts['label_to_idx']
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

def get_triplet(temp):
    sub = temp['obj_pair'][0]
    obj = temp['obj_pair'][1]
    triplet = [int(sub), int(obj), int(temp['predicate'])]

    return triplet

def main():
    pred_labels, obj_labels, triplets, imgs = read_input_data()

    itr = 0
    for img in imgs:
        vrg_use = dict(np.load(os.path.join("./data/coco_cmb_vrg_new/", str(img['id']) + '.npz')))
        triplets_keys = [key for key in triplets[str(img['id'])]]
        if not triplets_keys:
            np.savez_compressed("./data/gn_triplet/" + str(img['id']) + '.npz', prela=vrg_use['prela'],
                                wrela=vrg_use['wrela'], obj=vrg_use['obj'])
            continue
        itr += 1

        gt_triplet = []
        print('prela', vrg_use['prela'])
        print('wrela', vrg_use['wrela'])
        exit()
        for key in triplets_keys:
            tmp = [triplets[str(img['id'])][key][0][0], triplets[str(img['id'])][key][0][1], int(key)]
            gt_triplet.append(tmp)
        print(gt_triplet)
        exit()
        # triplet_out = get_triplet(temp)
        np.savez_compressed("./data/gn_triplet/" + str(img['id']) + '.npz', prela=vrg_use['prela'],
                            wrela=gt_triplet, obj=vrg_use['obj'])
        # temp = np.load(os.path.join("./data/vrg_ilp/" + str(img['id']) + '.npz'), 'r', allow_pickle=True)
        print('iteration', itr)


if __name__ == '__main__':
    main()



