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

    wrela_path = "/home/monika/Documents/PhD-Project/WeakVRD-Captioning/data/output_ilp/"
    itr = 0
    for img in imgs:
        vrg_use = dict(np.load(os.path.join("./data/coco_cmb_vrg_new/", str(img['id']) + '.npz')))
        triplets_keys = [key for key in triplets[str(img['id'])]]
        if not triplets_keys:
            np.savez_compressed("./data/vrg_ilp/" + str(img['id']) + '.npz', prela=vrg_use['prela'],
                                wrela=vrg_use['wrela'], obj=vrg_use['obj'])
            continue
        itr += 1
        try:
            temp = np.load(os.path.join(wrela_path, str(img['id']) + '.npz'), 'r', allow_pickle=True)
        except Exception as e:
            print(e)
            np.savez_compressed("./data/vrg_ilp/" + str(img['id']) + '.npz', prela=vrg_use['prela'],
                                wrela=vrg_use['wrela'], obj=vrg_use['obj'])
            continue
        triplet_out = get_triplet(temp)
        np.savez_compressed("./data/vrg_ilp/" + str(img['id']) + '.npz', prela=vrg_use['prela'],
                            wrela=[triplet_out], obj=vrg_use['obj'])
        # temp = np.load(os.path.join("./data/vrg_ilp/" + str(img['id']) + '.npz'), 'r', allow_pickle=True)
        print('iteration', itr)


if __name__ == '__main__':
    main()

#code to check the object labels into words for each predicted predicate
    # temp_res = []
    # temp = ["target"]
    # objectPair = []
    # temp_res.append(["image-id", image_id])
    # for i in target:
    #     objectPair.append(list(triplets[str(entry['id'])][str(i-1)]))
    #     temp.append(list(pred_labels.keys())[list(pred_labels.values()).index(int(i))])

    # fullpair = []
    # for i in objectPair:
    #     pair = []
    #     for j in i:
    #         pair.append([list(obj_labels.keys())[list(obj_labels.values()).index(j[0]+1)], list(obj_labels.keys())[list(obj_labels.values()).index(j[1]+1)]])
    #     fullpair.append(pair)
    # print(fullpair)
    # exit()
    # temp_res.append(temp)
    # temp_res.append(objectPair)
    #
    # print(temp_res)
    # exit()

    # temp = ["predicted"]
    # for k in predicted_tot:
    #     for j in k:
    #         try:
    #             temp.append(list(pred_labels.keys())[list(pred_labels.values()).index(int(i))])
    #         except Exception as e:
    #             print (e)
    #             pass
    #
    # temp_res.append(temp)
    #
    # with open('results.csv', "a", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(temp_res)


