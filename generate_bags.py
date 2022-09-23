import pandas as pd
import numpy as np
import os
import json
import csv
import pickle

sg_img_dir = './data/cocobu_att/'

def read_input_data():
    # read_iterator = pd.read_csv('./data/instance/59pos_instance.csv', header=None)
    # Positive and negative instance generation
    tmp = open("./data/coco_dicts.json")  # label_to_idx and predicate_to_idx
    coco_dicts = json.load(tmp)
    pred_labels = coco_dicts['predicate_to_idx']  # loads the labels of the predicate
    tmp1 = open("./data/aligned_triplets_final.json")     # loads predicates and its object-pairs
    triplets = json.load(tmp1)
    count = 1
    imgs = json.load(open('./data/cocotalk_final.json', 'r'))['images']    # loads image name, split and image id

    return pred_labels, triplets, imgs

# generates positive and negative instances for each predicates
# pred_labels dict: labels of each predicates
# triplets dict: predicates and obj pairs
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
            obj_feat = np.load(sg_img_file)['feat']
            for key in triplets[str(entry['id'])]:
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
            # print('positive bags', pos_bag_index)
            # exit()

    with open(dir_path + '/' + 'pos_feature_bags.csv', 'wb') as handle:
        pickle.dump(pos_bag_index, handle)

    with open(dir_path + '/' + 'neg_feature_bags.csv', 'wb') as handle:
        pickle.dump(neg_bag_index, handle)


    # with open(dir_path + '/' + 'pos_feature_bags.csv', 'w') as f:
    #     w = csv.writer(f)
    #     for item in pos_bag_index.values():
    #         w.writerows([item, ])
    #     # w.writerow(pos_bag_index.values())
    #     exit()
        # for key, value in pos_bag_index.items():
        #     w.writerow([key, value])
        #     exit()
        # w.writerow(pos_bag_index.keys())
        # w.writerow(pos_bag_index.values())

    # with open(dir_path + '/' + 'neg_feature_bags.csv', 'w') as f:
    #     w = csv.writer(f)
    #     w.writerow(neg_bag_index.keys())
    #     w.writerow(neg_bag_index.values())

    # np.savetxt('./data/instance/train/' + str(pred_label - 1) + '/' + 'pos_instance.csv', pos_bag_index, delimiter=',')
    # np.savetxt('./data/instance/train/' + str(pred_label - 1) + '/' + 'neg_instance.csv', neg_bag_index, delimiter=',')
    exit()


def main():
    pred_labels, triplets, imgs = read_input_data()

    for predicate, pred_label in pred_labels.items():
        generate_instances(pred_label, triplets, imgs)

if __name__ == '__main__':
    main()
