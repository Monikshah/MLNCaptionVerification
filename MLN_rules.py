import collections
from collections import defaultdict
import glob
import os
import numpy as np
import pickle
import ast
import json

coco_dicts = json.load(open("coco_dicts.json"))  # label_to_idx and predicate_to_idx
obj_labels = coco_dicts['label_to_idx']
pred_labels = coco_dicts['predicate_to_idx']  # loads the labels of the predicate

imgs = json.load(open('./cocotalk_final.json', 'r'))['images']
tmp1 = open("./aligned_triplets_final.json")     # loads predicates and its object-pairs
triplets = json.load(tmp1)
sg_img_obj_dir = 'coco_img_sg/'
sg_dict = np.load('coco_pred_sg_rela.npy', allow_pickle=True)[()]['i2w'] #contains all the objects in
tripletdict = defaultdict(list)
for img in imgs:
    if img['split'] in ['train']:
        name = str(img['id']) + '.npy'
        name_box = str(img['id']) + '.npy'
        sg_path = sg_img_obj_dir + name_box
        sg_use = np.load(sg_path, allow_pickle=True, encoding='latin1')[()]  #contains the objects
        obj_attr = sg_use['obj_attr']
        pred_list = [int(key) for key in triplets[str(img['id'])]]

        for key in triplets[str(img['id'])]:
            for obj_pair in triplets[str(img['id'])][key]:
                obj1 = sg_dict[int(obj_attr[obj_pair[0]][1])]
                obj2 = sg_dict[int(obj_attr[obj_pair[1]][1])]
                pred = [k for k,v in pred_labels.items() if v == int(key)+1]
                K = obj1+":"+pred[0]+":"+obj2
                V = img['id']
                if K not in tripletdict:
                    tripletdict[K].append(V)
                else:
                    if V not in tripletdict[K]:
                        tripletdict[K].append(V)

#FIND RULES WITH SHARED VARS

def gettriples(imgid):
    name = str(imgid) + '.npy'
    name_box = str(imgid) + '.npy'
    sg_path = sg_img_obj_dir + name_box
    sg_use = np.load(sg_path, allow_pickle=True, encoding='latin1')[()]  #contains the objects
    obj_attr = sg_use['obj_attr']
    #print(obj_attr)
    pred_list = [int(key) for key in triplets[str(imgid)]]
    preds = []
    for key in triplets[str(imgid)]:
        for obj_pair in triplets[str(imgid)][key]:
            obj1 = sg_dict[int(obj_attr[obj_pair[0]][1])]
            obj2 = sg_dict[int(obj_attr[obj_pair[1]][1])]
            pred = [k for k,v in pred_labels.items() if v == int(key)+1]
            K = obj1+":"+pred[0]+":"+obj2
            preds.append(K)
            break
    return preds

itr = 0
rulebases = {}
for k in tripletdict:
    for k1 in tripletdict[k]:
        trips = gettriples(k1)
        for t in trips:
            if t==k:
                continue
            parts = t.split(":")
            if parts[0] in k or parts[2] in k:
                st = k+","+t
                if st not in rulebases:
                    rulebases[st]=1
                else:
                    rulebases[st]= rulebases[st] + 1
    itr = itr + 1
    if itr > 100000:
        break

#MAKE FINAL RB
cnt = 0
itr = 0
TRESH1 = 0.1
TRESH2 = 0.1
finalRB = {}

for k in rulebases:
    parts = k.split(",")
    C1 = len(tripletdict[parts[0]])
    C2 = len(tripletdict[parts[1]])
    V1 = rulebases[k]
    symkey = parts[1]+","+parts[0]
    
    if symkey in rulebases:
        V2 = rulebases[symkey]
    else:
        V2= 0
    
    #symmetrical rule
    if V1/float(C1) > TRESH1 and V2/float(C2) > TRESH2 and C1> 1 and C2 > 1:
        cnt = cnt + 1
        itr = itr + 1
        
        if k not in finalRB and symkey not in finalRB:
            finalRB[k] = (V1/float(C1) + V2/float(C2))/2.0

#save final RuleBase in a text file
ofile = open("allrules.txt",'w')
for f in finalRB:
    ofile.write(f+"\n")
ofile.close()


