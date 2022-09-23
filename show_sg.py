from __future__ import print_function
import argparse
import numpy as np
import json

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='0',
                help='id of image')
parser.add_argument('--mode', type=str,  default='sen',
                help='image or sen')

parser.add_argument('--input_json', default='./data/cocotalk_final.json', help='input json file')

opt = parser.parse_args()
if opt.mode == 'sen':
    sg_dict = np.load('data/spice_sg_dict2.npz', allow_pickle=True)['spice_dict'][()]
    sg_dict = sg_dict['ix_to_word']
    folder = 'data/coco_spice_sg2/'
else:
    sg_dict = np.load('data/coco_pred_sg_rela.npy', allow_pickle=True)[()]
    sg_dict = sg_dict['i2w']
    folder = 'data/coco_img_sg/'


imgs = json.load(open(opt.input_json, 'r'))['images']

freq_dict = {}
for item in imgs:
    sg_path = folder + str(item['id']) + '.npy'
    sg_use = np.load(sg_path, allow_pickle=True, encoding='latin1')[()]
    if opt.mode == 'sen':
        rela = sg_use['rela_info']
        obj_attr = sg_use['obj_info']
    else:
        rela = sg_use['rela_matrix']
        obj_attr = sg_use['obj_attr']
        print (rela)
        exit()
    N_rela = len(rela)
    N_obj = len(obj_attr)

    # for i in range(N_obj):
    #     if opt.mode == 'sen':
    #         #print('obj # {0}' . format ( i ))
    #         if len(obj_attr[i]) >= 2:
    #             for j in range(len(obj_attr[i])-1):
    #                 print('{0} '.format(sg_dict[obj_attr[i][j + 1]]))
    #         print(sg_dict[obj_attr[i][0]])
    #     else:
    #         print('obj #{0}'.format(i))  # maybe it means 'bounding box' but not 'object'
    #         N_attr = 3
    #         for j in range(N_attr - 1):
    #             print('{0} {1}, '.format(sg_dict[obj_attr[i][j + 4]],\
    #                 sg_dict[obj_attr[i][j+1]]))
    #         j = N_attr - 1
    #         print('{0} {1}'.format(sg_dict[obj_attr[i][j + 4]],\
    #             sg_dict[obj_attr[i][j+1]]))

    for i in range(N_rela):
        obj_idx = 0 if opt.mode == 'sen' else 1
        # sbj = sg_dict[int(obj_attr[int(rela[i][0])][obj_idx])]
        sbj = int(obj_attr[int(rela[i][0])][obj_idx])
        #obj = sg_dict[int(obj_attr[int(rela[i][1])][obj_idx])]
        obj = int(obj_attr[int(rela[i][1])][obj_idx])
        rela_name = int(rela[i][2])
        try:
            freq_dict[(sbj,rela_name)] += 1
        except Exception as e:
            freq_dict[(sbj,rela_name)] = 1

    # print('rel #{3}: {0}-{1}-{2}'.format(sbj,rela_name,obj,i))
a1_sorted_keys = sorted(freq_dict, key=freq_dict.get, reverse=True)
count = 0
for r in a1_sorted_keys:
    print(r, freq_dict[r])
    count +=1
    if count >200:
        break
# print (freq_dict)
# json.dump(freq_dict, open('frequency.json', 'w'))















