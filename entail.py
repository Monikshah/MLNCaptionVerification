import numpy as np
import json
import os
import torch
import spacy
from fuzzywuzzy import fuzz
'''
	1. get selected rules for each image
	2. arrange the selected rules in descending order
	3. seperate the rules into two part head and tail
	4. check if head entails and tail entails
'''
def get_imageid():
	nlp_model = spacy.load('en_core_web_lg')
	temp = open('./../vis/vis_aoa_rl_test.json')
	sent_aoa = json.load(temp)
	temp = open('./../vis/vis_vrg_mle_test.json')
	sent_sgae = json.load(temp)
	itr = 0
	list_img = []
	for i in range(len(sent_sgae)):
	    _vrg = [j for j in sent_sgae[i].values()]
	    _aoa = [j for j in sent_aoa[i].values()]
	    sen1 = nlp_model(_vrg[1])
	    sen2 = nlp_model(_aoa[1])
	    score = sen1.similarity(sen2)
	    
	    if score < 0.8:
	        tmp = _vrg[0]
	        list_img.append(tmp) 
	
	with open('dif_cap_lst_aoarl_sgaemle.txt', 'w') as f:
		f.write(json.dumps(list_img))
	return list_img

def main():
	imgs = json.load(open('./../data/cocotalk_final.json', 'r'))['images']    # loads image name, split and image id
	pred_labels = json.load(open("./../data/coco_dicts.json"))['predicate_to_idx']
	sg_dict = np.load('./../data/coco_pred_sg_rela.npy', allow_pickle=True)[()]['i2w'] #contains all the objects in dataset
	sg_img_obj_dir = './../data/coco_img_sg/'
	
	read_caption = json.load(open('./../vis_val/vis_wvrd_rl.json', 'r'))

	gen_caption = {} #change read_caption to dict
	for i in read_caption:
		gen_caption[str(i['image_id'])] = i['caption']

	roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
	roberta.eval()  # disable dropout (or leave in train mode to finetune)

	# read rules from allrules_update.txt
	with open("./../rule_parameters_CS_val_update_cmnsns.txt") as ff:
		rules_file = json.load(ff)

	roberta_result_vrg = {}
	for img in imgs:
		if img['split'] in ['val', 'test', 'restval']:
			rules = rules_file[str(img['id'])]
			name = str(img['id']) + '.npy'
			name_box = str(img['id']) + '.npy'
			temp = {}
			if rules:
				caption = gen_caption[str(img['id'])]
				for i in rules:
					head = i.split(',')[0]
					tail = i.split(',')[1]

					head_result = roberta.encode(caption, head)
					tail_result = roberta.encode(caption, tail)
					temp[i] = [int(roberta.predict('mnli', head_result).argmax()), int(roberta.predict('mnli', tail_result).argmax())]
			
			roberta_result_vrg[str(img['id'])] = temp #entail results

						
	# if not os.path.exists('./../entail_res'):
		# os.makedirs('./../entail_res')
	json.dump(roberta_result_vrg, open('./../entail_res/res_wvrd_rl_CS_val_full.json', 'w')) 



if __name__ == "__main__":
	main()
