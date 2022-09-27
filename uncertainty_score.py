import json
import math
import shutil

f = open("/home/monika/Documents/PhD-Project/WeakVRD-Captioning/entail_res/res_vrg_att_rl_CS_val.json")
ff = open("/home/monika/Documents/PhD-Project/WeakVRD-Captioning/rule_parameters_CS.txt")

rule_parameter = json.load(ff)
entail_file = json.load(f)
tot_score = {}
cnt = 0 

for i in entail_file:
    if not any([True for k,v in rule_parameter[i].items() if v[1]>0]): #check if at wt is non zero
        continue
    if not entail_file[i]: #checks if the score disctionary is non empty
        continue
    atoms = {}
    numpos = 0
    numneg = 0
    numneu = 0
    for rule, val in entail_file[i].items():
        part1 = rule.split(',')[0]
        part2 = rule.split(',')[1]
        #print(part1+" "+str(val[0]))
        #print(part2+" "+str(val[1]))
        atoms[part1]=val[0]
        atoms[part2]=val[1]
        if val[0]==2:
            numpos = numpos + 1
        if val[1]==2:
            numpos = numpos + 1
        if val[0]==0:
            numneg = numneg + 1
        if val[1]==0:
            numneg = numneg + 1
        if val[0]==1:
            numneu = numneu + 1
        if val[1]==1:
            numneu = numneu + 1
    if numpos == 0 and numneg == 0:
        continue
    if len(atoms.keys())<5:
        continue
    probs = {}
    for am in atoms.keys():
        poswt = 0
        negwt = 0
        for rule, val in entail_file[i].items():
            ruleatoms = []
            part1 = rule.split(',')[0]
            part2 = rule.split(',')[1]
            ruleatoms.append(part1)
            ruleatoms.append(part2)
            j = -1
            if part1==am:
                j = 2
            elif part2==am:
                j=1
            if j==-1:
                continue
            rulewt = rule_parameter[i][rule][0]
            if atoms[ruleatoms[j-1]]==2:
                poswt = poswt + math.exp(rulewt)
                negwt = negwt + math.exp(0)
            elif atoms[ruleatoms[j-1]]==0:
                negwt = negwt + math.exp(0)
                poswt = poswt + math.exp(0)
            else:
                poswt = poswt + math.exp(rulewt) + math.exp(0)
                negwt = negwt + math.exp(0) + math.exp(0)
        P = poswt/(poswt+negwt)
        probs[am]=P

#print average uncertainty score
uncert_score = [v for k,v in probs.items()]
avg_uncert_score = np.average(probability)
var_uncert_score  = np.var(probability)
std_uncert_score = np.std(probability)