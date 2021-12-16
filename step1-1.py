import os
import shutil
import json
import numpy as np


class_names = {0:'grizzly+bear',1:'bobcat',2:'elephant',3:'gorilla',4:'rhinoceros',5:'dalmatian',6:'dolphin',\
               7:'antelope',8:'otter',9: 'german+shepherd',10:'cow',11:'hamster',12:'collie',13:'chimpanzee',\
               14:'rabbit',15:'deer',16:'wolf',17:'fox',18:'humpback+whale',19:'siamese+cat',20:'ox',\
               21:'giraffe',22:'seal',23:'tiger',24:'polar+bear'}

cof_labels = [cls for idx, cls in class_names.items()]


file_path = '/shared/hailey/softmax-files'
model_info = 'VGG16-R-Adam-92.628' # 'VGG16-R-SGD-91.827'
##################### open origin file and sort based on total confusion abl ######################

# for file_name in file_w_gt:
# with open(vgg_block1,'r') as f:
#     vote_all = json.load(f)
# # len = 종류, total = 각각의 종류가 등장한 총 합
# vote_all = sorted(vote_all.items(), key=lambda x: x[1]['total'], reverse=True) # count 많은 순

##################### open origin file and sort based on total confusion abl ######################


files = [x for x in os.listdir(file_path) if model_info in x]
# files = [x for x in files if 'wrong' not in x]

mode = 'violations' # violations, corrections, all_pass
mode_files = sorted([x for x in files if mode in x], reverse=True)


for file in files:
    # print(file)
    layer_name = file.split('.')[1].split('-')[-2] # -1 for non softmax
    with open(os.path.join(file_path,file),'r') as f:
        vote_all = json.load(f)

    # print(layer_name)

    ##################### make feature_votes_w_file_names.json ######################
    votes_w_gt = {x:{} for x in range(len(vote_all))}

    for channel_id, votes in vote_all.items():
        total = 0
        jsd_all = [0,0]
        for vote_id, vote_details in votes.items():
            gt, pred, _, abl = vote_id.split('_')
            total += vote_details['count']
            jsd_lists = vote_details['jsd']
            if vote_id not in votes_w_gt[int(channel_id)].keys():
                votes_w_gt[int(channel_id)][vote_id] = {}
                votes_w_gt[int(channel_id)][vote_id]['count'] = vote_details['count']
                votes_w_gt[int(channel_id)][vote_id]['image'] = vote_details['image']
                votes_w_gt[int(channel_id)][vote_id]['jsd_frm_gt']= []
                votes_w_gt[int(channel_id)][vote_id]['jsd_pred_abl'] = []
                votes_w_gt[int(channel_id)][vote_id]['jsd'] = [0, 0]

                for gt_pred, gt_abl, pred_abl in jsd_lists:

                    votes_w_gt[int(channel_id)][vote_id]['jsd_frm_gt'].append(gt_pred - gt_abl)
                    votes_w_gt[int(channel_id)][vote_id]['jsd_pred_abl'].append(pred_abl)
                    votes_w_gt[int(channel_id)][vote_id]['jsd'][0] += gt_pred - gt_abl
                    votes_w_gt[int(channel_id)][vote_id]['jsd'][1] += pred_abl
            else:
                votes_w_gt[int(channel_id)][vote_id]['count'] += vote_details['count']
                votes_w_gt[int(channel_id)][vote_id]['image'] = vote_details['image']

                for gt_pred, gt_abl, pred_abl in jsd_lists:
                    votes_w_gt[int(channel_id)][vote_id]['jsd_frm_gt'].append(gt_pred - gt_abl)
                    votes_w_gt[int(channel_id)][vote_id]['jsd_pred_abl'].append(pred_abl)
                    votes_w_gt[int(channel_id)][vote_id]['jsd'][0] += gt_pred - gt_abl
                    votes_w_gt[int(channel_id)][vote_id]['jsd'][1] += pred_abl

            votes_w_gt[int(channel_id)][vote_id]['jsd'][0] /= float(vote_details['count'])
            votes_w_gt[int(channel_id)][vote_id]['jsd'][1] /= float(vote_details['count'])
            jsd_all[0] += votes_w_gt[int(channel_id)][vote_id]['jsd'][0]
            jsd_all[1] += votes_w_gt[int(channel_id)][vote_id]['jsd'][1]


        try:
            jsd_all[0] /= len(votes)
            jsd_all[1] /= len(votes)
        except ZeroDivisionError:
            jsd_all = [0, 0]

        votes_w_gt[int(channel_id)]['total'] = total
        votes_w_gt[int(channel_id)]['len'] = len(votes)
        votes_w_gt[int(channel_id)]['jsd_all'] = jsd_all

    with open('/shared/hailey/graph_jsons/{}'.format(file), 'w') as f:
        json.dump(votes_w_gt, f)
