import os
import shutil
import json
import numpy as np

class_names = {0:'grizzly+bear',1:'bobcat',2:'elephant',3:'gorilla',4:'rhinoceros',5:'dalmatian',6:'dolphin',\
               7:'antelope',8:'otter',9: 'german+shepherd',10:'cow',11:'hamster',12:'collie',13:'chimpanzee',\
               14:'rabbit',15:'deer',16:'wolf',17:'fox',18:'humpback+whale',19:'siamese+cat',20:'ox',\
               21:'giraffe',22:'seal',23:'tiger',24:'polar+bear'}

cof_labels = [cls for idx, cls in class_names.items()]


file_path = '/shared/hailey/graph_jsons/'
model_info ='VGG16-R-SGD-91.827' #'VGG16-R-Adam-92.628'  'VGG16-R-SGD-91.827' 'ResNet50-P-SGD-94.48'
files = [x for x in os.listdir(file_path) if model_info in x]

mode = 'violations' # violations, corrections, all_pass
mode_files = sorted([x for x in files if mode in x], reverse=True)

file_summary_changes={}
file_summary_statistics={}

for file in files:
    # print(file)

    layer_name = file.split('.')[1].split('-')[-2] # -1 for non softmax
    with open(os.path.join(file_path,file),'r') as f:
        vote_all = json.load(f)
    ##################### make feature_votes_w_file_names.json ######################
    change_list = []
    for channel_id, votes in vote_all.items():
        for vote_id, vote_vals in votes.items():
            if (vote_id == 'total') | (vote_id == 'len') | (vote_id == 'jsd_all'): continue
            gt, pred, _, abl = vote_id.split('_')
            changes = '{}-{}'.format(class_names[int(pred)],class_names[int(abl)])
            if changes not in change_list: change_list.append(changes)


    file_summary_changes[file] = change_list

    change_dict_counts = {x: 0 for x in change_list}
    change_dict_channels = {x: [] for x in change_list}
    change_dict_jsd_frm_gt = {x: [] for x in change_list}
    change_dict_jsd_pred_abl = {x: [] for x in change_list}
    change_dict_images = {x: [] for x in change_list}

    for channel_id, votes in vote_all.items():
        for vote_id, vote_details in votes.items():
            if (vote_id == 'total') | (vote_id == 'len') | (vote_id == 'jsd_all'): continue
            gt, pred, _, abl = vote_id.split('_')
            changes = '{}-{}'.format(class_names[int(pred)],class_names[int(abl)])
            change_dict_counts[changes] += int(vote_details['count'])
            if len(vote_details['image']) != 1:
                for m in range(len(vote_details['image'])):
                    change_dict_channels[changes].append(channel_id)
            else:
                change_dict_channels[changes].append(channel_id)

            change_dict_jsd_frm_gt[changes].extend(vote_details['jsd_frm_gt'])
            change_dict_jsd_pred_abl[changes].extend(vote_details['jsd_pred_abl'])
            change_dict_images[changes].extend(vote_details['image'])

    file_summary_statistics[file] = {}
    file_summary_statistics[file]['change_dict_counts'] = change_dict_counts
    file_summary_statistics[file]['change_dict_channels'] = change_dict_channels
    file_summary_statistics[file]['change_dict_jsd_frm_gt'] = change_dict_jsd_frm_gt
    file_summary_statistics[file]['change_dict_jsd_pred_abl'] = change_dict_jsd_pred_abl
    file_summary_statistics[file]['change_dict_images'] = change_dict_images

with open('/shared/hailey/layer_infos/{}-per-layer-changes.json'.format(model_info),'w') as f:
    json.dump(file_summary_changes, f)

with open('/shared/hailey/layer_infos/{}-per-layer-statistics-img.json'.format(model_info),'w') as f:
    json.dump(file_summary_statistics, f)
