import torch
import argparse
from sentence_transformers import SentenceTransformer, util

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def result_maker_for_test(data_result, data_nv_mid, query_result, query_nv_mid, threshold=None):
    data_result = torch.tensor(data_result).to('cuda')
    query_result = torch.tensor(query_result).to('cuda')
    result_list = []
    for nv_mid in query_nv_mid:
        result_list.append((nv_mid,[])) ## [('1906368762', ['1906368762','1810466025','5159532445']], ['636762', ['636762','1146025','155245']]]

    for i, data in enumerate(data_result):
        # print('{}번째'.format(i),'data간의 거리차 구하는 중..' ,'/', '총 {} 개'.format(len(data_label)))
        sims = util.pytorch_cos_sim(data.unsqueeze(dim=0), query_result)[0]
        sims = sims.cpu()
        max_index = torch.argmax(sims)
        if sims[max_index] < threshold:
            continue
        else:
            result_list[max_index][1].append(data_nv_mid[i])

    return result_list