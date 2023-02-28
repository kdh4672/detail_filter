from train import train_iter_STS, test, threshold_test
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.readers import InputExample
from util import str2bool
from dataset import CatalogDataset, query_finder
from torch.utils.data import DataLoader
import numpy as np
import torch
import os
import sys
import argparse
sys.path.append('../.')

def add_args(parser):
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--model_path', type=str,
                        default='output/training_sts')
    parser.add_argument('--train_parquet_path', type=str,
                        # default='/home1/irteam/user/dhkong/data/text_data/hive_parenting_train_txt_20220128_recent')
                        default='/home1/irteam/user/dhkong/data/text_data/short_train_txt_20220128_recent')
    parser.add_argument('--test_parquet_path', type=str,
                        default='/home1/irteam/user/dhkong/data/text_data/short_train_txt_20220128_recent')
    parser.add_argument('--is_train', type=str2bool, default=True)


def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    #########################
    # dataset prepare #### --> 이 부분 추후 함수로 대체
    #########################
    testset = CatalogDataset(args, args.test_parquet_path)
    test_sentences = testset.sentences()
    test_labels = testset.labels()
    test_nv_mids = testset.get_nv_mid()
    test_query_labels, test_query_sentences, test_query_nv_mids, test_count_dict = query_finder(
        test_labels, test_sentences, test_nv_mids)
    test_query_nv_mid_list = []
    for nv_mid in test_query_nv_mids:  # 여기서 label은 카탈로그 label
        test_query_nv_mid_list.append((nv_mid, []))
    for i, test_label in enumerate(test_labels):
        test_nv_mid = test_nv_mids[i]
        # 여기서 label은 카탈로그 label
        for j, test_query_label in enumerate(test_query_labels):
            if test_label == test_query_label:  # if valid_label == gt_label
                # [ [ 101, [102,103,104,105] ], [...]]
                test_query_nv_mid_list[j][1].append(test_nv_mid)

    if args.is_train:
        #########################
        # dataset prepare #### --> 이 부분 추후 함수로 대체
        #########################
        trainset = CatalogDataset(args, args.train_parquet_path)
        train_sentences = trainset.sentences()
        train_labels = trainset.labels()
        train_nv_mids = trainset.get_nv_mid()
        query_labels, query_sentences, query_nv_mids, count_dict = query_finder(
            train_labels, train_sentences, train_nv_mids)
        model = SentenceTransformer(args.model_path)
        model.to('cuda')
        InputExample_list = []
        with torch.no_grad():
            train_embeddings = model.encode(
                train_sentences, convert_to_tensor=True)
            query_embeddings = model.encode(
                query_sentences, convert_to_tensor=True)
            print("Query 개수: ", len(query_sentences))
            positive_example = 0
            negative_example = 0
            for i, query_embedding in enumerate(query_embeddings):
                print("{}번째/{} query 처리중".format(i, len(query_embeddings)))

                query_label = query_labels[i]
                query_sentence = query_sentences[i]
                top_k = count_dict[query_label]
                cos_scores = util.pytorch_cos_sim(
                    query_embedding, train_embeddings)[0]
                cos_scores = cos_scores.cpu()
                # We use np.argpartition, to only partially sort the top_k results
                # top_k positive + 1 negative (모델이 잘 학습된 것을 가정)
                top_results = np.argpartition(-cos_scores,
                                              range(top_k + 1))[0:top_k + 1]
                low_correct = top_results[-2]
                high_incorrect = top_results[-1]

                if train_labels[low_correct] == query_label:
                    if cos_scores[low_correct] < 0.1:
                        pass
                    else:
                        InputExample_list.append(InputExample(texts=[
                                                 query_sentence, train_sentences[low_correct]], label=min(cos_scores[low_correct] + 0.1, 0.999)))
                        positive_example += 1
                else:
                    if cos_scores[low_correct] > 0.9:
                        pass
                    else:
                        InputExample_list.append(InputExample(texts=[
                                                 query_sentence, train_sentences[low_correct]], label=max(cos_scores[low_correct] - 0.1, 0.001)))
                        negative_example += 1

                if train_labels[high_incorrect] != query_labels:
                    if cos_scores[high_incorrect] > 0.9:
                        pass
                    else:
                        InputExample_list.append(InputExample(texts=[query_sentence, train_sentences[high_incorrect]], label=max(
                            cos_scores[high_incorrect] - 0.1, 0.001)))
                        negative_example += 1
                else:
                    if cos_scores[high_incorrect] < 0.1:
                        pass
                    else:
                        InputExample_list.append(InputExample(texts=[query_sentence, train_sentences[high_incorrect]], label=min(
                            cos_scores[high_incorrect] + 0.1, 0.999)))
                        positive_example += 1

            print("총 dataset 개수: ", len(InputExample_list))
            print("positve example 비율: ", round(
                positive_example/len(InputExample_list), 4))
            print("negative example 비율: ", round(
                negative_example/len(InputExample_list), 4))
            test(args, model, test_query_labels, test_query_sentences,
                 test_labels, test_sentences, test_count_dict, 0)
            best_score, best_threshold = threshold_test(
                args, model, test_query_sentences, test_query_nv_mids, test_sentences, test_nv_mids, 0, test_query_nv_mid_list)
        for epoch in range(args.epoch):    
            print('STS {}epoch 학습 중'.format(epoch+1))
            print("총 dataset 개수: ", len(InputExample_list))
            print("batch_size: ", args.batch_size)
            print("총 iteration: ", len(InputExample_list)//args.batch_size)
            model.train()
            model = train_iter_STS(args, epoch+1, args.batch_size, model, InputExample_list) ## 여기서부터 고치자
            test(args, model, test_query_labels, test_query_sentences,
                 test_labels, test_sentences, test_count_dict, 0)
            best_score, best_threshold = threshold_test(
                args, model, test_query_sentences, test_query_nv_mids, test_sentences, test_nv_mids, 0, test_query_nv_mid_list)

if __name__ == '__main__':
    main()
