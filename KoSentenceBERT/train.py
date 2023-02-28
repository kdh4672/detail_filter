import numpy as np
import torch
from torch.utils.data import DataLoader

from sentence_transformers import datasets, SentencesDataset, losses, util
from sentence_transformers.readers import InputExample
from evaluation import evaluate
from util import result_maker_for_test

def train_iter_STS(args, epoch, batch_size, model, InputExample_list):
    print('total_epoch:', epoch)
    print('batch_size:', batch_size)
    train_data = SentencesDataset(InputExample_list, model)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=None,
        epochs=5,
        optimizer_params={'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
        save_best_model=False,
        scheduler = 'constantlr',
    )
    return model

def test(args, model, query_labels, query_sentences, test_labels, test_sentences, count_dict, epoch):
    model.eval()
    with torch.no_grad():
        corpus = test_sentences
        top_k = 10
        corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
        print("Query 개수: ", print(len(query_sentences)))

        incorrect_max_score_list = []
        correct_min_score_list = []

        hard_incorrect = 0
        for i,query in enumerate(query_sentences):
            query_embedding = model.encode(query, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
            cos_scores = cos_scores.cpu()
            #We use np.argpartition, to only partially sort the top_k results
            top_k = count_dict[query_labels[i]] + 5
            top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
            # print("\n======================")
            # print("Query:", query, '[{}]'.format(query_labels[i]))
            # print('이 쿼리의 동일 label 개수:{}'.format(count_dict[query_labels[i]]))
            
            # print("\nTop 10 most similar sentences in corpus:")
            incorrect_max_score = -1
            correct_min_score = -1
            ims = -1
            cms = -1
            first_incorrect = True
            for e, idx in enumerate(top_results[0:top_k]):
                if query_labels[i] == test_labels[idx]:
                    answer = 'Correct!!'
                    correct_min_score = "(Score: %.4f)" % (cos_scores[idx])
                    cms = cos_scores[idx]
                else:
                    answer = 'Incorrect'        
                    if first_incorrect == True:
                        incorrect_max_score = "(Score: %.4f)" % (cos_scores[idx])
                        ims = cos_scores[idx]
                        first_incorrect = False
                        if e < count_dict[query_labels[i]]:
                            hard_incorrect+=1
                # print(corpus[idx].strip(), '[{}]'.format(test_labels[idx]), "(Score: %.4f)" % (cos_scores[idx]), answer)
                # print("correct_min_score:",correct_min_score)    
                # print("incorrect_max_score:",incorrect_max_score)
                # print('\n')
            if incorrect_max_score == -1:
                pass
            else:
                correct_min_score_list.append(cms)
                if ims > 0.9: ## mislabelling 제거
                    pass
                else:
                    incorrect_max_score_list.append(ims)
        print("correct_min_score 평균:", round(np.mean(correct_min_score_list),4))   
        print("incorrect_max_score 평균:", round(np.mean(incorrect_max_score_list),4))
        print("correct중 min: ", np.min(correct_min_score_list))
        print("incorrect중 max: ", np.max(incorrect_max_score_list))
        print("incorrect가 correct보다 넘는 비율:", round(hard_incorrect/len(query_sentences),4))
        print("\n======================")

        # nsml.report(summary=True, step=epoch, epoch_total=args.epoch,\
        # smallest_correct=float(round(np.min(correct_min_score_list),4)),\
        # correct_min_mean=float(round(np.mean(correct_min_score_list),4)),\
        # incorrect_max_mean=float(round(np.mean(incorrect_max_score_list),4)),\
        # min_max_difference= float(round(np.mean(correct_min_score_list)-np.mean(incorrect_max_score_list),4)),\
        # largest_incorrect=float(round(np.max(incorrect_max_score_list),4)),\
        # hard_incorrect_ratio=float(round(hard_incorrect/len(query_sentences),4)))

def threshold_test(args, model, test_query_sentences, test_query_nv_mids, test_sentences, test_nv_mids, epoch, test_query_nv_mid_list):
    print('threshold test start!!')
    model.eval()
    best_score = -1
    with torch.no_grad():
        test_query_embeddings = model.encode(test_query_sentences)
        test_embeddings = model.encode(test_sentences)
    threshold_list = [0.5 + 0.01 * x for x in range(10)]
    print("epoch:{} threshold_test 시작".format(epoch))
    for threshold in threshold_list:
        result = result_maker_for_test(test_embeddings, test_nv_mids, test_query_embeddings, test_query_nv_mids, threshold=threshold)
    ## 추측 list
        score = evaluate(result, test_query_nv_mid_list)
        print('threshold:{}'.format(threshold),'Score:{}'.format(score))
        best_score = max(score,best_score)
        if score == best_score:
            dict_ = {best_score:threshold}
            best_threshold = threshold
    print('best_threshold: ', best_threshold, 'best_score: ', best_score)
    return best_score, best_threshold