"""
Codes for training recommenders on semi-synthetic datasets used in the semi-synthetic experiments
in the paper "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback".
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops

import math
from evaluate.evaluator import aoa_evaluator, unbiased_evaluator
from models.recommenders import PointwiseRecommender
from datetime import datetime
from tqdm import tqdm

def pointwise_trainer(sess: tf.Session, data: str, model: PointwiseRecommender,
                      train: np.ndarray, val: np.ndarray, test: np.ndarray, num_users, num_items, 
                      pscore: np.ndarray, inv_pscore: np.ndarray, item_freq: np.ndarray,
                      max_iters: int = 1000, batch_size: int = 2**12, 
                      model_name: str = 'rel-mf', date_now: str = '1'):
    """Train and Evaluate Implicit Recommender."""
    train_loss_list = []
    test_dcg_list = []
    test_map_list = []
    test_recall_list = []

    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # pscore for train
    pscore = pscore[train[:, 1].astype(np.int)]

    # train the given implicit recommender
    max_score = 0
    er_stop_count = 0
    best_u_emb =[]
    best_i_emb =[]

    all_tr = np.arange(len(train))
    batch_size = batch_size
    early_stop = 5

    for i in np.arange(max_iters):
        np.random.RandomState(12345).shuffle(all_tr)
        batch_num = int(len(all_tr) / batch_size) +1
        for b in range(batch_num):
            # mini-batch samples
            train_batch = train[all_tr[b*batch_size : (b+1)*batch_size]]
            train_label = train_batch[:, 2]
            train_score = pscore[all_tr[b*batch_size : (b+1)*batch_size]]
            train_inv_score = inv_pscore[train_batch[:, 1]]

            # update user-item latent factors and calculate training loss
            _, loss = sess.run([model.apply_grads, model.mse],
                            feed_dict={model.users: train_batch[:, 0],
                                        model.items: train_batch[:, 1],
                                        model.labels: np.expand_dims(train_label, 1),
                                        model.scores: np.expand_dims(train_score, 1),
                                        model.inv_scores: np.expand_dims(train_inv_score, 1),
                                        # model.scores_normalization: np.expand_dims(train_score, 1), ################
                                        # model.inv_scores_normalization: np.expand_dims(train_inv_score, 1) ################
                                        })
        ############### evaluation
        if i % 1 == 0:
            print(i,":  ", loss)

            u_emb, i_emb = sess.run(
                [model.user_embeddings, model.item_embeddings])

            # validation
            at_k = 3
            
            # We need an unbiased evaluation for validating
            val_ret = unbiased_evaluator(user_embed=u_emb, item_embed=i_emb, 
                                    train=train, val=val, test=val, num_users=num_users, num_items=num_items, 
                                    pscore=item_freq, model_name=model_name, at_k=[at_k], flag_test=False, flag_unbiased = True)

            dim = u_emb.shape[1]
            best_score = val_ret.loc[f'MAP@{at_k}', f'{model_name}_{dim}']

            if max_score < best_score:
                max_score = best_score
                print(f"best_val_MAP@{at_k}: ", max_score)
                er_stop_count = 0
                
                best_u_emb = u_emb
                best_i_emb = i_emb

            else:
                er_stop_count += 1
                if er_stop_count > early_stop:
                    print("stopped!")
                    break

    sess.close()
    return best_u_emb, best_i_emb



class Trainer:
    """Trainer Class for ImplicitRecommender."""
    now = datetime.now()
    date_now = "%02d%02d_%02d%02d%02d" %(now.month, now.day, now.hour, now.minute, now.second)

    def __init__(self, data: str, random_state: list, hidden:int, max_iters: int = 1000, lam: float=1e-4, batch_size: int = 1024, \
                 clip: float = 0.1, eta: float = 0.1, model_name: str = 'mf') -> None:

        """Initialize class."""
        self.data = data
        self.at_k = [1, 3, 5]
        self.dim = int(hidden[0])
        self.lam = lam
        self.clip = clip
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.eta = eta
        self.model_name = model_name
        self.random_state = [r for r in range(1, int(random_state[0]) + 1)]

        print("======================================================")
        print("random state: ", self.random_state)
        print("======================================================")

    def run(self) -> None:
        print("======================================================")
        print("date: ", self.date_now)
        print("======================================================")

        """Train pointwise implicit recommenders."""
        train = np.load(f'../data/{self.data}/point_0.5_0.5/train.npy')
        val = np.load(f'../data/{self.data}/point_0.5_0.5/val.npy')
        test = np.load(f'../data/{self.data}/point_0.5_0.5/test.npy')
        pscore = np.load(f'../data/{self.data}/point_0.5_0.5/pscore.npy')
        inv_pscore = np.load(f'../data/{self.data}/point_0.5_0.5/inv_pscore.npy')
        item_freq = np.load(f'../data/{self.data}/point_0.5_0.5/item_freq.npy')
        num_users = np.int(train[:, 0].max() + 1)
        num_items = np.int(train[:, 1].max() + 1)

        ret_path = Path(f'../logs/{self.data}/{self.model_name}/results/')
        ret_path.mkdir(parents=True, exist_ok=True)

        sub_results_sum = pd.DataFrame()
        for random_state in tqdm(self.random_state):
            print("random seed now :", random_state)
            sub_results = pd.DataFrame(index=['NDCG@1', 'NDCG@3', 'NDCG@5', 'MAP@1', 'MAP@3', 'MAP@5', 'Recall@1', 'Recall@3', 'Recall@5'])

            ###### cpu
            tf.reset_default_graph()
            ops.reset_default_graph()
            tf.set_random_seed(random_state)
            sess = tf.Session()

            # train                  
            model = PointwiseRecommender(model_name=self.model_name,
                num_users=num_users, num_items=num_items,
                clip=self.clip, dim=self.dim, lam=self.lam, eta=self.eta)
            u_emb, i_emb = pointwise_trainer(
                sess, data=self.data, model=model, train=train, val = val, test=test, 
                num_users=num_users, num_items=num_items, pscore=pscore, inv_pscore=inv_pscore,
                max_iters=self.max_iters, batch_size=self.batch_size, item_freq=item_freq, 
                model_name=self.model_name, date_now=self.date_now)

            # test
            results = aoa_evaluator(user_embed=u_emb, item_embed=i_emb,
                                train=train, test=test, num_users=num_users, num_items=num_items,
                                model_name=self.model_name, at_k=self.at_k)

            sub_results_sum = sub_results_sum.add(results, fill_value=0) 
        sub_results_mean = sub_results_sum / len(self.random_state)
        sub_results_mean.to_csv(ret_path / f'{self.date_now}_dim_{self.dim}_lr_{self.eta}_reg_{self.lam}_{self.batch_size}_random_{len(self.random_state)}.csv')
