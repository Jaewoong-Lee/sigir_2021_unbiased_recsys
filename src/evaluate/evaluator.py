"""Evaluate Implicit Recommendation models."""
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from scipy import sparse
import numpy as np
import pandas as pd

from .metrics import average_precision_at_k, dcg_at_k, recall_at_k

class PredictRankings:
    """Predict rankings by trained recommendations."""

    def __init__(self, user_embed: np.ndarray, item_embed: np.ndarray) -> None:
        """Initialize Class."""
        # latent embeddings
        self.user_embed = user_embed
        self.item_embed = item_embed

    def predict(self, users: np.array, items: np.array) -> np.ndarray:
        """Predict scores for each user-item pairs."""
        # predict ranking score for each user
        user_emb = self.user_embed[users].reshape(1, self.user_embed.shape[1])
        item_emb = self.item_embed[items]
        scores = (user_emb @ item_emb.T).flatten()
        return scores

def aoa_evaluator(user_embed: np.ndarray,
                  item_embed: np.ndarray,
                  train: np.ndarray,
                  test: np.ndarray,
                  num_users: int, 
                  num_items: int,
                  model_name: str,
                  at_k: List[int] = [1, 3, 5],
                  only_dcg: bool = False) -> pd.DataFrame:
    """Calculate ranking metrics with average-over-all evaluator."""
    
    # test data
    users = test[:, 0]
    items = test[:, 1]
    relevances = test[:, 2] # actual relevance 

    # define model
    dim = user_embed.shape[1]
    model = PredictRankings(user_embed=user_embed, item_embed=item_embed)

    # prepare ranking metrics
    if only_dcg:
        metrics = {'NDCG': dcg_at_k}
    else: 
        metrics = {'NDCG': dcg_at_k,
           'Recall': recall_at_k,
           'MAP': average_precision_at_k}

    results = {}
    for k in at_k:
        for metric in metrics:
            results[f'{metric}@{k}'] = []

    # calculate ranking metrics
    for user in set(users):
        indices = users == user
        pos_items = items[indices] # item, relevance
        rel = relevances[indices]
        if len(rel) < max(at_k):
            print("no eval")
            continue
        # predict ranking score for each user
        scores = model.predict(users=user, items=pos_items)
        for k in at_k:
            for metric, metric_func in metrics.items():
                results[f'{metric}@{k}'].append(metric_func(rel, scores, k))

        # aggregate results
        results_df = pd.DataFrame(index=results.keys())
        results_df[f'{model_name}_{dim}'] = list(map(np.mean, list(results.values()))) 

    return results_df.sort_index()

# evaluation using pscore 
def unbiased_evaluator(user_embed: np.ndarray,
                  item_embed: np.ndarray,
                  train: np.ndarray,
                  test: np.ndarray,
                  num_users: int, 
                  num_items: int,
                  pscore: np.ndarray,
                  model_name: str,
                  val: np.ndarray,
                  flag_test: bool,
                  flag_unbiased: bool,
                  at_k: List[int] = [1, 3, 5],
                  only_dcg: bool = False) -> pd.DataFrame:
                  
    """Calculate ranking metrics by unbiased evaluator."""
    # test data
    users = test[:, 0]
    items = test[:, 1]
    if flag_test:
        train_val = np.r_[train, val, test]
    else:
        train_val = np.r_[train, test]
    positive_pairs = train_val[train_val[:, 2] == 1, :2]
    
    dim = user_embed.shape[1]
    model = PredictRankings(user_embed=user_embed, item_embed=item_embed)

    # prepare ranking metrics
    if only_dcg:
        metrics = {'NDCG': dcg_at_k}
    else: 
        metrics = {'NDCG': dcg_at_k,
           'Recall': recall_at_k,
           'MAP': average_precision_at_k}

    results = {}
    for k in at_k:
        for metric in metrics:
            results[f'{metric}@{k}'] = []

    unique_items = np.asarray( range( num_items ) )

    # calculate ranking metrics
    for user in set(users):
        indices = users == user
        pos_items = items[indices] 
        all_pos_items = positive_pairs[positive_pairs[:, 0] == user, 1] 
        neg_items = np.setdiff1d(unique_items, all_pos_items) 
        used_items = np.r_[pos_items, neg_items]
        pscore_ = pscore[used_items] 
        relevances = np.r_[np.ones_like(pos_items), np.zeros_like(neg_items)]

        # calculate an unbiased DCG score for a user
        scores = model.predict(users=user, items=used_items)
        for k in at_k:
            for metric, metric_func in metrics.items():
                if flag_unbiased:
                    results[f'{metric}@{k}'].append(metric_func(relevances, scores, k, pscore_)) 
                else:
                    results[f'{metric}@{k}'].append(metric_func(relevances, scores, k, None))

        # aggregate results
        results_df = pd.DataFrame(index=results.keys())
        results_df[f'{model_name}_{dim}'] = list(map(np.mean, list(results.values()))) 

    return results_df.sort_index()