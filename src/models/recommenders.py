"""
Recommender models used for the semi-synthetic experiments
in the paper "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback".
"""
from __future__ import absolute_import, print_function

from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf


class AbstractRecommender(metaclass=ABCMeta):
    """Abstract base class for evaluator class."""

    @abstractmethod
    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        raise NotImplementedError()

    @abstractmethod
    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        raise NotImplementedError()

    @abstractmethod
    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        raise NotImplementedError()


class PointwiseRecommender(AbstractRecommender):
    """Implicit Recommenders based on pointwise approach."""

    def __init__(self, model_name:str, num_users: int, num_items: int,
                 dim: int, lam: float, eta: float, weight: float = 1.0, clip: float = 0, num: int = 0) -> None:
        """Initialize Class."""
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.lam = lam
        self.eta = eta
        self.weight = weight
        self.clip = clip
        self.num = num

        # Build the graphs
        self.create_placeholders()
        self.build_graph()

        if model_name in ["mf"]:
            self.create_mf_losses()
        elif model_name in ["rel-mf"]:
            self.create_rel_mf_losses()
        else: # mf-du
            self.create_mf_du_losses()

        self.add_optimizer()

    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        self.users = tf.placeholder(tf.int32, [None], name='user_placeholder')
        self.items = tf.placeholder(tf.int32, [None], name='item_placeholder')
        self.scores = tf.placeholder(
            tf.float32, [None, 1], name='score_placeholder')
        self.inv_scores = tf.placeholder(
            tf.float32, [None, 1], name='inv_score_placeholder')
        self.scores_normalization = tf.placeholder(
            tf.float32, [None, 1], name='scores_normalization')
        self.inv_scores_normalization = tf.placeholder(
            tf.float32, [None, 1], name='inv_scores_normalization')

        self.labels = tf.placeholder(
            tf.float32, [None, 1], name='label_placeholder')


    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        with tf.name_scope('embedding_layer'):
            # initialize user-item matrices and biases
            self.user_embeddings = tf.get_variable(
                f'user_embeddings', shape=[self.num_users, self.dim],
                initializer=tf.contrib.layers.xavier_initializer())

            self.item_embeddings = tf.get_variable(
                f'item_embeddings', shape=[self.num_items, self.dim],
                initializer=tf.contrib.layers.xavier_initializer())

            # lookup embeddings of current batch
            self.u_embed = tf.nn.embedding_lookup(
                self.user_embeddings, self.users)
            self.i_embed = tf.nn.embedding_lookup(
                self.item_embeddings, self.items)

        with tf.variable_scope('prediction'):
            self.logits = tf.reduce_sum(
                tf.multiply(self.u_embed, self.i_embed), 1)
            self.preds = tf.sigmoid(tf.expand_dims(
                self.logits, 1))

    def create_mf_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope('mflosses'):
            self.mse = tf.reduce_sum(
                (self.labels) * tf.square(1. - self.preds) +
                (1 - self.labels) * tf.square(self.preds)  ) / \
                tf.reduce_sum(self.labels + (1 - self.labels))

            # add the L2-regularizer terms.
            reg_term_embeds = tf.nn.l2_loss(self.user_embeddings) \
                + tf.nn.l2_loss(self.item_embeddings)
            self.loss = self.mse + self.lam * reg_term_embeds

    def create_rel_mf_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope('crmflosses'):
            # clipping
            scores = tf.clip_by_value(
                self.scores, clip_value_min=0.1, clip_value_max=1.0)

            self.mse = tf.reduce_sum(
                (self.labels / scores) * tf.square(1. - self.preds) +
                (1 - self.labels / scores) * tf.square(self.preds)) / \
                tf.reduce_sum(self.labels + (1 - self.labels))

            # add the L2-regularizer terms.
            reg_term_embeds = tf.nn.l2_loss(self.user_embeddings) \
                + tf.nn.l2_loss(self.item_embeddings)
            self.loss = self.mse + self.lam * reg_term_embeds

    def create_mf_du_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope('mf_du'):
            eps = 0.000001

            # clipping
            scores = tf.clip_by_value(
                self.scores, clip_value_min=0.1, clip_value_max=1.0)
            inv_scores = tf.clip_by_value(
                self.inv_scores, clip_value_min=0.1, clip_value_max=1.0)

            # loss with SNIPS
            self.mse = tf.reduce_sum(
                (self.labels) * 1/(tf.reduce_sum((self.labels)*1/(scores))+eps) * (self.labels)/(scores)  * tf.square(1. - self.preds) +
                (1 - self.labels)* 1/(tf.reduce_sum((1 - self.labels)*1/(inv_scores))+eps) *  (1 - self.labels) / (inv_scores) * tf.square(self.preds)
                ) / \
                tf.reduce_sum(self.labels + (1 - self.labels))

            # add the L2-regularizer terms.
            reg_term_embeds = tf.nn.l2_loss(self.user_embeddings) \
                + tf.nn.l2_loss(self.item_embeddings)

            self.loss = self.mse + self.lam * reg_term_embeds


    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        with tf.name_scope('optimizer'):
            # set Adam Optimizer.
            self.apply_grads = tf.train.AdamOptimizer(
                learning_rate=self.eta).minimize(self.loss)
