# -*- coding: utf-8 -*-
"""
    train model

    @TODO
     - Optimizer Choosing
     - Hyper-parameter tuning
"""

import os
import random
import time
import pickle
import torch
import numpy as np
from math import ceil
from utils import data_helpers as dh
from config import Config
from rnn_model import DRModel


logger = dh.logger_fn("torch-log", "logs/training-{0}.log".format(time.asctime()))


def train():
    # Load data
    logger.info("✔︎ Loading data...")

    logger.info("✔︎ Training data processing...")
    train_data = dh.load_data(Config().TRAININGSET_DIR)

    logger.info("✔︎ Validation data processing...")
    validation_data = dh.load_data(Config().VALIDATIONSET_DIR)

    logger.info("✔︎ Load negative sample...")
    with open(Config().NEG_SAMPLES, 'rb') as handle:
        neg_samples = pickle.load(handle)

    # Model config
    dr_model = DRModel(Config())

    # Optimizer
    optimizer = torch.optim.Adam(dr_model.parameters(), lr=Config().learning_rate)

    def adjust_learning_rate(optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = Config().learning_rate * 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def bpr_loss(uids, baskets, dynamic_user, item_embedding):
        """
        Bayesian personalized ranking loss for implicit feedback.

        Args:
            uids: batch of users' ID
            baskets: batch of users' baskets
            dynamic_user: batch of users' dynamic representations
            item_embedding: item_embedding matrix
        """
        nll = 0
        for uid, bks, du in zip(uids, baskets, dynamic_user):
            du_p_product = torch.mm(du, item_embedding.t())  # shape: [pad_len, num_item]
            nll_u = []  # nll for user
            for t, basket_t in enumerate(bks):
                if basket_t[0] != 0 and t != 0:
                    pos_idx = torch.LongTensor(basket_t)

                    # Sample negative products
                    neg = random.sample(list(neg_samples[uid]), len(basket_t))
                    neg_idx = torch.LongTensor(neg)

                    # Score p(u, t, v > v')
                    score = du_p_product[t - 1][pos_idx] - du_p_product[t - 1][neg_idx]

                    # Average Negative log likelihood for basket_t
                    nll_u.append(torch.mean(-torch.nn.LogSigmoid()(score)))
            for i in nll_u:
                nll = nll + i / len(nll_u)
        avg_nll = torch.div(nll, len(baskets))
        return avg_nll

    def train_model():
        dr_model.train()  # turn on training mode for dropout
        dr_hidden = dr_model.init_hidden(Config().batch_size)
        train_loss = 0
        start_time = time.clock()
        num_batches = ceil(len(train_data) / Config().batch_size)
        for i, x in enumerate(dh.batch_iter(train_data, Config().batch_size, Config().seq_len, shuffle=True)):
            uids, baskets, lens = x
            dr_model.zero_grad()  # 如果不置零，Variable 的梯度在每次 backward 的时候都会累加
            dynamic_user, _ = dr_model(baskets, lens, dr_hidden)

            loss = bpr_loss(uids, baskets, dynamic_user, dr_model.encode.weight)
            loss.backward()

            # Clip to avoid gradient exploding
            torch.nn.utils.clip_grad_norm_(dr_model.parameters(), Config().clip)

            # Parameter updating
            optimizer.step()
            train_loss += loss.data

            # Logging
            if i % Config().log_interval == 0 and i > 0:
                elapsed = (time.clock() - start_time) / Config().log_interval
                cur_loss = train_loss.item() / Config().log_interval  # turn tensor into float
                train_loss = 0
                start_time = time.clock()
                logger.info('[Training]| Epochs {:3d} | Batch {:5d} / {:5d} | ms/batch {:02.2f} | Loss {:05.2f} |'
                            .format(epoch, i, num_batches, elapsed, cur_loss))

    def validate_model():
        dr_model.eval()
        dr_hidden = dr_model.init_hidden(Config().batch_size)
        val_loss = 0
        start_time = time.clock()
        num_batches = ceil(len(validation_data) / Config().batch_size)
        for i, x in enumerate(dh.batch_iter(validation_data, Config().batch_size, Config().seq_len, shuffle=False)):
            uids, baskets, lens = x
            dynamic_user, _ = dr_model(baskets, lens, dr_hidden)
            loss = bpr_loss(uids, baskets, dynamic_user, dr_model.encode.weight)
            val_loss += loss.data

        # Logging
        elapsed = (time.clock() - start_time) * 1000 / num_batches
        val_loss = val_loss.item() / num_batches
        logger.info('[Validation]| Epochs {:3d} | Elapsed {:02.2f} | Loss {:05.2f} |'
                    .format(epoch, elapsed, val_loss))
        return val_loss

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    checkpoint_dir = out_dir + '/model-{epoch:02d}-{loss:.4f}.model'

    best_val_loss = None

    try:
        # Training
        for epoch in range(Config().epochs):
            train_model()
            logger.info('-' * 89)

            val_loss = validate_model()
            logger.info('-' * 89)

            # Checkpoint
            if not best_val_loss or val_loss < best_val_loss:
                with open(checkpoint_dir.format(epoch=epoch, loss=val_loss), 'wb') as f:
                    torch.save(dr_model, f)
                best_val_loss = val_loss
            else:
                # Manual SGD slow down lr if no improvement in val_loss
                adjust_learning_rate(optimizer)
                pass
    except KeyboardInterrupt:
        logger.info('*' * 89)
        logger.info('Early Stopping!')


if __name__ == '__main__':
    train()
