"""
    Evaluation of DREAM
"""

import time
import random
import pickle
import torch
import numpy as np
from config import Config
from utils import data_helpers as dh

logger = dh.logger_fn("torch-log", "logs/test-{0}.log".format(time.asctime()))

MODEL = input("☛ Please input the model file you want to test: ")

while not (MODEL.isdigit() and len(MODEL) == 10):
    MODEL = input("✘ The format of your input is illegal, it should be like(1490175368), please re-input: ")
logger.info("✔︎ The format of your input is legal, now loading to next step...")

MODEL_DIR = dh.load_model_file(MODEL)


def eval():
    # Load data
    logger.info("✔︎ Loading data...")

    logger.info("✔︎ Training data processing...")
    train_data = dh.load_data(Config().TRAININGSET_DIR)

    logger.info("✔︎ Test data processing...")
    test_data = dh.load_data(Config().TESTSET_DIR)

    users = test_data.userID.values

    logger.info("✔︎ Load negative sample...")
    with open(Config().NEG_SAMPLES, 'rb') as handle:
        neg_samples = pickle.load(handle)

    # Load model
    dr_model = torch.load(MODEL_DIR)

    def hit_ratio(K):  # HR@k
        item_embedding = dr_model.encode.weight
        hidden = dr_model.init_hidden(Config().batch_size)
        avg_ratio = 0
        for i, x in enumerate(dh.batch_iter(train_data, Config().batch_size, Config().seq_len, shuffle=False)):
            uids, baskets, lens = x
            dynamic_user, _ = dr_model(baskets, lens, hidden)
            for uid, l, du in zip(uids, lens, dynamic_user):
                du_latest = du[l - 1].unsqueeze(0)

                # calculating <u,p> score for all test items <u,p> pair
                u_items = test_data[test_data['userID'] == uid].baskets.values[0]  # list  dim 1

                index_pos = set(np.arange(len(u_items)))

                neg_items = random.sample(list(neg_samples[uid]), 1000)
                u_items.extend(neg_items)

                u_items = torch.LongTensor(u_items)

                score_ui = torch.mm(du_latest, item_embedding[u_items].t())
                score_ui = list(score_ui.data.numpy()[0])
                index_k = []
                for k in range(K):
                    index = score_ui.index(max(score_ui))
                    index_k.append(index)
                    score_ui[index] = -99999
                ratio = len((index_pos & set(index_k)))/K
                avg_ratio = avg_ratio + ratio
        print('Hit ratio', K, avg_ratio/len(users))

    hit_ratio(3)


if __name__ == '__main__':
    eval()


