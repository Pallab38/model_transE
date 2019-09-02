import numpy as np
import os
import time
import torch
from torch.utils.data import DataLoader

from loading_dataset import ReadDataset, gen_corrupt_triples
from model_transE import TranslationalEmbedding
from utility import calc_mean_rank, calc_hit_rank_N, calc_mean_raw_rank

def training_model(data_dir, embedding_dim, batch_size, learning_rate, max_epoch, gamma = 1):
    np.random.seed(13)
    weight_decay = 0.002
    data = ReadDataset(data_dir)
    model = TranslationalEmbedding(data,embedding_dim=embedding_dim, learning_rate=learning_rate,gamma=gamma)
    optimizer = torch.optim.Adagrad(model.parameters(), model.learning_rate, weight_decay)
    path = data_dir

    for epoch in range(max_epoch):
        start = time.time()
        actual_loss =0
        dataLoader = DataLoader(np.array(data.train_triples), batch_size, True)

        for batch in dataLoader:
            if(batch.shape[0] < batch_size):
                break
            optimizer.zero_grad()
            pos_batch, neg_batch = gen_corrupt_triples(batch, data.num_entity, data.head_rel2_tail, data.tail_rel2_head)
            output = model.forward(pos_batch, neg_batch)
            pos_score = output.view(2,-1)[0]
            neg_score = output.view(2,-1)[1]
            loss = model.log_loss(pos_score, neg_score, temp =0)
            loss.backward(retain_graph=True)
            optimizer.step()

            actual_loss+=loss.item()

        print("epoch: " + str(epoch) + ", Learning Rate:" + str(learning_rate) + " , loss: " + str(actual_loss))
        end = time.time()
        print("Time taken :", str((end-start)/60)+ " mins")

    if (epoch+1 == max_epoch):
        torch.save(model.state_dict(), os.path.join(path, 'parameters{:.0f}.pkl'.format(epoch)))
        f = open(os.path.join(path, 'results{:.0f}.txt'.format(epoch)), 'w')
        print("Starting ranking.....")

        head_rank = model.head_rank(test_triples=data.test_triples)
        tail_rank = model.tail_rank(test_triples=data.test_triples)
        rank = head_rank+tail_rank
        mean_rank = calc_mean_rank(rank)
        mean_raw_rank = calc_mean_raw_rank(rank)
        hit_rank_5 = calc_hit_rank_N(rank,5)
        hit_rank_10 = calc_hit_rank_N(rank,10)

        f.write('Mean Rank: {:.0f}\n'.format(mean_rank))
        f.write('Mean RR: {:.4f}\n'.format(mean_raw_rank))
        f.write('Hit@1: {:.4f}\n'.format(hit_rank_5))
        f.write('Hit@3: {:.4f}\n'.format(hit_rank_10))

        for loss in actual_loss:
            f.write(str(loss))
            f.write('\n')
        f.close()

    print("Training Finished")


    return rank


if __name__ =='__main__':

    rank = training_model(data_dir="C:\SDA\SUBMIT\wn18", embedding_dim=50, batch_size=2000, learning_rate=0.01,
                          max_epoch=50)
    np.savetxt('rank.txt', rank)
