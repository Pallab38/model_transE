import numpy as np
import os
import time
import torch
from torch.utils.data import DataLoader

from loading_dataset import ReadDataset, gen_corrupt_triples
from model_transE import TranslationalEmbedding
from utility import *

def training_model(data_dir, embedding_dim, batch_size, learning_rate, max_epoch, gamma = 2):
    np.random.seed(13)
    data =  ReadDataset(data_dir)
    model = TranslationalEmbedding(data,embedding_dim=embedding_dim, learning_rate=learning_rate,gamma=gamma)
    optimizer = torch.optim.SGD(model.parameters(), model.learning_rate)
    path = data_dir

    loss_list  =[]
    for epoch in range(max_epoch):
        start = time.time()
        actual_loss = []

        dataLoader = DataLoader(np.array(data.train_triples), batch_size, True)
        for batch in dataLoader:
            if (batch.shape[0] < batch_size):
                break
            optimizer.zero_grad()
            pos_batch, neg_batch = gen_corrupt_triples(batch, data.num_entity, data.head_rel2_tail, data.tail_rel2_head)
            output = model.forward(pos_batch, neg_batch)
            pos_score = output.view(2,-1)[0]
            neg_score = output.view(2,-1)[1]

            loss = model.loss(pos_score, neg_score)
            loss.mean().backward()
            optimizer.step()
            actual_loss.append(loss.mean())

        print("epoch: " + str(epoch+1) + ", Learning Rate:" + str(learning_rate) + " , loss: " + str(loss.mean()))
        end = time.time()
        print("Time taken :", str(round((end-start)/60))+ " mins")
        loss_list.append(loss.mean())
    np.savetxt('losses.txt', loss_list)



    if (epoch+1 == max_epoch):
        torch.save(model.state_dict(), os.path.join(path, 'parameters{:.0f}.pkl'.format(epoch)))
        f = open(os.path.join(path, 'results{:.0f}.txt'.format(epoch)), 'w')

        model.eval()
        batch_size = 10
        dataLoader = DataLoader(np.array(data.test_triples), batch_size, True)
        examples_count =0.0
        global rank_10
        rank_10 = 0.0
        mean_ranks = 0.0
        start = time.time()

        #for triple in data.valid_triples:
        for batch in dataLoader:
            if (batch.shape[0] < batch_size):
                break
            prediction,grnd_truth = model.rank_entity(batch)
            rank_10 += hit_at_k(predictions=prediction, ground_truth=grnd_truth, k=10)
            mean_ranks += gen_mean_rank(prediction,grnd_truth)
            examples_count += prediction.size()[0]


        print("Example Count: ", examples_count)
        hit_rank_10 = rank_10 / examples_count * 100
        print("Hit Rank 10: ",hit_rank_10)
        mean_rank = mean_ranks /examples_count * 100
        print("Mean Rank(examples_count * 100): ",mean_rank)

        end = time.time()
        print("Time taken :", str(round((end - start) / 60)) + " mins")
        f.write('Hit@10: {:.4f}\n'.format(hit_rank_10))
        f.write('Mean Rank: {:.4f}\n'.format(mean_rank))

        for loss in actual_loss:
            f.write(str(loss))
            f.write('\n')
        f.close()

    print("Training Finished")

    return [hit_rank_10,mean_rank]


if __name__ =='__main__':
    ranks =[]
    ranks = training_model(data_dir="wn18", embedding_dim=20, batch_size=512, learning_rate=0.01,
                          max_epoch=1000)
    np.savetxt('rank.txt', ranks)