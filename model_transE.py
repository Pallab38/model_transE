import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')


class TranslationalEmbedding(nn.Module):
    def __init__(self, data, embedding_dim, learning_rate, gamma):
        super(TranslationalEmbedding,self).__init__()
        self.data = data
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.relu = nn.ReLU()

        self.all_pos = {}
        for i in data.test_triples:
            self.all_pos[i[0], i[1], i[2]] = 1
        for i in data.train_triples:
            self.all_pos[i[0], i[1], i[2]] = 1
        for i in data.valid_triples:
            self.all_pos[i[0], i[1], i[2]] = 1

        bound = 6./ math.sqrt(self.embedding_dim)

        self.entity_emb = nn.Embedding(self.data.num_entity, self.embedding_dim)
        self.entity_emb.weight.data.uniform_(-bound, bound)
        F.normalize(self.entity_emb.weight.data, p = 2)

        self.relation_emb = nn.Embedding(self.data.num_relation, self.embedding_dim)
        self.relation_emb.weight.data.uniform_(-bound, bound)
        F.normalize(self.relation_emb.weight.data, p = 2)



    def forward(self, pos_triples, neg_triples):
        pos_h_emb = self.entity_emb(torch.LongTensor(pos_triples['h'])).to(device)
        pos_r_emb = self.relation_emb(torch.LongTensor(pos_triples['r'])).to(device)
        pos_t_emb = self.entity_emb(torch.LongTensor(pos_triples['t'])).to(device)

        pos_h_emb = pos_h_emb.view(-1, self.embedding_dim)
        pos_r_emb = pos_r_emb.view(-1, self.embedding_dim)
        pos_t_emb = pos_t_emb.view(-1, self.embedding_dim)

        neg_h_emb = self.entity_emb(torch.LongTensor(neg_triples['h'])).to(device)
        neg_r_emb = self.relation_emb(torch.LongTensor(neg_triples['r'])).to(device)
        neg_t_emb = self.entity_emb(torch.LongTensor(neg_triples['t'])).to(device)

        neg_h_emb = neg_h_emb.view(-1, self.embedding_dim)
        neg_r_emb = neg_r_emb.view(-1, self.embedding_dim)
        neg_t_emb = neg_t_emb.view(-1, self.embedding_dim)

        neg_score = torch.norm((neg_h_emb + neg_r_emb - neg_t_emb),2, 1).to(device)
        pos_score = torch.norm((pos_h_emb + pos_r_emb - pos_t_emb), 2, 1).to(device)


        return torch.cat((pos_score,neg_score))

    def rank_loss(self, y_pos, y_neg, temp):
        M = y_pos.size(0)
        N = y_neg.size(0)
        C = int(N / M)
        y_pos = y_pos.repeat(C)
        y_pos = y_pos.view(C, -1).transpose(0, 1)
        y_neg = y_neg.view(C, -1).transpose(0, 1)
        p = F.softmax(-1 * temp * y_neg)
        loss = torch.sum(p * self.relu(y_pos + self.gamma - y_neg)) / M
        return loss


    def head_rank(self, test_triples):
        rank = []

        for triple in test_triples:
            tmp_data = np.ones([self.data.num_entity, 3])
            for i in range(0, self.data.num_entity):
                tmp_data[i,0] = i
                tmp_data[i,1] = triple[1]
                tmp_data[i,2] = triple[2]
            score = self.calculate_score(tmp_data)
            score, ranked_heads = torch.topk(score,self.data.num_entity, largest=False)

            triple_rank = 1

            for tmp_head in ranked_heads:
                if tmp_head.cpu().numpy() == triple[0]:

                    break
                else:
                    triple_rank += 1
            rank.append(triple_rank)

        return rank


    def tail_rank(self, test_triples):
        rank = []

        for triple in test_triples:
            tmp_data = np.ones([self.data.num_entity, 3])
            for i in range(0, self.data.num_entity):
                tmp_data[i,0] = triple[0]
                tmp_data[i,1] = triple[1]
                tmp_data[i,2] = i
            score = self.calculate_score(tmp_data)
            score, ranked_tails = torch.topk(score,self.data.num_entity, largest=False)

            triple_rank = 1
            for tmp_tail in ranked_tails:
                if tmp_tail.cpu().numpy() == triple[2]:
                    break
                else:
                    triple_rank += 1

            rank.append(triple_rank)

        return rank

    def log_loss(self,y_pos, y_neg, temp):
        M = y_pos.size(0)
        N = y_neg.size(0)
        C = int(N/M)
        y_neg = y_neg.view(C, -1).transpose(0,1)
        p = F.softmax(-1*temp*y_neg)
        loss_pos = torch.sum(torch.log(1+y_pos))
        loss_neg = torch.sum(p*torch.log(1+1/y_neg))
        loss = (loss_pos + loss_neg) / 2 / M

        return loss

    def calculate_score(self, triples):
        '''
        if self.gpu:
            h_emb = self.entity_emb(torch.LongTensor(triples[:, 0])).cuda()
            r_emb = self.relation_emb(torch.LongTensor(triples[:, 1])).cuda()
            t_emb = self.entity_emb(torch.LongTensor(triples[:, 2])).cuda()
        else:
            h_emb = self.entity_emb(torch.LongTensor(triples[:, 0]))
            r_emb = self.relation_emb(torch.LongTensor(triples[:, 1]))
            t_emb = self.entity_emb(torch.LongTensor(triples[:, 2]))

        '''
        h_emb = self.entity_emb(torch.LongTensor(triples[:, 0])).to(device)
        r_emb = self.relation_emb(torch.LongTensor(triples[:, 1])).to(device)
        t_emb = self.entity_emb(torch.LongTensor(triples[:, 2])).to(device)

        h_emb = h_emb.view(-1, self.embedding_dim)
        r_emb = r_emb.view(-1, self.embedding_dim)
        t_emb = t_emb.view(-1, self.embedding_dim)

        score = torch.sum(torch.abs(h_emb + r_emb - t_emb), 1)  # is the dimension  correct?

        return score
