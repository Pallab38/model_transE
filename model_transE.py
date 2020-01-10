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
        self.criterion = nn.MarginRankingLoss(margin=gamma, reduction='none')
        self.entity_emb = self.entity_embedding()
        self.relation_emb = self.relation_embedding()

        self.all_pos = {}
        for i in data.test_triples:
            self.all_pos[i[0], i[1], i[2]] = 1
        for i in data.train_triples:
            self.all_pos[i[0], i[1], i[2]] = 1
        for i in data.valid_triples:
            self.all_pos[i[0], i[1], i[2]] = 1

    def entity_embedding(self):
        bound = 6./ math.sqrt(self.embedding_dim)
        entity_emb = nn.Embedding(self.data.num_entity, self.embedding_dim)
        entity_emb.weight.data.uniform_(-bound, bound)
        F.normalize(entity_emb.weight.data, p = 1)

        return  entity_emb

    def relation_embedding(self):
        bound = 6. / math.sqrt(self.embedding_dim)
        relation_emb = nn.Embedding(self.data.num_relation, self.embedding_dim)
        relation_emb.weight.data.uniform_(-bound, bound)
        F.normalize(relation_emb.weight.data, p = 1)

        return relation_emb

    def forward(self, pos_triples, neg_triples):
        pos_h_emb = self.entity_emb(torch.LongTensor(pos_triples[:,0])).to(device)
        pos_r_emb = self.relation_emb(torch.LongTensor(pos_triples[:,1])).to(device)
        pos_t_emb = self.entity_emb(torch.LongTensor(pos_triples[:,2])).to(device)


        pos_h_emb = pos_h_emb.view(-1, self.embedding_dim).to(device)
        pos_r_emb = pos_r_emb.view(-1, self.embedding_dim).to(device)
        pos_t_emb = pos_t_emb.view(-1, self.embedding_dim).to(device)

        neg_h_emb = self.entity_emb(torch.LongTensor(neg_triples[:,0])).to(device)
        neg_r_emb = self.relation_emb(torch.LongTensor(neg_triples[:,1])).to(device)
        neg_t_emb = self.entity_emb(torch.LongTensor(neg_triples[:,2])).to(device)


        neg_h_emb = neg_h_emb.view(-1, self.embedding_dim).to(device)
        neg_r_emb = neg_r_emb.view(-1, self.embedding_dim).to(device)
        neg_t_emb = neg_t_emb.view(-1, self.embedding_dim).to(device)

        neg_score = torch.norm((neg_h_emb + neg_r_emb - neg_t_emb),1, 1).to(device)
        pos_score = torch.norm((pos_h_emb + pos_r_emb - pos_t_emb),1, 1).to(device)

        return torch.cat((pos_score,neg_score),dim=0)

    def loss(self, positive_score, negative_score):
        target = torch.tensor([-1], dtype=torch.long).to(device)
        loss_value =  self.criterion(positive_score, negative_score, target)
        return loss_value

    def rank_entity(self, batch):

        h = batch[:, 0].to(device)
        r = batch[:, 1].to(device)
        t = batch[:, 2].to(device)

        batch_size = h.size()[0]
        all_entities = torch.arange(end=self.data.num_entity).unsqueeze(0).repeat(batch_size,1).to(device)
        heads = h.reshape(-1,1).repeat(1,all_entities.size()[1]).to(device)
        relations = r.reshape(-1,1).repeat(1,all_entities.size()[1]).to(device)
        tails = t.reshape(-1,1).repeat(1,all_entities.size()[1]).to(device)

        triple_head = torch.stack((all_entities,relations,tails),dim=2).reshape(-1,3).to(device)
        triple_tail = torch.stack((heads,relations,all_entities),dim=2).reshape(-1,3).to(device)

        prediction =self.forward(triple_head.cpu(),triple_tail.cpu()).reshape(2*batch_size,self.data.num_entity)
        truth_entity_id = torch.cat((t.reshape(-1, 1), h.reshape(-1, 1)))

        return prediction, truth_entity_id

