import pandas as pd
import numpy as np
import os
import torch



class ReadDataset():
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.num_triple = 0
        self.num_entity = 0
        self.num_relation = 0
        self.nums = [0, 0, 0]  # num_triple, num_entity, num_relation
        self.train_triples = []
        self.test_triples = []
        self.head_rel2_tail = {}
        self.tail_rel2_head = {}
        self.triple2id = {}



        self.reading_triples()
        self.reading_entity_relation()

        self.nums[0] = self.num_triple
        self.nums[1] = self.num_entity
        self.nums[2] = self.num_relation


    def reading_triples(self):
        train_file_path = "train.txt"
        training_df = pd.read_csv(os.path.join(self.dir_path, train_file_path), sep='\t', header=None)
        self.train_triples = list(zip([h for h in training_df[0]],
                                      [r for r in training_df[1]],
                                      [t for t in training_df[2]]))
        self.triple2id['h'] = []
        self.triple2id['r'] = []
        self.triple2id['t'] = []

        for tmp_head, tmp_rel, tmp_tail in self.train_triples:
            self.triple2id['h'].append(tmp_head)
            self.triple2id['r'].append(tmp_rel)
            self.triple2id['t'].append(tmp_tail)


            if tmp_head not in self.head_rel2_tail.keys():
                self.head_rel2_tail[tmp_head] = {}
                self.head_rel2_tail[tmp_head][tmp_rel] = []
                self.head_rel2_tail[tmp_head][tmp_rel].append(tmp_tail)
            else:
                if tmp_rel not in self.head_rel2_tail[tmp_head].keys():
                    self.head_rel2_tail[tmp_head][tmp_rel] = []
                    self.head_rel2_tail[tmp_head][tmp_rel].append(tmp_tail)
                else:
                    if tmp_tail not in self.head_rel2_tail[tmp_head][tmp_rel]:
                        self.head_rel2_tail[tmp_head][tmp_rel].append(tmp_tail)


            if tmp_tail not in self.tail_rel2_head.keys():
                self.tail_rel2_head[tmp_tail] = {}
                self.tail_rel2_head[tmp_tail][tmp_rel] = []
                self.tail_rel2_head[tmp_tail][tmp_rel].append(tmp_head)
            else:
                if tmp_rel not in self.tail_rel2_head[tmp_tail].keys():
                    self.tail_rel2_head[tmp_tail][tmp_rel] = []
                    self.tail_rel2_head[tmp_tail][tmp_rel].append(tmp_head)
                else:
                    if tmp_tail not in self.tail_rel2_head[tmp_tail][tmp_rel]:
                        self.tail_rel2_head[tmp_tail][tmp_rel].append(tmp_head)



        self.num_train_triple = len(self.train_triples)

        test_file_path = "test.txt"
        test_df = pd.read_csv(os.path.join(self.dir_path, test_file_path), sep="\t", header = None)
        self.test_triples = list(zip([h for h in test_df[0]],
                                     [r for r in test_df[1]],
                                     [t for t in test_df[2]]))
        test_file_path = "valid.txt"
        test_df = pd.read_csv(os.path.join(self.dir_path, test_file_path), sep="\t", header=None)
        self.valid_triples = list(zip([h for h in test_df[0]],
                                     [r for r in test_df[1]],
                                     [t for t in test_df[2]]))



    def reading_entity_relation(self):
        entity_file = "entity2id.txt"
        relation_file = "relation2id.txt"

        self.num_entity = self.file_loading(os.path.join(self.dir_path,entity_file))
        self.num_relation= self.file_loading(os.path.join(self.dir_path,relation_file))



    def file_loading(self,file_path):
        content_list = []
        with open(file_path) as f:
            for line in f:
                idx,content = line.strip().split('\t')
                content_list.extend(content)

        return len(content_list)


def gen_corrupt_triples(train_triples, num_entity,head_rel2_tail, tail_rel2_head):
    pos_batch = {}
    neg_batch = {}
    pos_batch['h'] = []
    pos_batch['r'] = []
    pos_batch['t'] = []
    neg_batch['h'] = []
    neg_batch['r'] = []
    neg_batch['t'] = []
    for triples in train_triples:
        head = triples[0].item()
        rel = triples[1].item()
        tail = triples[2].item()

        pos_batch['h'].append(head)
        pos_batch['r'].append(rel)
        pos_batch['t'].append(tail)


        if (np.random.rand(1) >= 0.5):
            tmp_neg_head = int(np.random.uniform(0, num_entity))
            while tmp_neg_head in tail_rel2_head[tail][rel] or tmp_neg_head == head:
                tmp_neg_head = int(np.random.uniform(0, num_entity))
            head = tmp_neg_head
        else:
            tmp_neg_tail = int(np.random.uniform(0, num_entity))
            while tmp_neg_tail in head_rel2_tail[head][rel] or tmp_neg_tail == tail:
                tmp_neg_tail = int(np.random.uniform(0, num_entity))
            tail = tmp_neg_tail

        neg_batch['h'].append(head)
        neg_batch['r'].append(rel)
        neg_batch['t'].append(tail)

    for key_i in pos_batch:
        pos_batch[key_i] = torch.LongTensor(pos_batch[key_i])
    for key_i in neg_batch:
        neg_batch[key_i] = torch.LongTensor(neg_batch[key_i])

    return pos_batch, neg_batch
