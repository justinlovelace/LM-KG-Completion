import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import os
from collections import defaultdict
import sys
import torch
import transformers

import utils
from CONSTANTS import DATA_DIR, COLUMN_NAMES


class KbDataset(Dataset):
    def __init__(self, args: argparse.ArgumentParser):
        self.args = args
        csv_file = os.path.join(DATA_DIR[args.dataset], f'df_train.csv')
        e1_col, rel_col, e2_col = COLUMN_NAMES[args.dataset]
        print(f'Loading dataset from {csv_file}')
        df_kg = pd.read_csv(csv_file)
        self.pos_samples = df_kg[[e1_col,
                                  rel_col, e2_col]].to_numpy(np.int64)
        if args.strategy == 'one_to_n':
            self.gen_one_to_n_data(df_kg)
        elif args.strategy == 'k_to_n' or args.strategy == 'gen_triplets':
            self.gen_k_to_n_data(df_kg)
        elif args.strategy == 'softmax':
            self.gen_softmax_data(df_kg)
        else:
            raise NotImplementedError

    def gen_one_to_n_data(self, df_kg):
        e1_col, rel_col, e2_col = COLUMN_NAMES[self.args.dataset]
        self.data = df_kg[[
            e1_col, rel_col]].to_numpy(np.int64)
        self.labels = np.zeros(
            (len(self.data), self.args.num_entities), dtype=np.float32)
        self.labels[np.arange(start=0, stop=self.labels.shape[0],
                            step=1), df_kg[e2_col].to_numpy(np.int64)] = 1
        self.tails = df_kg[e2_col].to_numpy(np.int64)

    def gen_softmax_data(self, df_kg):
        e1_col, rel_col, e2_col = COLUMN_NAMES[self.args.dataset]
        self.data = df_kg[[
            e1_col, rel_col]].to_numpy(np.int64)
        self.labels = df_kg[e2_col].to_numpy(np.int64)
        self.tails = df_kg[e2_col].to_numpy(np.int64)

    def gen_k_to_n_data(self, df_kg):
        e1_col, rel_col, e2_col = COLUMN_NAMES[self.args.dataset]
        self.data = df_kg[[
            e1_col, rel_col]].drop_duplicates().to_numpy(np.int64)
        e2_lookup = defaultdict(set)
        for e1, r, e2 in zip(df_kg[e1_col], df_kg[rel_col], df_kg[e2_col]):
            e2_lookup[(e1, r)].add(e2)
        self.labels = np.zeros(
            (len(self.data), self.args.num_entities), dtype=np.float32)
        for idx, query in enumerate(self.data):
            e1, r = query[0], query[1]
            for e2 in e2_lookup[(e1, r)]:
                self.labels[idx, e2] = 1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = {}
        item['rel_id'] = torch.tensor(self.data[idx, 1])
        item['ent_id'] = torch.tensor(self.data[idx, 0])
        labels = self.labels[idx]
        return item, labels

class KbEvalGenerator(Dataset):
    def __init__(self, eval_split: str, args: argparse.ArgumentParser):
        self.args = args
        pos_samples = defaultdict(set)
        splits = ['train', 'valid', 'test']
        e1_col, rel_col, e2_col = COLUMN_NAMES[args.dataset]
        assert eval_split in splits
        for spl in splits:
            csv_file = os.path.join(DATA_DIR[args.dataset], f'df_{spl}.csv')
            df_data = pd.read_csv(csv_file)
            if spl == eval_split:
                self.queries = df_data[[
                    e1_col, rel_col]].to_numpy(np.int64)
                e2_list = df_data[e2_col].tolist()
            for e1, r, e2 in zip(df_data[e1_col], df_data[rel_col], df_data[e2_col]):
                pos_samples[(e1, r)].add(e2)
        self.labels = np.zeros((self.queries.shape[0], self.args.num_entities))
        self.filtered_labels = np.zeros(
            (self.queries.shape[0], self.args.num_entities))
        for i, query in enumerate(self.queries):
            e1, r = query[0], query[1]
            e2 = e2_list[i]
            self.labels[i, e2] = 1
            self.filtered_labels[i, list(pos_samples[(e1, r)] - {e2})] = 1


    def __len__(self):
        return self.queries.shape[0]

    def __getitem__(self, idx):
        item = {}
        item['rel_id'] = torch.tensor(self.queries[idx, 1])
        item['ent_id'] = torch.tensor(self.queries[idx, 0])
        labels = self.labels[idx]
        filtered_labels = self.filtered_labels[idx]
        return item, labels, filtered_labels

class KbTextDataset(Dataset):
    def __init__(self, args):
        
        self.args = args
        train_file = 'df_train.csv'
        csv_file = os.path.join(DATA_DIR[args.dataset], train_file)
        print(f'Loading dataset from {csv_file}')
        df_kg = pd.read_csv(csv_file)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.head_bert_model)
        print(len(self.tokenizer))
        args.vocab_size = len(self.tokenizer)
        if args.strategy == 'one_to_n':
            self.gen_one_to_n_data(df_kg)
        elif args.strategy == 'softmax':
            self.gen_softmax_data(df_kg)
        elif args.strategy == 'k_to_n':
            self.gen_k_to_n_data(df_kg)
        else:
            raise NotImplementedError
        
        self.ent2id, self.ent2name, self.rel2id = utils.get_kg_dicts(self.args)
        self.id2ent = {v: k for k,v in self.ent2id.items()}

    def gen_one_to_n_data(self, df_kg):
        e1_col, rel_col, e2_col = COLUMN_NAMES[self.args.dataset]
        self.data = df_kg[[
            e1_col, rel_col]].to_numpy(np.int64)
        self.labels = np.zeros(
            (len(self.data), self.args.num_entities), dtype=np.float32)
        self.labels[np.arange(start=0, stop=self.labels.shape[0],
                            step=1), df_kg[e2_col].to_numpy(np.int64)] = 1
        self.tails = df_kg[e2_col].to_numpy(np.int64)

    def gen_k_to_n_data(self, df_kg):
        e1_col, rel_col, e2_col = COLUMN_NAMES[self.args.dataset]
        self.data = df_kg[[
            e1_col, rel_col]].drop_duplicates().to_numpy(np.int64)
        e2_lookup = defaultdict(set)
        for e1, r, e2 in zip(df_kg[e1_col], df_kg[rel_col], df_kg[e2_col]):
            e2_lookup[(e1, r)].add(e2)
        self.labels = np.zeros(
            (len(self.data), self.args.num_entities), dtype=np.float32)
        for idx, query in enumerate(self.data):
            e1, r = query[0], query[1]
            for e2 in e2_lookup[(e1, r)]:
                self.labels[idx, e2] = 1

    def gen_softmax_data(self, df_kg):
        e1_col, rel_col, e2_col = COLUMN_NAMES[self.args.dataset]
        self.data = df_kg[[
            e1_col, rel_col]].to_numpy(np.int64)
        self.tails = df_kg[e2_col].to_numpy(np.int64)
        self.labels = df_kg[e2_col].to_numpy(np.int64)

    def __getitem__(self, idx):
        triple = self.data[idx]
        head_strings = self.ent2name[self.id2ent[triple[0]]]
        encodings = self.tokenizer(text=head_strings, is_split_into_words=False, padding='max_length', truncation=True, max_length=16 if 'FB15' not in self.args.dataset else 32)

        encodings['rel_id'] = triple[1]
        encodings['ent_id'] = triple[0]
        item = {key: torch.tensor(val) for key, val in encodings.items()}

        return item, self.labels[idx]  

    def __len__(self):
        return len(self.labels)


class KbTextEvalGenerator(Dataset):
    def __init__(self, eval_split: str, args: argparse.ArgumentParser):
        self.args = args
        self.ent2id, self.ent2name, self.rel2id = utils.get_kg_dicts(self.args)
        self.id2ent = {v: k for k,v in self.ent2id.items()}

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.head_bert_model)
        args.vocab_size = len(self.tokenizer)

        pos_samples = defaultdict(set)
        splits = ['train', 'valid', 'test']
        e1_col, rel_col, e2_col = COLUMN_NAMES[args.dataset]
        assert eval_split in splits
        for spl in splits:
            csv_file = os.path.join(DATA_DIR[args.dataset], f'df_{spl}.csv')
            df_data = pd.read_csv(csv_file)
            if spl == eval_split:
                self.queries = df_data[[
                    e1_col, rel_col]].to_numpy(np.int64)
                e2_list = df_data[e2_col].tolist()
            for e1, r, e2 in zip(df_data[e1_col], df_data[rel_col], df_data[e2_col]):
                pos_samples[(e1, r)].add(e2)
        self.labels = np.zeros((self.queries.shape[0], self.args.num_entities))
        self.filtered_labels = np.zeros(
            (self.queries.shape[0], self.args.num_entities))
        for i, query in enumerate(self.queries):
            e1, r = query[0], query[1]
            e2 = e2_list[i]
            self.labels[i, e2] = 1
            self.filtered_labels[i, list(pos_samples[(e1, r)] - {e2})] = 1

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        query = self.queries[idx]
        
        head_strings = self.ent2name[self.id2ent[query[0]]]
        encodings = self.tokenizer(text=head_strings, is_split_into_words=False, padding='max_length', truncation=True, max_length=16 if 'FB15' not in self.args.dataset else 32)

        encodings['rel_id'] = query[1]
        encodings['ent_id'] = query[0]

        item = {key: torch.tensor(val) for key, val in encodings.items()}
        
        return item, self.labels[idx], self.filtered_labels[idx]
