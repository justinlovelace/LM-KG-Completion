from re import A
import numpy as np
import typing
from models import PretrainedBertResNet
import os
from CONSTANTS import DATA_DIR
import argparse
import dataset
import json
import torch
from tqdm import tqdm
import sys
import time


def get_dataset(split: str, args: argparse.ArgumentParser):
    assert split in {'train', 'valid', 'test'}

    if split == 'train':
        return dataset.KbTextDataset(args) if args.head_bert_pool == 'prompt' else dataset.KbDataset(args)
    else:
        return dataset.KbTextEvalGenerator(split, args) if args.head_bert_pool == 'prompt' else dataset.KbEvalGenerator(split, args)

def get_model(args):
    if args.model == 'PretrainedBertResNet':
        return PretrainedBertResNet(args)
    else:
        raise RuntimeError

def get_kg_dicts(args):
    entity_file = 'entity_idx.json'
    unique_entities = json.load(
        open(os.path.join(args.dataset_folder, entity_file)))

    entity_file = 'entity_names.json'
    unique_entity_names = json.load(
        open(os.path.join(args.dataset_folder, entity_file)))

    relation_file = 'rel_idx.json'
    unique_relations = json.load(
        open(os.path.join(args.dataset_folder, relation_file)))
    return unique_entities, unique_entity_names, unique_relations
        

def set_kg_stats(args):
    entity_file = 'entity_idx.json'
    unique_entities = json.load(
        open(os.path.join(args.dataset_folder, entity_file)))
    args.num_entities = len(unique_entities)

    relation_file = 'rel_idx.json'
    unique_relations = json.load(
        open(os.path.join(args.dataset_folder, relation_file)))
    args.num_relations = len(unique_relations)
    print(f'{args.num_entities} unique entities and {args.num_relations} unique relations')


def set_model_dir(args):
    if args.run_id == '':
        model_dir = f'{args.dataset}/{args.model}/{time.time()}'
    else:
        model_dir = f'{args.dataset}/{args.model}/{args.run_id}'
    args.output_dir = os.path.abspath(os.path.join(args.save_dir, model_dir))
    assert not os.path.exists(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(f'Will save model to {args.output_dir}')

def load_args(args):
    assert len(args.args_dir) > 0
    config = json.load(open(os.path.join(args.args_dir, 'args.json')))
    for key in config:
        if key == 'num_epochs' or key=='grid_search':
            continue
        args.__dict__[key] = config[key]

def get_optimizer(model, args):
    params, heldout_params, head_params, head_heldout_params, tail_params, tail_heldout_params = [], [], [], [], [], []
    no_decay = ['prelu', 'bn', 'ln', 'bias'] 
    tail_names = ['tail_embedding']
    head_names = ['head_embedding']
    for name, p in model.named_parameters():
        if any(tail_p in name for tail_p in tail_names):
            if p.requires_grad == False or any(nd in name for nd in no_decay):
                tail_heldout_params += [p]
            else:
                tail_params += [p]
        elif any(head_p in name for head_p in head_names):
            if p.requires_grad == False or any(nd in name for nd in no_decay):
                head_heldout_params += [p]
            else:
                head_params += [p]
        elif p.requires_grad == False or any(nd in name for nd in no_decay):
            heldout_params += [p]
        else:
            # print(name)
            params += [p]
    optimizer = torch.optim.AdamW(
        [
            {'params': params, 'weight_decay': args.weight_decay},
            {'params': heldout_params, 'weight_decay': 0},
            {'params': head_params, 'lr': args.head_lr, 'weight_decay': args.head_weight_decay},
            {'params': head_heldout_params, 'lr': args.head_lr, 'weight_decay': 0},
            {'params': tail_params, 'lr': args.tail_lr, 'weight_decay': args.weight_decay},
            {'params': tail_heldout_params, 'lr': args.tail_lr, 'weight_decay': 0},
        ],
        lr=args.lr)
    return optimizer

