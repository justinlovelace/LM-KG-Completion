"""Train Glow on CIFAR-10.

Train script adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
import argparse
import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from dataset import EmbeddingDataset
import util
import loss
import json
from tqdm import tqdm
import wandb
import sys
from CONSTANTS import DATA_DIR

from flow import LinearFlow


def main(args):
    # Set up main device and scale batch size
    device = 'cuda' if torch.cuda.is_available() and args.gpu_ids else 'cpu'
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seeds
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    

    util.set_model_dir(args)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    dataset = EmbeddingDataset(args)
    
    dataloaders = {}
    dataloaders['test'] = data.DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    dataloaders['train'] = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # Model
    print('Building model..')
    net = LinearFlow(args=args, in_channels=args.in_channels)
    net = net.to(device)
    print('Model built')

    start_epoch = 0

    loss_fn = loss.NLLLoss(args, k=args.in_channels).to(device)
    
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_name = f'{args.dataset}_{args.bert_model}_{args.bert_pool}_flow'
    writer = wandb.init(project='lm_kg', config=args, dir=args.output_dir, name=run_name)
    best_loss = float('inf')
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train_loss = train(args, epoch, net, dataloaders['train'], device, optimizer, loss_fn, args.max_grad_norm, writer)
        test(args, epoch, net, dataloaders['test'], device, loss_fn, args.num_samples, write_embs=(train_loss<best_loss))
        best_loss = min(train_loss, best_loss)


@torch.enable_grad()
def train(args, epoch, net, dataloader, device, optimizer, loss_fn, max_grad_norm, writer):
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = util.AverageMeter()
    meters = {k:util.AverageMeter() for k in ['loss', 'prior_ll', 'sldj']}
    with tqdm(dataloader) as progress_bar:
        for idx, x in enumerate(progress_bar):
            x = x.to(device)
            optimizer.zero_grad()
            z, sldj = net(x, reverse=False)
            loss_dict = loss_fn(z, sldj)
            for k, v in loss_dict.items():
                meters[k].update(v.item(), x.size(0))
            loss = loss_dict['loss']
            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            progress_bar.set_postfix(loss=meters['loss'].avg,
                                     prior_ll=meters['prior_ll'].avg,
                                     sldj=meters['sldj'].avg,
                                     lr=optimizer.param_groups[0]['lr'])
            # break
        if writer:
            for k,v in meters.items():
                if v.count == 0:
                    print(f'No recorded values of {k}')
                    continue
                wandb.log({f'train/{k}': v.avg}, step=epoch*len(dataloader)+idx)
        return loss_meter.avg



@torch.no_grad()
def test(args, epoch, net, testloader, device, loss_fn, num_samples, write_embs=False):
    global best_loss
    net.eval()
    loss_meter = util.AverageMeter()
    transformed_embeddings = []
    with tqdm(testloader) as progress_bar:
        for x in progress_bar:
            x = x.to(device)
            z, sldj = net(x, reverse=False)
            if write_embs:
                transformed_embeddings.append(z.to('cpu'))

    if write_embs:
        # Save checkpoint
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'test_loss': loss_meter.avg,
            'epoch': epoch,
        }
        os.makedirs('ckpts', exist_ok=True)
        torch.save(state, os.path.join(args.output_dir, f'best.pth.tar'))

        # Save samples and data
        transformed_embeddings = torch.cat(transformed_embeddings, dim=0)
        file_path = os.path.join(args.output_dir, f'{os.path.basename(args.bert_model)}_{args.bert_pool}_flow.pt')
        print(f'Saving {file_path}...')
        torch.save(transformed_embeddings, file_path)

        file_path = os.path.join(DATA_DIR[args.dataset], 'embeddings', f'{os.path.basename(args.bert_model)}_{args.bert_pool}_flow.pt')
        print(f'Saving {file_path}...')
        torch.save(transformed_embeddings, file_path)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Glow on CIFAR-10')


    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU')
    parser.add_argument('--eval_batch_size', default=256, type=int, help='Batch size per GPU')
    parser.add_argument('--gpu_ids', default=[0], type=eval, help='IDs of GPUs to use')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=-1., help='Max gradient norm for clipping')
    parser.add_argument('--weight_decay', type=float, default=0., help='Weight decay coefficient')
    parser.add_argument('--in_channels', default=768, type=int, help='Number of channels in input')
    parser.add_argument('--num_channels', '-C', default=768, type=int, help='Number of channels in hidden layers')
    parser.add_argument('--num_levels', '-L', default=3, type=int, help='Number of levels in the Glow model')
    parser.add_argument('--num_steps', '-K', default=32, type=int, help='Number of steps of flow in each level')
    parser.add_argument('--num_epochs', default=3, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loader threads')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--warm_up', default=1000, type=int, help='Number of steps for lr warm-up')
    parser.add_argument('--dataset', default='UMLS1', type=str, help='Dataset identifier')
    parser.add_argument('--bert_model', default='ms_pubmed_bert', type=str, help='Bert Model Identifier')
    parser.add_argument('--bert_pool', default='mean', type=str, help='Bert Pooling Method')
    parser.add_argument('--save_dir', default='saved_models', type=str, help='Where to save model')
    
    best_loss = 0
    global_step = 0


    main(parser.parse_args())
