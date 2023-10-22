import torch
import argparse
import utils
import random
import numpy as np
import json
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluation import ranking_and_hits
from CONSTANTS import DATA_DIR
import wandb
from transformers import get_cosine_schedule_with_warmup


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train(model, trainloader, optimizer, loss_fn, args, epoch, scheduler, writer):
    model.train()
    running_loss = 0.0
    batches = tqdm(trainloader)
    num_batches = len(trainloader)

    for i, data in enumerate(batches):
        inputs, labels = {key: val.to('cuda', non_blocking=False) for key, val in data[0].items(
            )}, data[1].to('cuda', non_blocking=False)

        if 'to_n' in args.strategy:
            labels = (1.0 - args.label_smoothing_epsilon)*labels + (args.label_smoothing_epsilon/args.num_entities)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        if args.clip != 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        writer.log({"train/grad_norm": grad_norm, "train/loss": loss.item(), "train/lr": scheduler._last_lr[0]}, step=epoch*num_batches+i)
        optimizer.step()
        if args.warmup:
            scheduler.step()
        
        if i % 50 == 49:
            batches.set_postfix(loss=running_loss / i, lr=scheduler._last_lr[0])
            # break
        
    return running_loss/(i+1)


def main(args: argparse.ArgumentParser, writer=None):
    print('Setting seeds...')
    set_seeds(args.seed)
    train_dataset = utils.get_dataset('train', args)
    trainloader = DataLoader(
    train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers, drop_last=True)
    

    eval_dataset = utils.get_dataset('valid', args)
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.eval_batch_size, pin_memory=True, shuffle=False, num_workers=args.num_workers)
    model = utils.get_model(args)
    

    # Use GPU
    if torch.cuda.is_available():
        model.to('cuda')
    else:
        raise RuntimeError

    # optimizer
    optimizer = utils.get_optimizer(model, args)
    if args.warmup:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=4000, num_training_steps=len(trainloader)*args.num_epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, mode='max', verbose=True)
        scheduler.step(-1)

    # loss function
    if 'softmax' in args.strategy:
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    best_mrr = 0
    epochs_since_improvement = 0

    for epoch in range(args.num_epochs):
        # print(f'Epoch {epoch}')
        train_loss = train(model, trainloader, optimizer, loss_fn, args, epoch, scheduler, writer)

        metrics = ranking_and_hits(
            model, eval_loader, args)
        mrr = metrics['MRR']
        

        if writer:
            val_metrics_dict = {f'val/{key}': val for key, val in metrics.items()}
            val_metrics_dict.update({"train/avg_loss": train_loss})
            writer.log(val_metrics_dict, step=(epoch+1)*len(trainloader))

        if mrr < best_mrr:
            epochs_since_improvement += 1
            if epochs_since_improvement >= args.patience:
                break
        else:
            epochs_since_improvement = 0
            best_mrr = mrr
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                        os.path.join(args.output_dir, 'state_dict.pt'))
            with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
                f.write(json.dumps(metrics))
        if not args.warmup:
            scheduler.step(mrr)
        
        print("Epoch {:04d} | Best MRR {:.5f} | Current MRR {:.5f} | H@1 {:.5f} | H@10 {:.5f} | Epochs Since Improvement {:04d}".
              format(epoch, best_mrr, mrr, metrics['H@1'], metrics['H@10'], epochs_since_improvement))

    print("Epoch {:04d} | Best MRR {:.5f} | Current MRR {:.5f} | H@1 {:.5f} | H@10 {:.5f} | Epochs Since Improvement {:04d}".
              format(epoch, best_mrr, mrr, metrics['H@1'], metrics['H@10'], epochs_since_improvement))
    if writer:
        writer.finish()            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Options for Knowledge Base Completion')

    # General
    parser.add_argument("--model", type=str, required=False, default='PretrainedBertResNet',
                        help="model to use")
    parser.add_argument("--dataset", type=str, required=False, default='CN82K',
                        help="dataset to use")
    parser.add_argument("--strategy", type=str, required=False, default='one_to_n',
                        help="training strategy")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="number of maximum training epochs")
    parser.add_argument("--patience", type=int, default=20,
                        help="patience for early stopping")
    parser.add_argument("--save_dir", type=str, required=False, default="saved_models",
                        help="output directory to store metrics and model file")
    parser.add_argument("--run_id", type=str, required=False, default="",
                        help="folder name for model run")
    parser.add_argument("--eval_batch_size", type=int, default=128,
                        help="batch size when evaluating")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed value")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of threads")

    # Embedding options
    parser.add_argument("--head_bert_model", type=str, required=False, default='bert-base-uncased',
                        help="identifies the bert model")
    parser.add_argument("--tail_bert_model", type=str, required=False, default='bert-base-uncased',
                        help="identifies the bert model")
    parser.add_argument("--head_bert_pool", type=str, required=False, default='prior',
                        help="bert embedding extraction method")
    parser.add_argument("--head_lr", type=float, default=1e-5,
                        help="learning rate")
    parser.add_argument("--tail_bert_pool", type=str, required=False, default='prior',
                        help="bert embedding extraction method")
    parser.add_argument("--tail_embed", type=str, default='default',
                    help="Method for tail embedding. One of ['default', 'mlp', 'res_mlp', 'flow']")
    parser.add_argument("--tail_lr", type=float, default=1e-5,
                        help="learning rate")
    parser.add_argument("--unfreeze_bert", action='store_true', help="Whether to limit prefixes to embedding layer")
    parser.add_argument("--head_weight_decay", type=float, default=0.01,
                        help="weight decay coefficient")

    # Prefix options
    parser.add_argument('--num_prefixes', default=3, type=int, help='Number of prefixes for prefix tuning')
    parser.add_argument('--prefix_dim', default=0, type=int, help='Number of prefixes for prefix tuning')
    parser.add_argument("--prefix_embed", type=str, required=False, default='mlp', help="intermediate layer extraction method")
    parser.add_argument('--span_extraction', type=str, required=False, default='mean', help="intermediate layer extraction method")
    parser.add_argument('--layer_aggr', type=str, required=False, default='lin_comb', help="layer aggregation method")
    parser.add_argument('--use_prefix_projection', action='store_true', help="whether to use intermediate linear projection")

    # Model Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument('--warmup', action='store_true', help="whether to use an lr warmup")
    parser.add_argument("--dropout", type=float, default=0,
                        help="dropout probability")
    parser.add_argument("--input_dropout", type=float, default=0,
                        help="input dropout")
    parser.add_argument("--feature_map_dropout", type=float, default=0,
                        help="feature map dropout")
    parser.add_argument("--label_smoothing_epsilon", type=float, default=0.1,
                        help="epsilon for label smoothing")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay coefficient")
    parser.add_argument("--clip", type=int, default=0,
                        help="value used for gradient clipping")
    parser.add_argument("--embedding_dim", type=int, default=200,
                        help="embedding dimension for entities and relations")
    parser.add_argument("--channels", type=int, default=200,
                    help="output dimension of convolution")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--reshape_len", type=int, default=4,
                        help="Side length for deep convolutional models")
    parser.add_argument("--resnet_num_blocks", type=int, default=2,
                        help="Number of resnet blocks")
    parser.add_argument("--resnet_block_depth", type=int, default=3,
                        help="Depth of each resnet block")
    parser.add_argument("--args_dir", type=str, default='',
                        help="Whether to set hyperparameters from experiment")

    args = parser.parse_args()

    if args.head_bert_model[-2:] == 'ft':
        args.head_bert_model = os.path.join(f'lm_finetuning/ckpts/{args.dataset}', os.path.basename(args.head_bert_model))

    args.dataset_folder = DATA_DIR[args.dataset]
    utils.set_kg_stats(args)

    if args.args_dir != '':
        utils.load_args(args)

    try:   
        utils.set_model_dir(args)
        run_name = f'{args.dataset}_{args.model}_{args.run_id}'
        writer = wandb.init(project='lm_kg', config=args, dir=args.output_dir, name=run_name)
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        main(args, writer)
    except KeyboardInterrupt:
        os.rename(args.output_dir, args.output_dir+'terminated')
        print('Interrupted')    
