import datetime
import os
from argparse import ArgumentParser

import torch
import torch_geometric

import models.models
from data.datasets import get_dataset
from gvp.data import BatchSampler


def pretrain(dataset, dir, encoder, epochs, lr, batch_size):
    count = 0
    pretrain_dataset = get_dataset(dataset, "pretrain", encoder)
    pretrain_loader = torch_geometric.loader.DataLoader(pretrain_dataset,
                                                        batch_sampler=BatchSampler(pretrain_dataset.node_counts,
                                                                                   max_nodes=batch_size))

    dev_dataset = get_dataset(dataset, "dev", encoder, pretrain_dataset.pos_map, pretrain_dataset.dependency_map)
    dev_loader = torch_geometric.loader.DataLoader(dev_dataset,
                                                        batch_sampler=BatchSampler(pretrain_dataset.node_counts,
                                                                                   max_nodes=batch_size))

    model = models.models.GraphEncoder((300, 2), (37, 1), pos_map=pretrain_dataset.pos_map,
                                       dep_map=pretrain_dataset.dependency_map)
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=.0001)
    pos_loss = torch.nn.CrossEntropyLoss(reduction='sum')
    model.train()

    for i in range(epochs):
        losses = []
        for x in pretrain_loader:
            x = x.to(device)
            pos = x.pos
            depend = x.dependency.to(device)

            # forward the model
            optimizer.zero_grad()

            latent, pos_pred, depend_pred = model(x)
            dep_scores = []
            for raw in depend_pred:
                n, _, _ = raw.shape
                dep_scores.append(raw[list(range(n)), depend[slice(x.ptr[i], x.ptr[i+1]), 0], depend[slice(x.ptr[i], x.ptr[i+1]), 1]])
            dep_scores = torch.log(torch.cat(dep_scores))
            loss = pos_loss(pos_pred, pos) - dep_scores.sum()
            losses.append(loss.item())

            # backprop and update the parameters
            loss.backward()
            optimizer.step()

            print(loss)
        print(sum(losses) / len(losses))
        # with torch.no_grad():
        #     for x, depend, pos in dev_dataset:
        #         x = x.to(device)
        #         pos = pos.to(device)
        #         depend = depend.to(device)


def train(dataset, dir, encoder, epochs, lr, batch_size):
    train_dataset = get_dataset(dataset, "train", encoder, batch_size)
    dev_dataset = get_dataset(dataset, "dev", encoder, batch_size)
    for i in range(epochs):
        losses = []
        for x, depend, pos in dataset:
            x = x.to(device)
            y = y.to(device)

            # # forward the model
            # with torch.set_grad_enabled(True):
            #     logits, loss = model(x, y)
            #     loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
            #     losses.append(loss.item())
            #
            # # backprop and update the parameters
            # model.zero_grad()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            # optimizer.step()
            #
            # # decay the learning rate based on our progress
            # if config.lr_decay:
            #     self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
            #     if self.tokens < config.warmup_tokens:
            #         # linear warmup
            #         lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
            #     else:
            #         # cosine learning rate decay
            #         progress = float(self.tokens - config.warmup_tokens) / float(
            #             max(1, config.final_tokens - config.warmup_tokens))
            #         lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            #     lr = config.learning_rate * lr_mult
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr
            # else:
            #     lr = config.learning_rate
            #
            # # report progress
            # pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")


def test(dataset, dir, encoder):
    test_dataset = get_dataset(dataset, "test", encoder, 1)
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--mode")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--dir")
    parser.add_argument("--encoder")
    args = parser.parse_args()

    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    run_dir = args.dir or datetime.datetime.now().strftime("%d%m%Y%H%M%S")
    working_dir = os.path.join(os.getcwd(), 'runs', run_dir)

    if args.mode == "pretrain":
        os.mkdir(working_dir)
        pretrain(args.dataset, args.dir, args.encoder, args.epochs, args.lr, args.batch_size)

    if args.mode == "train":
        train(args.dataset, args.dir, args.encoder, args.epochs, args.lr, args.batch_size)

    elif args.mode == "test":
        test(args.dataset, args.dir, args.encoder)
