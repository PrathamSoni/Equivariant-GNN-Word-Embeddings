import datetime
import os
from argparse import ArgumentParser
import time

import torch
import torch_geometric
from torch.utils.tensorboard import SummaryWriter

import models.models
from data.datasets import get_dataset
from gvp.data import BatchSampler


def pretrain(dataset, dir, encoder, epochs, lr, batch_size):
    writer = SummaryWriter(os.path.join(dir, 'pretrain'))

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
    writer.add_graph(model)
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=.0001)
    pos_loss = torch.nn.CrossEntropyLoss(reduction='sum')

    for i in range(epochs):
        start = time.time()
        pretrain_total_loss = 0
        pretrain_total_pos_loss = 0
        pretrain_total_dep_loss = 0
        nodes = 0

        model.train()
        for x in pretrain_loader:
            x = x.to(device)
            pos = x.pos
            depend = x.dependency.to(device)

            # forward the model
            optimizer.zero_grad()

            latent, pos_pred, depend_pred = model(x)
            dep_scores = []
            for j, raw in enumerate(depend_pred):
                n, _, _ = raw.shape
                dep_scores.append(raw[list(range(n)), depend[slice(x.ptr[j], x.ptr[j + 1]), 0], depend[
                    slice(x.ptr[j], x.ptr[j + 1]), 1]])

            dep_scores = torch.log(torch.cat(dep_scores))
            pos_loss_batch = pos_loss(pos_pred, pos)
            dep_loss_batch = -1 * dep_scores.sum()
            loss = pos_loss_batch + dep_loss_batch

            pretrain_total_loss += loss.item()
            pretrain_total_pos_loss += pos_loss_batch.item()
            pretrain_total_dep_loss += dep_loss_batch.item()

            nodes += x.x.shape[0]
            # backprop and update the parameters
            loss.backward()
            optimizer.step()

        pretrain_total_loss /= nodes
        pretrain_total_pos_loss /= nodes
        pretrain_total_dep_loss /= nodes

        writer.add_scalar("total_loss/pretrain", pretrain_total_loss, epochs)
        writer.add_scalar("total_pos_loss/pretrain", pretrain_total_pos_loss, epochs)
        writer.add_scalar("total_dep_loss/pretrain", pretrain_total_dep_loss, epochs)

        dev_total_loss = 0
        dev_total_pos_loss = 0
        dev_total_dep_loss = 0
        nodes = 0
        model.eval()
        with torch.no_grad():
            for x in dev_loader:
                x = x.to(device)
                pos = x.pos
                depend = x.dependency.to(device)

                # forward the model
                optimizer.zero_grad()

                latent, pos_pred, depend_pred = model(x)
                dep_scores = []
                for raw in depend_pred:
                    n, _, _ = raw.shape
                    dep_scores.append(raw[list(range(n)), depend[slice(x.ptr[i], x.ptr[i + 1]), 0], depend[
                        slice(x.ptr[i], x.ptr[i + 1]), 1]])

                dep_scores = torch.log(torch.cat(dep_scores))
                pos_loss_batch = pos_loss(pos_pred, pos)
                dep_loss_batch = -1 * dep_scores.sum()
                loss = pos_loss_batch + dep_loss_batch

                dev_total_loss += loss.item()
                dev_total_pos_loss += pos_loss_batch.item()
                dev_total_dep_loss += dep_loss_batch.item()

                nodes += x.x.shape[0]

        dev_total_loss /= nodes
        dev_total_pos_loss /= nodes
        dev_total_dep_loss /= nodes

        writer.add_scalar("total_loss/pretrain_dev", dev_total_loss, epochs)
        writer.add_scalar("total_pos_loss/pretrain_dev", dev_total_pos_loss, epochs)
        writer.add_scalar("total_dep_loss/pretrain_dev", dev_total_dep_loss, epochs)

        print(f"Epoch: {i}\tTime Elapsed: {time.time() - start}\n"
              f"\tPretrain\tTotal: {pretrain_total_loss}\tPOS: {pretrain_total_pos_loss}\tDep: {pretrain_total_dep_loss}\n"
              f"\tPretrain\tTotal: {dev_total_loss}\tPOS: {dev_total_pos_loss}\tDep: {dev_total_dep_loss}\n")

    with open(os.path.join(dir, "encoder.pt"), "w") as model_file:
        torch.save(model.state_dict(), model_file)

    writer.close()


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
    print(f"writing to {working_dir}")

    if args.mode == "pretrain":
        os.mkdir(working_dir)
        pretrain(args.dataset, working_dir, args.encoder, args.epochs, args.lr, args.batch_size)

    if args.mode == "train":
        train(args.dataset, working_dir, args.encoder, args.epochs, args.lr, args.batch_size)

    elif args.mode == "test":
        test(args.dataset, working_dir, args.encoder)
