import datetime
import os
from argparse import ArgumentParser
import time

import torch
import torch_geometric
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, recall_score, precision_score, average_precision_score

import models
from data.datasets import get_dataset, custom_iter
from gvp.data import BatchSampler


def pretrain(dataset, dir, encoder, epochs, lr, batch_size):
    writer = SummaryWriter(os.path.join(dir, 'pretrain'))

    pretrain_dataset = get_dataset(dataset, "pretrain", encoder)
    pretrain_loader = torch_geometric.loader.DataLoader(pretrain_dataset,
                                                        batch_sampler=BatchSampler(pretrain_dataset.node_counts,
                                                                                   max_nodes=batch_size))

    dev_dataset = get_dataset(dataset, "dev", encoder, pos_map=pretrain_dataset.pos_map,
                              dependency_map=pretrain_dataset.dependency_map)
    dev_loader = torch_geometric.loader.DataLoader(dev_dataset,
                                                   batch_sampler=BatchSampler(pretrain_dataset.node_counts,
                                                                              max_nodes=batch_size))

    model = models.GraphModel((pretrain_dataset.encoder.base_dim, 2), (37, 1),
                              vector_dim=pretrain_dataset.encoder.base_dim,
                              pos_map=pretrain_dataset.pos_map, dep_map=pretrain_dataset.dependency_map).to(
        device)

    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=.0001)
    pos_loss = torch.nn.CrossEntropyLoss(reduction='sum')
    best_loss = float('inf')
    counter = 0
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

        writer.add_scalar("total_loss/pretrain", pretrain_total_loss, i)
        writer.add_scalar("total_pos_loss/pretrain", pretrain_total_pos_loss, i)
        writer.add_scalar("total_dep_loss/pretrain", pretrain_total_dep_loss, i)

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
                for j, raw in enumerate(depend_pred):
                    n, _, _ = raw.shape
                    dep_scores.append(raw[list(range(n)), depend[slice(x.ptr[j], x.ptr[j + 1]), 0], depend[
                        slice(x.ptr[j], x.ptr[j + 1]), 1]])

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

        writer.add_scalar("total_loss/pretrain_dev", dev_total_loss, i)
        writer.add_scalar("total_pos_loss/pretrain_dev", dev_total_pos_loss, i)
        writer.add_scalar("total_dep_loss/pretrain_dev", dev_total_dep_loss, i)

        print(f"Epoch: {i}\tTime Elapsed: {time.time() - start}\n"
              f"\tPretrain\tTotal: {pretrain_total_loss}\tPOS: {pretrain_total_pos_loss}\tDep: {pretrain_total_dep_loss}\n"
              f"\tDev\tTotal: {dev_total_loss}\tPOS: {dev_total_pos_loss}\tDep: {dev_total_dep_loss}\n")

        if dev_total_loss < best_loss:
            best_loss = dev_total_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(dir, "encoder.pt"))
        else:
            counter += 1
            if counter == 10:
                print("Early stop...")
                break

    writer.close()


def train(model, dataset, pretrain_dataset, dir, encoder, epochs, lr, batch_size, lr_decay=1.0):
    writer = SummaryWriter(os.path.join(dir, 'train'))

    pretrain_dataset = get_dataset(pretrain_dataset, "pretrain", encoder)

    enc_model = models.GraphModel((pretrain_dataset.encoder.base_dim, 2), (37, 1),
                                  vector_dim=pretrain_dataset.encoder.base_dim,
                                  pos_map=pretrain_dataset.pos_map,
                                  dep_map=pretrain_dataset.dependency_map).to(device)
    enc_model.load_state_dict(torch.load(os.path.join(dir, 'encoder.pt'), map_location=device))
    enc_model.eval()

    train_dataset = get_dataset(dataset, "train", encoder, graph_encoder=enc_model, pos_map=pretrain_dataset.pos_map,
                                dependency_map=pretrain_dataset.dependency_map)

    dev_dataset = get_dataset(dataset, "dev", encoder, graph_encoder=enc_model, pos_map=pretrain_dataset.pos_map,
                              dependency_map=pretrain_dataset.dependency_map, vocab=train_dataset.vocab,
                              preprocessor=train_dataset.preprocessor)

    if model == "bilstm":
        model = models.BiLSTM(pretrain_dataset.encoder.base_dim, 150, 1, device, dropout_prob=0).to(device=device)
        loss_func = torch.nn.BCEWithLogitsLoss()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    else:
        raise ValueError("bad model name")

    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
    best_loss = float('inf')
    counter = 0
    for i in range(epochs):
        start = time.time()
        train_total_loss = 0
        nodes = 0

        model.train()
        batch_counter = 0
        for x in custom_iter(train_dataset, shuffle=True):
            batch_counter += 1
            # forward the model
            triple_labels, pair_labels = x[-2:]
            triple_logits, pair_logits = model(*(x[:-2]))

            loss = loss_func(triple_logits, triple_labels)
            for t1, t2 in models.ALL_ENTITY_TYPE_PAIRS:
                loss += loss_func(pair_logits[(t1, t2)], pair_labels[(t1, t2)])

            loss /= batch_size
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            train_total_loss += loss.item()

            nodes += 1

            # backprop and update the parameters
            if batch_counter == batch_size:
                optimizer.step()
                model.zero_grad()
                batch_counter = 0

        train_total_loss /= nodes / batch_size

        writer.add_scalar("total_loss/train", train_total_loss, i)

        dev_total_loss = 0
        nodes = 0
        model.eval()
        with torch.no_grad():
            for x in custom_iter(dev_dataset, shuffle=True):
                model.zero_grad()
                triple_labels, pair_labels = x[-2:]
                triple_logits, pair_logits = model(*x[:-2])

                loss = loss_func(triple_logits, triple_labels)
                for t1, t2 in models.ALL_ENTITY_TYPE_PAIRS:
                    loss += loss_func(pair_logits[(t1, t2)], pair_labels[(t1, t2)])

                dev_total_loss += loss.item()
                nodes += 1

        dev_total_loss /= nodes

        writer.add_scalar("total_loss/train_dev", dev_total_loss, i)

        print(f"Epoch: {i}\tTime Elapsed: {time.time() - start}\n"
              f"\ttrain\tTotal: {train_total_loss}\n"
              f"\tDev\tTotal: {dev_total_loss}\n")

        if dev_total_loss < best_loss:
            best_loss = dev_total_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(dir, "model.pt"))
        else:
            counter += 1
            if counter == 10:
                print("Early stop...")
                break

        scheduler.step()

    writer.close()


def test(model, dataset, pretrain_dataset, dir, encoder):
    pretrain_dataset = get_dataset(pretrain_dataset, "pretrain", encoder)

    enc_model = models.GraphModel((pretrain_dataset.encoder.base_dim, 2), (37, 1),
                                  vector_dim=pretrain_dataset.encoder.base_dim,
                                  pos_map=pretrain_dataset.pos_map,
                                  dep_map=pretrain_dataset.dependency_map).to(device)
    enc_model.load_state_dict(torch.load(os.path.join(dir, 'encoder.pt'), map_location=device))
    enc_model.eval()

    train_dataset = get_dataset(dataset, "train", encoder, graph_encoder=enc_model, pos_map=pretrain_dataset.pos_map,
                                dependency_map=pretrain_dataset.dependency_map)
    test_dataset = get_dataset(dataset, "test", encoder, graph_encoder=enc_model, pos_map=pretrain_dataset.pos_map,
                               dependency_map=pretrain_dataset.dependency_map, vocab=train_dataset.vocab,
                               preprocessor=train_dataset.preprocessor)

    if model == "bilstm":
        model = models.BiLSTM(pretrain_dataset.encoder.base_dim, 150, 1, device, dropout_prob=0).to(device=device)
        loss_func = torch.nn.BCEWithLogitsLoss()
        model.load_state_dict(torch.load(os.path.join(dir, 'model.pt'), map_location=device))
        model.eval()
    else:
        raise ValueError("bad model name")

    nodes = 0
    total_loss = 0
    start = time.time()
    y_true = []
    y_proba = []
    for x in custom_iter(test_dataset, shuffle=False):
        model.zero_grad()
        triple_labels, pair_labels = x[-2:]
        triple_logits, pair_logits = model(*x[:-2])
        # print(triple_labels)
        y_true.extend(triple_labels.tolist())
        y_proba.extend(torch.sigmoid(triple_logits).tolist())

        loss = loss_func(triple_logits, triple_labels)
        for t1, t2 in models.ALL_ENTITY_TYPE_PAIRS:
            loss += loss_func(pair_logits[(t1, t2)], pair_labels[(t1, t2)])

        total_loss += loss.item()
        nodes += 1

    total_loss /= nodes

    print(f"Testing...\tTime Elapsed: {time.time() - start}\n"
          f"\ttest\tTotal: {total_loss}\n")

    y_pred = [1 if y >= .5 else 0 for y in y_proba]

    # print(y_true, y_proba, y_pred)
    a = accuracy_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    p = precision_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_proba)
    print(f"Testing...\tTime Elapsed: {time.time() - start}\n"
          f"\ttest\tTotal loss: {total_loss}\tAccuracy: {a}\tRecall: {r}\tPrecision: {p}\tAverage Precision: {ap}\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--pretrain_dataset")
    parser.add_argument("--mode")
    parser.add_argument("--model")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--dir")
    parser.add_argument("--encoder")
    args = parser.parse_args()

    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')

    run_dir = args.dir or datetime.datetime.now().strftime("%d%m%Y%H%M%S")
    working_dir = os.path.join(os.getcwd(), 'runs', run_dir)
    print(f"writing to {working_dir}")

    if args.mode == "pretrain":
        os.mkdir(working_dir)
        pretrain(args.dataset, working_dir, args.encoder, args.epochs, args.lr, args.batch_size)

    elif args.mode == "train":
        train(args.model, args.dataset, args.pretrain_dataset, working_dir, args.encoder, args.epochs, args.lr,
              args.batch_size)

    elif args.mode == "test":
        test(args.model, args.dataset, args.pretrain_dataset, working_dir, args.encoder)
