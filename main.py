import datetime
import os
from argparse import ArgumentParser

from data.datasets import get_dataset
import torch


def pretrain(dataset, epochs, lr, dir):
    for i in range(epochs):
        losses = []
        for x in dataset:
            print(x)


def train(dataset, epochs, lr, dir):
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


def test(dataset, dir):
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--mode")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dir")
    parser.add_argument("--encoder")
    args = parser.parse_args()

    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    run_dir = args.dir or datetime.datetime.now().strftime("%d%m%Y%H%M%S")
    working_dir = os.path.join(os.getcwd(), 'runs', run_dir)

    if args.mode == "pretrain":
        os.mkdir(working_dir)
        dataset = get_dataset(args.dataset, args.mode, args.encoder, args.batch_size)
        pretrain(dataset, args.epochs, args.lr, args.dir)

    if args.mode == "train":
        dataset = get_dataset(args.dataset, args.mode, args.encoder, args.batch_size)
        train(dataset, args.epochs, args.lr, args.dir)

    elif args.mode == "test":
        dataset = get_dataset(args.dataset, args.mode, args.encoder, 1)
        test(dataset, args.epochs)
