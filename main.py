import datetime
import os
from argparse import ArgumentParser

from data.datasets import get_dataset
import stanza

def pretrain(dataset, epochs, lr, dir):
    pass


def train(dataset, epochs, lr, dir):
    for i in epochs:
        pass


def test(dataset, dir):
    pass


if __name__ == "__main__":
    stanza.download('en')  # This downloads the English models for the neural pipeline

    parser = ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--mode")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dir")
    args = parser.parse_args()

    run_dir = args.dir or datetime.datetime.now().strftime("%d%m%Y%H%M%S")
    working_dir = os.path.join(os.getcwd(), 'runs', run_dir)

    if args.mode == "pretrain":
        os.mkdir(working_dir)
        dataset = get_dataset(args.dataset, args.mode, args.batch_size)
        pretrain(dataset, args.epochs, args.lr, args.dir)

    if args.mode == "train":
        dataset = get_dataset(args.dataset, args.mode, args.batch_size)
        train(dataset, args.epochs, args.lr, args.dir)

    elif args.mode == "test":
        dataset = get_dataset(args.dataset, args.mode, 1)
        test(dataset, args.epochs)
