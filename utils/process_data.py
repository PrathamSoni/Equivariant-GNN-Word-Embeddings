import argparse
import os
import os.path
import random
from glob import glob

import stanza


def process_billion():
    training_files = glob(os.path.join(os.getcwd(), "../data/billion/training-monolingual.tokenized.shuffled/*"))
    heldout = glob(os.path.join(os.getcwd(), "../data/billion/heldout-monolingual.tokenized.shuffled/*"))
    os.makedirs(os.path.join(os.getcwd(), "../data/billion/splits/train"))
    os.makedirs(os.path.join(os.getcwd(), "../data/billion/splits/dev"))
    sample_size = 10

    training_data = []
    print("processing train files")
    for file in training_files:
        with open(file) as f:
            training_data += [line.strip() for line in f.readlines()]

    train_sample = random.sample(training_data, 30000)
    with open(os.path.join(os.getcwd(), "../data/billion/splits/train/data.txt"), 'w') as train_out:
        print(f"{len(train_sample)} samples for train")
        train_out.write('\n'.join(train_sample))

    train_depend = []
    train_pos = []
    for data_batch in train_sample:
        train_parsed_data = nlp(data_batch)
        train_depend_fragments = [" ".join(sent.dependencies_string().split("\n")) for sent in
                                  train_parsed_data.sentences]
        train_pos_fragments = [" ".join([word.upos for word in sent.words]) for sent in train_parsed_data.sentences]
        train_depend.append(" ".join(train_depend_fragments))
        train_pos.append(" ".join(train_pos_fragments))

    with open(os.path.join(os.getcwd(), "../data/billion/splits/train/dependencies.txt"), 'w') as train_depend_out:
        train_depend_out.write("\n".join(train_depend))

    with open(os.path.join(os.getcwd(), "../data/billion/splits/train/pos.txt"), 'w') as train_pos_out:
        train_pos_out.write("\n".join(train_pos))
    del training_data

    dev_data = []
    print("processing dev files")
    for file in heldout:
        with open(file) as f:
            dev_data += [line.strip() for line in f.readlines()]

    dev_sample = random.sample(dev_data, sample_size)

    with open(os.path.join(os.getcwd(), "../data/billion/splits/dev/data.txt"), 'w') as dev_out:
        print(f"{len(dev_sample)} samples for dev")
        dev_out.write('\n'.join(dev_sample))

    dev_depend = []
    dev_pos = []
    for data_batch in dev_sample:
        dev_parsed_data = nlp(data_batch)
        dev_depend_fragments = [" ".join(sent.dependencies_string().split("\n")) for sent in
                                  dev_parsed_data.sentences]
        dev_pos_fragments = [" ".join([word.upos for word in sent.words]) for sent in dev_parsed_data.sentences]
        dev_depend.append(" ".join(dev_depend_fragments))
        dev_pos.append(" ".join(dev_pos_fragments))

    with open(os.path.join(os.getcwd(), "../data/billion/splits/dev/dependencies.txt"), 'w') as dev_depend_out:
        dev_depend_out.write("\n".join(dev_depend))
    with open(os.path.join(os.getcwd(), "../data/billion/splits/dev/pos.txt"), 'w') as dev_pos_out:
        dev_pos_out.write("\n".join(dev_pos))

    del dev_data


if __name__ == "__main__":
    stanza.download('en')  # This downloads the English models for the neural pipeline
    nlp = stanza.Pipeline('en')  # This sets up a default neural pipeline in English

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    args = parser.parse_args()
    print(f"processing {args.dataset} dataset")

    if args.dataset == "billion":
        process_billion()
