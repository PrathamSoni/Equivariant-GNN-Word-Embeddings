import argparse
import os
import os.path
import random
from glob import glob

import stanza


def process_billion():
    pretraining_files = glob(os.path.join(os.getcwd(), "../data/billion/training-monolingual.tokenized.shuffled/*"))
    heldout = glob(os.path.join(os.getcwd(), "../data/billion/heldout-monolingual.tokenized.shuffled/*"))
    os.makedirs(os.path.join(os.getcwd(), "../data/billion/splits/pretrain"))
    os.makedirs(os.path.join(os.getcwd(), "../data/billion/splits/dev"))
    sample_size = 100

    pretraining_data = []
    print("processing pretrain files")
    for file in pretraining_files:
        with open(file) as f:
            pretraining_data += [line.strip() for line in f.readlines()]

    pretrain_sample = random.sample(pretraining_data, sample_size)
    with open(os.path.join(os.getcwd(), "../data/billion/splits/pretrain/data.txt"), 'w') as pretrain_out:
        print(f"{len(pretrain_sample)} samples for pretrain")
        pretrain_out.write('\n'.join(pretrain_sample))

    pretrain_depend = []
    pretrain_pos = []
    for data_batch in pretrain_sample:
        pretrain_parsed_data = nlp(data_batch)
        pretrain_depend_fragments = [" ".join(sent.dependencies_string().split("\n")) for sent in
                                  pretrain_parsed_data.sentences]
        pretrain_pos_fragments = [" ".join([word.upos for word in sent.words]) for sent in pretrain_parsed_data.sentences]
        pretrain_depend.append(" ".join(pretrain_depend_fragments))
        pretrain_pos.append(" ".join(pretrain_pos_fragments))

    with open(os.path.join(os.getcwd(), "../data/billion/splits/pretrain/dependencies.txt"), 'w') as pretrain_depend_out:
        pretrain_depend_out.write("\n".join(pretrain_depend))

    with open(os.path.join(os.getcwd(), "../data/billion/splits/pretrain/pos.txt"), 'w') as pretrain_pos_out:
        pretrain_pos_out.write("\n".join(pretrain_pos))
    del pretraining_data

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
