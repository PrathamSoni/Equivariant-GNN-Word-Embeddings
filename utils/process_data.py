import argparse
import os
import os.path
import random
from glob import glob
from xml.dom import minidom

import stanza
import spacy


def process_billion():
    stanza.download('en')  # This downloads the English models for the neural pipeline
    nlp = stanza.Pipeline('en')  # This sets up a default neural pipeline in English

    pretraining_files = glob(os.path.join(os.getcwd(), "../data/billion/training-monolingual.tokenized.shuffled/*"))
    heldout = glob(os.path.join(os.getcwd(), "../data/billion/heldout-monolingual.tokenized.shuffled/*"))
    os.makedirs(os.path.join(os.getcwd(), "../data/billion/splits/pretrain"))
    os.makedirs(os.path.join(os.getcwd(), "../data/billion/splits/dev"))
    sample_size = 30000

    pretraining_data = []
    print("processing pretrain data")
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
        pretrain_pos_fragments = [" ".join([word.upos for word in sent.words]) for sent in
                                  pretrain_parsed_data.sentences]
        pretrain_depend.append(" ".join(pretrain_depend_fragments))
        pretrain_pos.append(" ".join(pretrain_pos_fragments))

    with open(os.path.join(os.getcwd(), "../data/billion/splits/pretrain/dependencies.txt"),
              'w') as pretrain_depend_out:
        pretrain_depend_out.write("\n".join(pretrain_depend))

    with open(os.path.join(os.getcwd(), "../data/billion/splits/pretrain/pos.txt"), 'w') as pretrain_pos_out:
        pretrain_pos_out.write("\n".join(pretrain_pos))
    del pretraining_data

    dev_data = []
    print("processing dev data")
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


def process_pubmed():
    raw = glob(os.path.join(os.getcwd(), "../data/pubmed/raw/*.xml"))
    os.makedirs(os.path.join(os.getcwd(), "../data/pubmed/splits/pretrain"))
    os.makedirs(os.path.join(os.getcwd(), "../data/pubmed/splits/dev"))

    raw_data = []
    for i, xml in enumerate(raw):
        print(i + 1, xml)
        doc = minidom.parse(xml)
        title_list = doc.getElementsByTagName('ArticleTitle')
        print(f"titles: {len(title_list)}")
        raw_data.extend([str(title.firstChild.nodeValue).strip().replace("\n", "") for title in title_list if
                         title.firstChild is not None and title.firstChild.nodeValue is not None])

        abstract_list = doc.getElementsByTagName('AbstractText')
        print(f"abstracts: {len(abstract_list)}")
        raw_data.extend([str(abstract.firstChild.nodeValue).strip().replace("\n", "") for abstract in abstract_list if
                         abstract.firstChild is not None and abstract.firstChild.nodeValue is not None])

        print()
        del doc

        if i == 9:
            break

    random.shuffle(raw_data)
    pretrain_data = raw_data[:len(raw_data) // 2]
    dev_data = raw_data[len(raw_data) // 2:]
    sample_size = 30000
    pretrain_sample = random.sample(pretrain_data, sample_size)
    dev_sample = random.sample(dev_data, sample_size)
    del dev_data
    del pretrain_data

    nlp = spacy.load("en_core_sci_lg")

    print("processing pretrain data")
    with open(os.path.join(os.getcwd(), "../data/pubmed/splits/pretrain/data.txt"), 'w') as pretrain_out:
        print(f"{len(pretrain_sample)} samples for pretrain")
        pretrain_out.write('\n'.join(pretrain_sample))

    pretrain_depend = []
    pretrain_pos = []
    for data_batch in pretrain_sample:
        pretrain_parsed_data = nlp(data_batch)
        pretrain_depend_fragments = [" ".join(
            [f"(\'{word.text}\', {word.head.i + 1 if word.dep_ != 'ROOT' else 0}, \'{word.dep_}\')" for word in sent])
                                     for sent in pretrain_parsed_data.sents]
        pretrain_pos_fragments = [" ".join([word.pos_ for word in sent]) for sent in
                                  pretrain_parsed_data.sents]
        pretrain_depend.append(" ".join(pretrain_depend_fragments))
        pretrain_pos.append(" ".join(pretrain_pos_fragments))

    with open(os.path.join(os.getcwd(), "../data/pubmed/splits/pretrain/dependencies.txt"),
              'w') as pretrain_depend_out:
        pretrain_depend_out.write("\n".join(pretrain_depend))

    with open(os.path.join(os.getcwd(), "../data/pubmed/splits/pretrain/pos.txt"), 'w') as pretrain_pos_out:
        pretrain_pos_out.write("\n".join(pretrain_pos))

    print("processing dev data")
    with open(os.path.join(os.getcwd(), "../data/pubmed/splits/dev/data.txt"), 'w') as dev_out:
        print(f"{len(dev_sample)} samples for dev")
        dev_out.write('\n'.join(dev_sample))

    dev_depend = []
    dev_pos = []
    for data_batch in dev_sample:
        dev_parsed_data = nlp(data_batch)
        dev_depend_fragments = [" ".join([f"(\'{word.text}\', {word.head.i}, \'{word.dep_}\')" for word in sent]) for
                                sent in dev_parsed_data.sents]
        dev_pos_fragments = [" ".join([word.pos_ for word in sent]) for sent in dev_parsed_data.sents]
        dev_depend.append(" ".join(dev_depend_fragments))
        dev_pos.append(" ".join(dev_pos_fragments))

    with open(os.path.join(os.getcwd(), "../data/pubmed/splits/dev/dependencies.txt"), 'w') as dev_depend_out:
        dev_depend_out.write("\n".join(dev_depend))
    with open(os.path.join(os.getcwd(), "../data/pubmed/splits/dev/pos.txt"), 'w') as dev_pos_out:
        dev_pos_out.write("\n".join(dev_pos))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    args = parser.parse_args()
    print(f"processing {args.dataset} dataset")

    if args.dataset == "billion":
        process_billion()
    elif args.dataset == "pubmed":
        process_pubmed()
