import math
import os
import time

import numpy as np
import spacy
import torch
import torch.nn.functional as F
import torch_geometric
from gensim.models import KeyedVectors
from torch.utils.data import Dataset

from data.drug.machinereading.models.backoffnet import Preprocessor as DrugPreprocesor, Example, get_entity_lists, \
    get_ds_train_dev_pmids, JAX_TEST_PMIDS_FILE, make_vocab


class GraphEncoder:
    def __init__(self, base_word_encoder, num_rbf=16, num_positional_embeddings=16):
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        if base_word_encoder == "wikipedia":
            self.word_encoder = KeyedVectors.load_word2vec_format("data/wikipediaword2vec/wikipedia200.bin",
                                                                  binary=True)
            self.base_dim = 200
        elif base_word_encoder == "pubmed":
            self.word_encoder = KeyedVectors.load_word2vec_format("data/pubmedword2vec/pubmed2018_w2v_200D.bin",
                                                                  binary=True)
            self.base_dim = 200

    def _normalize(self, tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

    def _orientations(self, X):
        forward = self._normalize(X[1:] - X[:-1])
        backward = self._normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _positional_embeddings(self, edge_index,
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _rbf(self, D, D_min=0., D_max=20., D_count=16, device='cpu'):
        '''
        From https://github.com/jingraham/neurips19-graph-protein-design

        Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
        That is, if `D` has shape [...dims], then the returned tensor will have
        shape [...dims, D_count].
        '''
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)

        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def __call__(self, sentence, dependencies, pos):
        with torch.no_grad():
            n = len(sentence)
            scalar_embeddings = []

            for word in sentence:
                try:
                    scalar_embeddings.append(self.word_encoder[word])

                except KeyError:
                    scalar_embeddings.append(np.full(self.base_dim, 0))

            coords = torch.as_tensor(np.array(scalar_embeddings),
                                     device=self.device, dtype=torch.float32)

            mask = torch.isfinite(coords.sum(dim=1))
            coords[~mask] = np.inf

            deps = dependencies[:, 0].tolist()
            edge_one_hot_map = {}

            for i in range(n):
                dep = int(deps[i] - 1)
                if dep != -1:
                    forward_embedding = edge_one_hot_map.get((i, dep))
                    reverse_embedding = edge_one_hot_map.get((dep, i))
                    if forward_embedding is not None:
                        edge_one_hot_map[(i, dep)][0] += 1
                    else:
                        edge_one_hot_map[(i, dep)] = torch.zeros(5)
                        edge_one_hot_map[(i, dep)][0] += 1

                    if reverse_embedding is not None:
                        edge_one_hot_map[(dep, i)][1] += 1
                    else:
                        edge_one_hot_map[(dep, i)] = torch.zeros(5)
                        edge_one_hot_map[(dep, i)][1] += 1
                if i != n - 1:
                    forward_embedding = edge_one_hot_map.get((i, i + 1))
                    if forward_embedding is not None:
                        edge_one_hot_map[(i, i + 1)][2] += 1
                    else:
                        edge_one_hot_map[(i, i + 1)] = torch.zeros(5)
                        edge_one_hot_map[(i, i + 1)][2] += 1
                if i != 0:
                    reverse_embedding = edge_one_hot_map.get((i, i - 1))
                    if reverse_embedding is not None:
                        edge_one_hot_map[(i, i - 1)][3] += 1
                    else:
                        edge_one_hot_map[(i, i - 1)] = torch.zeros(5)
                        edge_one_hot_map[(i, i - 1)][3] += 1

                edge_one_hot_map[(i, i)] = torch.zeros(5)
                edge_one_hot_map[(i, i)][4] += 1

            edge_index = torch.as_tensor([list(edge) for edge in edge_one_hot_map.keys()],
                                         device=self.device).T.long()
            edge_one_hot = torch.stack(list(edge_one_hot_map.values())).to(self.device)
            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
            rbf = self._rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)

            orientations = self._orientations(coords)

            node_s = coords
            node_v = orientations
            edge_s = torch.cat([rbf, pos_embeddings, edge_one_hot], dim=-1)
            edge_v = self._normalize(E_vectors).unsqueeze(-2)

            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                                                 (node_s, node_v, edge_s, edge_v))

        data = torch_geometric.data.Data(x=coords,
                                         node_s=node_s, node_v=node_v,
                                         edge_s=edge_s, edge_v=edge_v,
                                         edge_index=edge_index, mask=mask,
                                         dependency=dependencies.long(), pos=pos.long())
        return data


class PretrainDataset(Dataset):
    def __init__(self, split, base_word_encoder, data_loc, pos_map=None, dependency_map=None):
        start = time.time()
        self.encoder = GraphEncoder(base_word_encoder)
        self.device = self.encoder.device
        self.padding = "<PAD>"
        self.node_counts = []

        with open(f"data/{data_loc}/splits/{split}/data.txt") as f:
            self.text = [line.strip() for line in f.readlines()]
        with open(f"data/{data_loc}/splits/{split}/dependencies.txt") as f:
            self.depend = []
            self.tokens = []
            for line in f.readlines():
                processed_line = [dependency.replace("(", "").replace("'", "").replace(")", "").split(", ") for
                                  dependency in
                                  line.strip().split(") (")]
                tokens = [token[0] for token in processed_line]
                self.tokens.append(tokens)
                self.node_counts.append(len(tokens))
                self.depend.append([token[1:] for token in processed_line])
            unique_depend = list(sorted(list(set([i[1] for sublist in self.depend for i in sublist]))))
            unique_depend.insert(0, self.padding)
            self.dependency_map = dependency_map or {pos: i for i, pos in enumerate(unique_depend)}
        with open(f"data/{data_loc}/splits/{split}/pos.txt") as f:
            self.pos = [line.strip().split(" ") for line in f.readlines()]
            unique_pos = list(sorted(list(set([i for sublist in self.pos for i in sublist]))))
            unique_pos.insert(0, self.padding)
            self.pos_map = pos_map or {pos: i for i, pos in enumerate(unique_pos)}

        print(f"processed dataset in {time.time() - start}s")

    def encode_dependence(self, depend):
        return torch.as_tensor(
            [[int(dependency[0]) - 1 if int(dependency[0]) != 0 else i, self.dependency_map.get(dependency[1], 0)] for
             i, dependency in enumerate(depend)], device=self.device)

    def encode_pos(self, pos):
        out = []
        for i in pos:
            out.append(self.pos_map.get(i, 0))
        return torch.as_tensor(out, device=self.device)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        depend = self.depend[item]
        processed_depend = self.encode_dependence(depend)
        pos = self.pos[item]
        return self.encoder(self.tokens[item], processed_depend, self.encode_pos(pos))


class ACE2005Dataset(Dataset):
    def __init__(self, split, base_word_encoder):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


class DrugGeneDataset(Dataset):
    def __init__(self, split, base_word_encoder, graph_encoder, pos_map=None, dependency_map=None, vocab=None,
                 preprocessor=None):
        start = time.time()
        self.encoder = GraphEncoder(base_word_encoder)
        self.model_encoder = graph_encoder
        self.device = self.encoder.device
        self.pos_map = pos_map
        self.dependency_map = dependency_map

        entity_lists = get_entity_lists()
        # Read data
        train_pmids_set, dev_ds_pmids_set = get_ds_train_dev_pmids("drug/data/pmid_lists/init_pmid_list.txt")
        ds_train_dev_data = Example.read_examples("drug/data/examples_v2/sentence/ds_train_dev.txt")
        # Filter out examples that doesn't contain pair or triple candidates
        ds_train_dev_data = [x for x in ds_train_dev_data if x.triple_candidates]
        if split == "train":
            self.data = [x for x in ds_train_dev_data if x.pmid in train_pmids_set]

        elif split == "dev":
            self.data = [x for x in ds_train_dev_data if x.pmid in dev_ds_pmids_set]

        else:
            jax_dev_test_data = Example.read_examples("drug/data/examples_v2/sentence/jax_dev_test.txt")
            jax_dev_test_data = [x for x in jax_dev_test_data if x.triple_candidates]

            # with open(os.path.join("data/drug/data", JAX_DEV_PMIDS_FILE)) as f:
            #     dev_jax_pmids_set = set(x.strip() for x in f if x.strip())
            with open(os.path.join("data/drug/data", JAX_TEST_PMIDS_FILE)) as f:
                test_pmids_set = set(x.strip() for x in f if x.strip())

            # dev_jax_data = [x for x in jax_dev_test_data if x.pmid in dev_jax_pmids_set]
            self.data = [x for x in jax_dev_test_data if x.pmid in test_pmids_set]

        self.vocab = vocab or make_vocab(self.data, entity_lists, 0)
        self.preprocessor = preprocessor or DrugPreprocesor(entity_lists, self.vocab, self.device)
        self.nlp = spacy.load("en_core_sci_lg")

        print(f"processed dataset in {time.time() - start}s")

    def encode_dependence(self, depend):
        return torch.as_tensor(
            [[int(dependency[0]) - 1 if int(dependency[0]) != 0 else i, self.dependency_map.get(dependency[1], 0)] for
             i, dependency in enumerate(depend)], device=self.device)

    def encode_pos(self, pos):
        out = []
        for i in pos:
            out.append(self.pos_map.get(i, 0))
        return torch.as_tensor(out, device=self.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        preproc = self.preprocessor.preprocess(self.data[item], None)
        labels = preproc[-2:]
        word_idx_mat, lens, para_vecs, mentions, triple_candidates, pair_candidates = preproc[:-2]
        sent = self.vocab.recover_sentence(torch.squeeze(word_idx_mat).tolist())

        pretrain_parsed_data = self.nlp(sent)
        pretrain_depend_fragments = [" ".join(
            [f"(\'{word.text}\', {word.head.i + 1 if word.dep_ != 'ROOT' else 0}, \'{word.dep_}\')" for word in sent])
            for sent in pretrain_parsed_data.sents]
        pos = [" ".join([word.pos_ for word in sent]) for sent in
               pretrain_parsed_data.sents]
        depend = " ".join(pretrain_depend_fragments)

        processed_line = [dependency.replace("(", "").replace("'", "").replace(")", "").split(", ") for
                          dependency in
                          depend.strip().split(") (")]
        depend = [token[1:] for token in processed_line]
        tokens = [token[0] for token in processed_line]

        processed_depend = self.encode_dependence(depend)
        g_encode = self.encoder(tokens, processed_depend, self.encode_pos(pos))
        x = self.model_encoder.embedding(g_encode)
        return torch.unsqueeze(x, dim=1), mentions, triple_candidates, pair_candidates, *labels


def custom_iter(data, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in index_array:
        yield data[i]


def get_dataset(dataset, split, base_word_encoder, pos_map=None, dependency_map=None, graph_encoder=None, vocab=None,
                preprocessor=None):
    if dataset == "ace":
        data = ACE2005Dataset(split, base_word_encoder)
    elif dataset == "drug":
        data = DrugGeneDataset(split, base_word_encoder, graph_encoder, pos_map=pos_map, dependency_map=dependency_map,
                               vocab=vocab, preprocessor=preprocessor)
    elif dataset == "billion" or dataset == "pubmed":
        data = PretrainDataset(split, base_word_encoder, dataset, pos_map=pos_map, dependency_map=dependency_map)
    else:
        raise ValueError("bad dataset name")

    return data
