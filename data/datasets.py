from torch.utils.data import Dataset
import torch
import torch_geometric
from gensim.models import KeyedVectors
import torch.nn.functional as F
import numpy as np
import time


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


class BillionDataset(Dataset):
    def __init__(self, split, base_word_encoder, pos_map=None, dependency_map=None):
        start = time.time()
        self.encoder = GraphEncoder(base_word_encoder)
        self.device = self.encoder.device
        self.padding = "<PAD>"
        self.block_size = 100
        self.node_counts = []

        with open(f"data/billion/splits/{split}/data.txt") as f:
            self.text = [line.strip() for line in f.readlines()]
        with open(f"data/billion/splits/{split}/dependencies.txt") as f:
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
        with open(f"data/billion/splits/{split}/pos.txt") as f:
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
    def __init__(self, split, base_word_encoder):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


def get_dataset(dataset, split, base_word_encoder, pos_map=None, dependency_map=None):
    if dataset == "ace":
        data = ACE2005Dataset(split, base_word_encoder)
    elif dataset == "drug":
        data = DrugGeneDataset(split, base_word_encoder)
    elif dataset == "billion":
        data = BillionDataset(split, base_word_encoder, pos_map=pos_map, dependency_map=dependency_map)

    return data
