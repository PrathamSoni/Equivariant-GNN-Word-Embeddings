from torch.nn import *
import torch.nn.utils.rnn as rnn

import gvp
import torch
import collections

from data.drug.machinereading.models.backoffnet import Candidate, ALL_ENTITY_TYPE_PAIRS


def get_sentences(x, indices):
    sents = []
    for i in range(x.num_graphs):
        sents.append(gvp.tuple_index(x, slice(indices[i], indices[i + 1])))

    return sents


class GraphModel(Module):
    def __init__(self, node_dims, edge_dims, vector_dim=300, pos_map=None, dep_map=None):
        super().__init__()
        self.W1 = gvp.GVPConvLayer(node_dims, edge_dims, vector_dim=vector_dim, vector_gate=True)
        self.W2 = gvp.GVPConvLayer(node_dims, edge_dims, vector_dim=vector_dim, vector_gate=True)

        self.POS_embedding = gvp.GVP(node_dims, (16, 0), activations=(torch.tanh, None), vector_gate=True)

        self.arc_head_embedding = gvp.GVP(node_dims, (100, 0), activations=(torch.relu, None), vector_gate=True)
        self.arc_dep_embedding = gvp.GVP(node_dims, (100, 0), activations=(torch.relu, None), vector_gate=True)
        self.label_head_embedding = gvp.GVP(node_dims, (100, 0), activations=(torch.relu, None), vector_gate=True)
        self.label_dep_embedding = gvp.GVP(node_dims, (100, 0), activations=(torch.relu, None), vector_gate=True)

        self.POS_classifier = Linear(16, len(pos_map))

        self.arc_layer = Linear(100, 100, bias=False)
        self.arc_label_layers = ModuleList([Linear(100, 100, bias=False) for i in range(len(dep_map))])

    def embedding(self, x, mask=None):
        h_V = (x.node_s, x.node_v)
        h_E = (x.edge_s, x.edge_v)
        pass1 = self.W1(h_V, x.edge_index, h_E, node_mask=mask)
        pass2 = self.W2(pass1, x.edge_index, h_E, node_mask=mask)
        return h_V[0] + pass1[0] + pass2[0]

    def forward(self, x, mask=None):
        h_V = (x.node_s, x.node_v)
        h_E = (x.edge_s, x.edge_v)
        pass1 = self.W1(h_V, x.edge_index, h_E, node_mask=mask)
        pass2 = self.W2(pass1, x.edge_index, h_E, node_mask=mask)

        pos_pred = self.POS_embedding(pass2)
        pos_pred = self.POS_classifier(pos_pred)

        sents = get_sentences(pass2, x.ptr)

        scores = []
        for sent in sents:
            H_arc_head = self.arc_head_embedding(sent)
            arc_dep_score = self.arc_layer(self.arc_dep_embedding(sent))
            s_arc = torch.softmax(H_arc_head @ arc_dep_score.T, dim=-1)

            H_label_head = self.label_head_embedding(sent)
            label_dep_score = torch.stack([layer(self.label_dep_embedding(sent)) for layer in self.arc_label_layers])
            s_label = torch.softmax(H_label_head @ label_dep_score.permute(1, 2, 0), dim=-1)

            scores.append(s_label * s_arc.unsqueeze(2))

        return pass2, pos_pred, scores


def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Copied from https://github.com/pytorch/pytorch/issues/2591

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


class BiLSTM(Module):
    # TODO: add source paper
    """Combine triple and pairwise information."""

    def __init__(self, embed_dim, lstm_size, lstm_layers, device, pool_method='max', dropout_prob=0.5):
        super().__init__()
        self.device = device
        self.pool_method = pool_method
        self.dropout = Dropout(p=dropout_prob)
        self.lstm_layers = lstm_layers
        self.lstm = LSTM(embed_dim, lstm_size, bidirectional=True, num_layers=lstm_layers)
        for t1, t2 in ALL_ENTITY_TYPE_PAIRS:
            setattr(self, 'hidden_%s_%s' %
                    (t1, t2), Linear(4 * lstm_size, 2 * lstm_size))
            setattr(self, 'out_%s_%s' % (t1, t2), Linear(2 * lstm_size, 1))
            setattr(self, 'backoff_%s_%s' % (t1, t2), Parameter(
                torch.zeros(1, 2 * lstm_size)))
        self.hidden_triple = Linear(3 * 2 * lstm_size, 2 * lstm_size)
        self.backoff_triple = Parameter(torch.zeros(1, 2 * lstm_size))
        self.hidden_all = Linear(4 * 2 * lstm_size, 2 * lstm_size)
        self.out_triple = Linear(2 * lstm_size, 1)

    def pool(self, grouped_vecs):
        if self.pool_method == 'mean':
            return torch.stack([torch.mean(g, dim=0) for g in grouped_vecs])
        elif self.pool_method == 'sum':
            return torch.stack([torch.sum(g, dim=0) for g in grouped_vecs])
        elif self.pool_method == 'max':
            return torch.stack([torch.max(g, dim=0)[0] for g in grouped_vecs])
        elif self.pool_method == 'softmax':
            return torch.stack([logsumexp(g, dim=0) for g in grouped_vecs])
        raise NotImplementedError

    def forward(self, x, mentions,
                triple_candidates, pair_candidates):
        """Forward pass.

        Args:
          word_idxs: list of word indices, size (B, T, P)
          lens: list of paragraph lengths, size (P)
          para_vecs: list of paragraph vectors, size (P, pe)
          mentions: list of list of ParaMention
          triple_candidates: list of unlabeled Candidate
          pair_candidates: list of unlabeled Candidate
        """
        T, P, _ = x.shape  # T=num_toks, P=num_paras

        # Organize the candidate pairs and triples
        pair_to_idx = {}
        pair_sets = collections.defaultdict(set)
        for (t1, t2), cands in pair_candidates.items():
            pair_to_idx[(t1, t2)] = {c: i for i, c in enumerate(cands)}
            for c in cands:
                pair_sets[(t1, t2)].add(c)
        triple_to_idx = {c: i for i, c in enumerate(triple_candidates)}

        # Build local embeddings of each word
        # embs = x  # T, P, e
        # lstm_in = rnn.pack_padded_sequence(embs, lens)  # T, P, e + pe
        embs, _ = self.lstm(x)
        # embs, _ = rnn.pad_packed_sequence(lstm_out_packed)  # T, P, 2*h

        # Gather co-occurring mention pairs and triples
        pair_inputs = {(t1, t2): [[] for i in range(len(cands))]
                       for (t1, t2), cands in pair_candidates.items()}
        triple_inputs = [[] for i in range(len(triple_candidates))]

        for para_idx, m_list in enumerate(mentions):
            typed_mentions = collections.defaultdict(list)
            for m in m_list:
                typed_mentions[m.type].append(m)
            for t1, t2 in ALL_ENTITY_TYPE_PAIRS:
                for m1 in typed_mentions[t1]:
                    for m2 in typed_mentions[t2]:
                        query_cand = Candidate(**{t1: m1.name, t2: m2.name})
                        if query_cand in pair_to_idx[(t1, t2)]:
                            idx = pair_to_idx[(t1, t2)][query_cand]
                            cur_vecs = torch.cat([embs[m1.start, para_idx, :],
                                                  embs[m2.start, para_idx, :]])  # 4*h
                            pair_inputs[(t1, t2)][idx].append(cur_vecs)

            for m1 in typed_mentions['drug']:
                for m2 in typed_mentions['gene']:
                    for m3 in typed_mentions['variant']:
                        query_cand = Candidate(m1.name, m2.name, m3.name)
                        if query_cand in triple_to_idx:
                            idx = triple_to_idx[query_cand]
                            cur_vecs = torch.cat([embs[m1.start, para_idx, :],
                                                  embs[m2.start, para_idx, :],
                                                  embs[m3.start, para_idx, :]])  # 6*h
                            triple_inputs[idx].append(cur_vecs)

        # Compute local mention pair/triple representations
        pair_vecs = {}
        for t1, t2 in ALL_ENTITY_TYPE_PAIRS:
            cur_group_sizes = [len(vecs) for vecs in pair_inputs[(t1, t2)]]
            if sum(cur_group_sizes) > 0:
                cur_stack = torch.stack([
                    v for vecs in pair_inputs[(t1, t2)] for v in vecs])  # M, 4*h
                cur_m_reps = getattr(self, 'hidden_%s_%s' %
                                     (t1, t2))(cur_stack)  # M, 2*h
                cur_pair_grouped_vecs = list(torch.split(cur_m_reps, cur_group_sizes))
                for i in range(len(cur_pair_grouped_vecs)):
                    if cur_pair_grouped_vecs[i].shape[0] == 0:  # Back off
                        cur_pair_grouped_vecs[i] = getattr(
                            self, 'backoff_%s_%s' % (t1, t2))
            else:
                cur_pair_grouped_vecs = [getattr(self, 'backoff_%s_%s' % (t1, t2))
                                         for vecs in pair_inputs[(t1, t2)]]
            pair_vecs[(t1, t2)] = torch.tanh(
                self.pool(cur_pair_grouped_vecs))  # P, 2*h

        triple_group_sizes = [len(vecs) for vecs in triple_inputs]
        if sum(triple_group_sizes) > 0:
            triple_stack = torch.stack([
                v for vecs in triple_inputs for v in vecs])  # M, 6*h
            triple_m_reps = self.hidden_triple(triple_stack)  # M, 2*h
            triple_grouped_vecs = list(
                torch.split(triple_m_reps, triple_group_sizes))
            for i in range(len(triple_grouped_vecs)):
                if triple_grouped_vecs[i].shape[0] == 0:  # Back off
                    triple_grouped_vecs[i] = self.backoff_triple

        triple_vecs = torch.tanh(self.pool(triple_grouped_vecs))  # C, 2*h

        # Score candidate pairs
        pair_logits = {}
        for t1, t2 in ALL_ENTITY_TYPE_PAIRS:
            pair_logits[(t1, t2)] = getattr(self, 'out_%s_%s' % (t1, t2))(
                pair_vecs[(t1, t2)])[:, 0]  # M

        # Score candidate triples
        pair_feats_per_triple = [[], [], []]
        for c in triple_candidates:
            for i in range(3):
                pair = c.remove_entity(i)
                t1, t2 = pair.get_types()
                pair_idx = pair_to_idx[(t1, t2)][pair]
                pair_feats_per_triple[i].append(
                    pair_vecs[(t1, t2)][pair_idx, :])  # 2*h
        triple_feats = torch.cat(
            [torch.stack(pair_feats_per_triple[0]),  # C, 2*h
             torch.stack(pair_feats_per_triple[1]),  # C, 2*h
             torch.stack(pair_feats_per_triple[2]),  # C, 2*h
             triple_vecs],
            dim=1)  # C, 8*h
        final_hidden = torch.relu(self.hidden_all(triple_feats))  # C, 2*h
        triple_logits = self.out_triple(final_hidden)[:, 0]  # C
        return triple_logits, pair_logits
