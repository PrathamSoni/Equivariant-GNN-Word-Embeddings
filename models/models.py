from torch.nn import *
import gvp
import torch


class GraphEncoder(Module):
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
        self.arc_label_layers = [Linear(100, 100, bias=False) for i in range(len(dep_map))]

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

        sents = []
        for i in range(x.num_graphs):
            sents.append(gvp.tuple_index(pass2, slice(x.ptr[i], x.ptr[i+1])))

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
