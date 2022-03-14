import torch
from models import GraphModelDataWrapper, GraphModel
from data.datasets import get_dataset
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
pretrain_dataset = get_dataset("billion", "pretrain", "wikipedia")

enc_model = GraphModel((pretrain_dataset.encoder.base_dim, 2), (37, 1),
                       vector_dim=pretrain_dataset.encoder.base_dim,
                       pos_map=pretrain_dataset.pos_map,
                       dep_map=pretrain_dataset.dependency_map).to(device)
enc_model.load_state_dict(torch.load(os.path.join("runs/ner", 'encoder.pt'), map_location=device))
enc_model.eval()
embedder = GraphModelDataWrapper(pretrain_dataset.encoder, enc_model,
                                 pos_map=pretrain_dataset.pos_map,
                                 dep_map=pretrain_dataset.dependency_map).to(device)
embedder.eval()

a, b = embedder.example("he leaves the pile of leaves in disarray")
c, d = embedder.example("he finds the pile of leaves in disarray")
e, f = embedder.example("he leaves the pile of sticks in disarray")
g, h = embedder.example("he finds the pile of sticks in disarray")
vecs = [a[1], a[5], c[1], c[5], e[1], e[5], g[1], g[5], b[1], d[1], h[5]]

# print(a[1], a[5])
# print(b[1], b[5])
X = np.concatenate([np.expand_dims(x.detach().numpy(), axis=0).T for x in vecs], axis=1)
# plt.imshow(X, interpolation='none')
# plt.show()

pca = PCA(n_components=2)
pca.fit(X.T)
X_pca = pca.transform(X.T)
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=[0, 1, 2, 1, 0, 3, 2, 3, 4, 5, 6])
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend1)
plt.show()
plt.clf()