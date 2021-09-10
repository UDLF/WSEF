# <gcn.py>
#
# Implementation calls of GCN classifier through torch-geometric.
#
# @Authors and Contributors:
#       Lucas Pascotti Valem <lucas.valem@unesp.br>
#       João Gabriel Camacho Presotto <joaopresotto@gmail.com>
#       Nikolas Gomes de Sá <NIKOLAS567@hotmail.com>
#       Daniel Carlos Guimarães Pedronette <daniel.pedronette@unesp.br>
#
# ------------------------------------------------------------------------------
#
# This file is part of Weakly Supervised Experiments Framework (WSEF).
# Official Repository: https://github.com/UDLF/WSEF
#
# WSEF is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# WSEF is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with WSEF.  If not, see <http://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------------


import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Net(torch.nn.Module):
    def __init__(self, pNFeatures, pNNeurons, numberOfClasses):
        super(Net, self).__init__()
        self.conv1 = GCNConv(pNFeatures, pNNeurons) #dataset.num_node_features
        self.conv2 = GCNConv(pNNeurons, numberOfClasses) #dataset.num_classes

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCNClassifier():
    def __init__(self, gcn_type, rks, pN, number_neighbors=40):
        # Parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pK = number_neighbors
        self.pN = pN
        self.rks = rks
        self.pLR = 0.001
        self.pNNeurons = 32
        self.pNEpochs = 50
        self.gcn_type = gcn_type

    def fit(self, test_index, train_index, features, labels):
        # masks
        print('Creating masks ...')
        self.train_mask = []
        self.val_mask = []
        self.test_mask = []
        self.train_size = len(train_index)
        self.test_size = len(test_index)
        self.train_mask = [False for i in range(self.pN)]
        self.val_mask = [False for i in range(self.pN)]
        self.test_mask = [False for i in range(self.pN)]
        for index in train_index:
            self.train_mask[index] = True
        for index in test_index:
            self.test_mask[index] = True
        self.train_mask = torch.tensor(self.train_mask)
        self.val_mask = torch.tensor(self.val_mask)
        self.test_mask = torch.tensor(self.test_mask)
        # labels
        print('Set labels ...')
        y = labels
        self.numberOfClasses = max(y)+1
        self.y = torch.tensor(y).to(self.device)
        # features
        self.x = torch.tensor(features).to(self.device)
        self.pNFeatures = len(features[0])
        # build graph
        self.create_graph()

    def create_graph(self):
        print('Making edge list ...')
        self.top_k = self.pK
        # compute traditional knn graph
        edge_index = []
        for img1 in range(len(self.rks)):
            for pos in range(self.top_k):
                img2 = self.rks[img1][pos]
                edge_index.append([img1, img2])
        edge_index = torch.tensor(edge_index)
        # convert to torch format
        self.edge_index = edge_index.t().contiguous().to(self.device)

    def predict(self):
        # data object
        print('Loading data object...')
        data = Data(x=self.x.float(),
                    edge_index=self.edge_index,
                    y=self.y,
                    test_mask=self.test_mask,
                    train_mask=self.train_mask,
                    val_mask=self.val_mask)
        # TRAIN MODEL #
        model = Net(self.pNFeatures, self.pNNeurons, self.numberOfClasses).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.pLR, weight_decay=5e-4)

        print('Training...')
        model.train()
        for epoch in range(self.pNEpochs):
            print("Training epoch: ", epoch)
            optimizer.zero_grad()
            out = model(data)
            data.y = torch.tensor(data.y, dtype=torch.long)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

        # MODEL EVAL #
        model.eval()
        _, pred = model(data).max(dim=1)
        pred = torch.masked_select(pred, data.test_mask)

        return pred.tolist()
