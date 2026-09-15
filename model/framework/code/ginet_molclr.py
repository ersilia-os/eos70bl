import os
import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from tqdm import tqdm
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np

num_atom_type = 119 # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5 # including aromatic and self-loop edge
num_bond_direction = 3 

class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim), 
            nn.ReLU(), 
            nn.Linear(2*emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GINet(nn.Module):
    def __init__(self, num_layer=5, emb_dim=300, feat_dim=256, drop_ratio=0, pool='mean'):
        super(GINet, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool
        
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        self.out_lin = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim//2)
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        out = self.out_lin(h)
        
        return h, out
    
    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.load_state_dict(torch.load(model_file,map_location=torch.device('cpu')),strict=False)

allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def eval(model, device, loader):
    model.eval()
    node_representations = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            _, node_representation = model(batch)  # 修改为直接传递batch对象
        graph_representation = torch.mean(node_representation, dim=0)
        graph_representation = graph_representation.cpu().tolist()

        node_representations.append(graph_representation)

    return node_representations

def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

def molclr(drug_list):
    drug_fingerprint_list = []
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GINet(num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0.0, pool='mean')
    # ERSILIA FIX: resolve checkpoint relative to this file, not the CWD
    _checkpoints_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "checkpoints")
    model.from_pretrained(os.path.join(_checkpoints_dir, "gin_model.pth"))


    for x, drug in enumerate(drug_list):
        try:
            rdkit_mol = AllChem.MolFromSmiles(drug)
            if rdkit_mol is None:
                print(f"Invalid SMILES at index {x}: {drug}")
                drug_fingerprint_list.append([0]*128)  
                continue
                
            print("Processing SMILES number:", x)
            data = mol_to_graph_data_obj_simple(rdkit_mol)
            dataset = [data]
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            node_representations = eval(model, device, data_loader)
            drug_fingerprint_list.append(node_representations[0])
            
        except Exception as e:
            print(f"Error processing molecule {x}: {e}")
            drug_fingerprint_list.append([0]*128)  
            continue

    features = pd.DataFrame(drug_fingerprint_list)
    print(features)
    return features
