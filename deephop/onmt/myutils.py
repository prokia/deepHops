import re
import numpy as np
import networkx as nx
import rdkit
import torchtext
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures, AllChem
import os
import torch
import time
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import multiprocessing
import shutil
import torch.utils.data as data

ATOMS = ['C', 'Si', 'F', 'I', 'Sn', 'N', 'P', 'Cl', 'B', 'Se', 'S', 'O', 'Br']

# ATOMS = ['Si', 'S', 'C', 'O', 'Se', 'P', 'F', 'Br', 'I', 'B', 'Cl', 'Sn', 'N']
EMB_ATOMS = ['Si', 'S', 'C', 'O', 'Se', 'P', 'F', 'Br', 'I', 'B', 'Cl', 'Sn', 'N', 'c', 'n', 'o', 's', 'p']


def extend_atoms_in_smiles(smiles):
    patterns = ['B r', 'C l', 'S i']
    t_patterns = ['Br', 'Cl', 'Si']
    for i in range(len(patterns)):
        smiles = smiles.replace(patterns[i], t_patterns[i])
    return smiles


def get_atoms(smiles):
    atoms = []
    smiles = smiles.strip().split(' ')
    for i in range(len(smiles)):
        if smiles[i] in ATOMS:
            atoms.append(smiles[i])
    return atoms


def rawsmiles2graph(smiles):
    # smiles = smiles.strip().replace(' ','')
    m = Chem.MolFromSmiles(smiles)
    g = nx.Graph()

    fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    feats = factory.GetFeaturesForMol(m)

    # Nodes
    for i in range(0, m.GetNumAtoms()):
        atom_i = m.GetAtomWithIdx(i)

        g.add_node(i, a_type=atom_i.GetSymbol(), a_num=atom_i.GetAtomicNum(), acceptor=0, donor=0,
                   aromatic=atom_i.GetIsAromatic(), hybridization=atom_i.GetHybridization(),
                   num_h=atom_i.GetTotalNumHs())

    # Donor and Acceptor properties
    for i in range(0, len(feats)):
        if feats[i].GetFamily() == 'Donor':
            node_list = feats[i].GetAtomIds()
            for i in node_list:
                g.node[i]['donor'] = 1
        elif feats[i].GetFamily() == 'Acceptor':
            node_list = feats[i].GetAtomIds()
            for i in node_list:
                g.node[i]['acceptor'] = 1

    # Edges
    for i in range(0, m.GetNumAtoms()):
        for j in range(0, m.GetNumAtoms()):
            e_ij = m.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                g.add_edge(i, j, b_type=e_ij.GetBondType())
            else:
                # Unbonded
                g.add_edge(i, j, b_type=None, )

    return g


def extract_graph_feature(g, hydrogen=True, with_3d=False):
    h = []
    # node features
    for n, d in list(g.nodes()(data=True)):
        is_atom = True
        try:
            a = d['a_type']
        except:
            is_atom = False
        if is_atom:
            h_t = []
            # Atom type (One-hot)
            h_t += [int(d['a_type'] == x) for x in ATOMS]
            # Atomic number
            h_t.append(d['a_num'])
            # Acceptor
            h_t.append(d['acceptor'])
            # Donor
            h_t.append(d['donor'])
            # Aromatic
            h_t.append(int(d['aromatic']))
            # Hybradization
            h_t += [int(d['hybridization'] == x) for x in
                    [rdkit.Chem.rdchem.HybridizationType.SP, rdkit.Chem.rdchem.HybridizationType.SP2,
                     rdkit.Chem.rdchem.HybridizationType.SP3]]
            # If number hydrogen is used as a
            if hydrogen:
                h_t.append(d['num_h'])

            if with_3d:
                h_t.extend(d['corr'])
        else:
            if not with_3d:
                h_t = [0] * 21
            else:
                h_t = [0] * 24

        h.append(h_t)

    # edge features
    remove_edges = []
    e = {}
    for n1, n2, d in list(g.edges()(data=True)):
        e_t = []
        if d['b_type'] is None:
            remove_edges += [(n1, n2)]
        else:
            e_t += [int(d['b_type'] == x) for x in
                    [rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE,
                     rdkit.Chem.rdchem.BondType.TRIPLE, rdkit.Chem.rdchem.BondType.AROMATIC]]
        if e_t:
            e[(n1, n2)] = e_t
    for edg in remove_edges:
        g.remove_edge(*edg)

    return h, e


class MolData(data.Dataset):

    def __init__(self, data, lengths):
        self.data = data
        self.lengths = lengths
        self.gs = []
        for i in range(len(data)):
            self.gs.append(smiles2graph(self.data[i], self.lengths[i]))

    def __getitem__(self, index):
        return self.gs[index]

    def __len__(self):
        return len(self.data)


def collate_dgl(samples):
    # The input `samples` is a list of graphs
    batched_graph = dgl.batch(samples)

    return batched_graph


def collate_g(batch):
    batch_sizes = np.max(np.array([[len(input_b[1]), len(input_b[1][0]), len(input_b[2]),
                                    len(list(input_b[2].values())[0])]
                                   if input_b[2] else
                                   [len(input_b[1]), len(input_b[1][0]), 0, 0]
                                   for (input_b, target_b) in batch]), axis=0)

    g = np.zeros((len(batch), batch_sizes[0], batch_sizes[0]))
    h = np.zeros((len(batch), batch_sizes[0], batch_sizes[1]))
    e = np.zeros((len(batch), batch_sizes[0], batch_sizes[0], batch_sizes[3]))

    target = np.zeros((len(batch), len(batch[0][1])))

    for i in range(len(batch)):

        num_nodes = len(batch[i][0][1])

        # Adjacency matrix
        g[i, 0:num_nodes, 0:num_nodes] = batch[i][0][0]

        # Node features
        h[i, 0:num_nodes, :] = batch[i][0][1]

        # Edges
        for edge in batch[i][0][2].keys():
            e[i, edge[0], edge[1], :] = batch[i][0][2][edge]
            e[i, edge[1], edge[0], :] = batch[i][0][2][edge]

        # Target
        target[i, :] = batch[i][1]

    g = torch.FloatTensor(g)
    h = torch.FloatTensor(h)
    e = torch.FloatTensor(e)
    target = torch.FloatTensor(target)

    return g, h, e, target


def get_all_atoms(filenames, with_reaction_type=True):
    atom_set = set()
    for filename in filenames:
        data = []
        with open(filename, 'r') as f:
            data = f.readlines()
        for i in range(len(data)):
            if with_reaction_type:
                data[i] = ''.join(data[i].strip().split()[1:])
            else:
                data[i] = ''.join(data[i].strip().split())
            m = Chem.MolFromSmiles(data[i])
            for i in range(0, m.GetNumAtoms()):
                atom_i = m.GetAtomWithIdx(i)
                symbol = atom_i.GetSymbol()
                atom_set.add(symbol)
    return atom_set


def canonicalize(filenames, atom_set):
    for filename in filenames:
        with open(filename, 'r') as f:
            data = f.readlines()
        for i in range(len(data)):
            for atom in atom_set:
                if len(atom) > 1:
                    data[i] = data[i].strip().replace(atom[0] + ' ' + atom[1], atom)

        with open('modify_data/' + filename, 'w') as f:
            for line in data:
                f.write(line)
                f.write('\n')


vocab = {'<unk>': 0, '<blank>': 1, '<s>': 2, '</s>': 3, 'c': 4, 'C': 5, '(': 6, ')': 7, '1': 8, 'O': 9, '2': 10,
         '=': 11, 'N': 12, 'n': 13, '3': 14, 'F': 15, '[': 16, ']': 17, '@': 18, 'H': 19, '-': 20, 'Cl': 21, '.': 22,
         '4': 23, 'S': 24, 'Br': 25, '#': 26, 's': 27, '+': 28, 'o': 29, '5': 30, '/': 31, 'B': 32, 'I': 33, 'Si': 34,
         '\\': 35, '6': 36, 'P': 37, 'M': 38, 'g': 39, 'Sn': 40, '7': 41, 'Z': 42, 'u': 43, 'e': 44, 'L': 45, 'i': 46,
         '8': 47, 'Se': 48, '9': 49, 'K': 50, 't': 51, 'd': 52}

# vocab = {'<unk>': 0, '<blank>': 1, '<s>': 2, '</s>': 3, 'c': 4, 'C': 5, '(': 6, ')': 7, '1': 8, 'O': 9, '2': 10, '=': 11, 'N': 12, 'n': 13, '3': 14, 'F': 15, '[': 16, ']': 17, '@': 18, 'H': 19, '-': 20, 'Cl': 21, '.': 22, '4': 23, 'S': 24, 'Br': 25, '<RX_1>': 26, '<RX_2>': 27, '#': 28, 's': 29, '<RX_6>': 30, '+': 31, 'o': 32, '<RX_3>': 33, '5': 34, '/': 35, '<RX_7>': 36, 'B': 37, 'I': 38, 'Si': 39, '<RX_9>': 40, '\\': 41, '<RX_4>': 42, '<RX_8>': 43, '6': 44, 'P': 45, '<RX_5>': 46, 'M': 47, 'g': 48, 'Sn': 49, '<RX_10>': 50, '7': 51, 'Z': 52, 'u': 53, 'e': 54, 'L': 55, 'i': 56, '8': 57, 'Se': 58, '9': 59, 'K': 60, 't': 61, 'd': 62}
inver_vocab = {vocab[key]: key for key in vocab}

invalid_words = ['<unk>', '<RX_9>', '<RX_5>', '<RX_2>', '<blank>', '<RX_10>', '<RX_8>', '</s>', '<RX_6>', '<s>',
                 '<RX_3>', '<RX_4>', '<RX_1>', '<RX_7>']


def recover_to_raw(src):  # len * batch
    src = src.transpose(0, 1).contiguous()  # batch * len

    w_batch, w_len = src.size()
    rawstr = []
    for i in range(w_batch):
        smile = []
        for j in range(w_len):
            word = inver_vocab[src[i][j].item()]
            smile.append(word)
        rawstr.append(smile)
    return rawstr


def mpnn_emb(model, g_loader, ccuda=1):
    for i, (g, h, e, target) in enumerate(g_loader):
        if ccuda >= 0:
            g = g.cuda(ccuda)
            h = h.cuda(ccuda)
            e = e.cuda(ccuda)

        return model(g, h, e)


def gcn_emb(model, gs, device):
    batched_graph = dgl.batch(gs)
    batched_graph = batched_graph.to(device)
    graph_encode_emb = model(batched_graph, batched_graph.ndata['init_h'])
    return graph_encode_emb


def check_num_zero(emb):
    num = []
    e1, e2, e3 = emb.size()
    for i in range(e1):
        n = 0
        for j in range(e2):
            if torch.sum(emb[i][j] == torch.zeros(e3).cuda()).item() == e3:
                n += 1
        num.append(n)
    return num


# def cat_two_emb(emb1, emb2):
#     e1, e2 = emb1.size(2), emb2.size(2)
#     src = src.transpose(0, 1).contiguous()[:,:,0] # batch * len
#     w_batch, w_len = src.size()
#     pre_pad = torch.zeros(w_batch, w_len, e2).cuda(0)
#     emb1 = torch.cat((emb1, em), dim = 3)

#     # num_atom = []
#     for i in range(w_batch):
#         num = 0
#         index = 0
#         for j in range(w_len):
#             if inver_vocab[src[i][j].item()] in EMB_ATOMS:
#              num += 1
#              emb1[i][j][e1:] = emb2[i][index]
#              index += 1
#         # num_atom.append(num)
#         # print(emb2[i][index:])    # should be all zero tensors
#     # print(num_atom)
#     return emb1


def need_emb(word, EMB_ATOMS):
    return word in EMB_ATOMS


def str2molgraph(rawstr,
                 length):  # rawstr :tuple() e.g. ('<RX_6>', 'N', 'c', '1', 'n', 'c', '2', '[', 'n', 'H', ']', 'c', '(', 'C', 'C', 'C', 'c', '3', 'c', 's', 'c', '(', 'C', '(', '=', 'O', ')', 'O', ')', 'c', '3', ')', 'c', 'c', '2', 'c', '(', '=', 'O', ')', '[', 'n', 'H', ']', '1')
    # if length == 1:
    #     rawstr = [s for s in rawstr[0]]
    #     length = len(rawstr)
    smiles = ''.join(rawstr[:length])

    m = Chem.MolFromSmiles(smiles)
    code = AllChem.EmbedMolecule(m)
    code = AllChem.MMFFOptimizeMolecule(m, maxIters=200)

    g = nx.Graph()
    fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    feats = factory.GetFeaturesForMol(m)

    atom_true_index = {}
    atom_index = 0
    # Nodes
    for i in range(len(rawstr)):
        if not need_emb(rawstr[i], EMB_ATOMS):
            g.add_node(i)

        else:
            atom_true_index[atom_index] = i  # meanwhile, set a map dict to find the true index of atoms
            atom_i = m.GetAtomWithIdx(atom_index)

            g.add_node(i, a_type=atom_i.GetSymbol(), a_num=atom_i.GetAtomicNum(), acceptor=0, donor=0,
                       aromatic=atom_i.GetIsAromatic(), hybridization=atom_i.GetHybridization(),
                       num_h=atom_i.GetTotalNumHs(), corr=list(m.GetConformer().GetAtomPosition(atom_index)))
            atom_index += 1

    # Donor and Acceptor properties
    for i in range(0, len(feats)):
        if feats[i].GetFamily() == 'Donor':
            node_list = feats[i].GetAtomIds()
            for i in node_list:
                g.nodes[atom_true_index[i]]['donor'] = 1
        elif feats[i].GetFamily() == 'Acceptor':
            node_list = feats[i].GetAtomIds()
            for i in node_list:
                g.nodes[atom_true_index[i]]['acceptor'] = 1

    # Edges
    for i in range(0, m.GetNumAtoms()):
        for j in range(0, m.GetNumAtoms()):
            e_ij = m.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                g.add_edge(atom_true_index[i], atom_true_index[j], b_type=e_ij.GetBondType())

    return g


# DGL molecular graph
def smiles2graph(rawstr, length):
    g = str2molgraph(rawstr, length)

    h = extract_graph_feature(g)
    g = dgl.DGLGraph(g)

    g.ndata['init_h'] = torch.tensor(h).float()
    return g


# rawstr = ('<RX_6>', 'N', 'c', '1', 'n', 'c', '2', '[', 'n', 'H', ']', 'c', '(', 'C', 'C', 'C', 'c', '3', 'c', 's', 'c', '(', 'C', '(', '=', 'O', ')', 'O', ')', 'c', '3', ')', 'c', 'c', '2', 'c', '(', '=', 'O', ')', '[', 'n', 'H', ']', '1')    
# rawsmiles2graph(rawstr, len(rawstr))


def str2graph(rawstr, with_3d_confomer):
    length = len(rawstr)
    g = str2molgraph(rawstr, length)
    h, e = extract_graph_feature(g, with_3d=with_3d_confomer)
    g = dgl.DGLGraph(g)
    g.ndata['init_h'] = torch.tensor(h).float()
    for key in e:
        a, b = key[0], key[1]
        g.edges[a, b].data['w'] = torch.tensor(e[key]).unsqueeze(0).float()
        g.edges[b, a].data['w'] = torch.tensor(e[key]).unsqueeze(0).float()
    return g


def pad_for_graph(gs, length):
    for i in range(len(gs)):
        if gs[i].number_of_nodes() < length:
            n = length - gs[i].number_of_nodes()
            gs[i].add_nodes(n)
            # gs[i].ndata['init_h'] = torch.cat((gs[i].ndata['init_h'], torch.zeros(n, 21)), dim = 0)
            # It can add all-zero node feature vectors automatically.
    return gs


def samegraph(rawstr):
    g = dgl.DGLGraph()
    g.add_nodes(len(rawstr))
    g.ndata['init_h'] = torch.zeros(len(rawstr), 21).float()
    return g


def add_more_field(fields):
    fields['graph'] = torchtext.data.Field(sequential=False)
    fields['condition_target'] = torchtext.data.Field(sequential=False)

# rawstr = ('<RX_6>', 'N', 'c', '1', 'n', 'c', '2', '[', 'n', 'H', ']', 'c', '(', 'C', 'C', 'C', 'c', '3', 'c', 's', 'c', '(', 'C', '(', '=', 'O', ')', 'O', ')', 'c', '3', ')', 'c', 'c', '2', 'c', '(', '=', 'O', ')', '[', 'n', 'H', ']', '1')
# g = smiles2graph(rawstr, len(rawstr))
# from MYGCN import *

# model = MGCN(21,21,3)
# out = model(g, g.ndata['init_h'])
# print(out.size())
