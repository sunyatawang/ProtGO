# Setup
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["WANDB_DISABLED"] = "true"

import torch
import numpy as np
import math
import random

seed = 7

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

import esm
import ankh
import clip
import csv
import pickle

from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import EvalPrediction
import datasets
from datasets import load_dataset

from sklearn import metrics
from scipy import stats
from functools import partial
import pandas as pd
from tqdm.auto import tqdm
import scipy.stats as stats
from datetime import datetime
import re

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
# import wandb
# wandb.init(mode='offline')
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, matthews_corrcoef, multilabel_confusion_matrix, \
    precision_recall_fscore_support

from graph import Graph, Prediction, propagate
from parser import obo_parser, gt_parser, pred_parser, ia_parser
from evaluation import get_leafs_idx, get_roots_idx, evaluate_prediction

from transformers import AutoTokenizer, AutoModel

nlp_path = '/BioLinkBERT-large/'
# nlp_path = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract/'
nlp_tokenizer = AutoTokenizer.from_pretrained(nlp_path)
nlp_model = AutoModel.from_pretrained(nlp_path)
nlp_dim = 1024
# nlp_dim = 768
nlp_model.cuda()
nlp_model.eval()


# nlp_tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
# nlp_model = AutoModel.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')

def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


# Select the available device
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    # for i in range(torch.cuda.device_count()):
    #     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print("Available device:", torch.cuda.current_device())
else:
    print('Available device: CPU')
# import pdb; pdb.set_trace()

# Load pretrained model
# model, tokenizer = esm.pretrained.esm2_t33_650M_UR50D()
# num_layers = 33
# embed_dim = 1280

model, tokenizer = esm.pretrained.esm2_t30_150M_UR50D()
num_layers = 30
embed_dim = 640

# model, tokenizer = esm.pretrained.esm2_t36_3B_UR50D()
# num_layers = 36
# embed_dim = 2560

# model, tokenizer = esm.pretrained.esm2_t48_15B_UR50D()
# num_layers = 48
# embed_dim = 5120
model.cuda()
model.eval()

train_path = "Kaggle/FunP/kaggle_train_0.9.txt"
test_path = "Kaggle/FunP/kaggle_test_0.9.txt"
output_path = "eval/kaggle/function"
# predict_path = '/Kaggle/FunP/Test (Targets)/testsuperset2.fasta'

label_space = {
    'biological_process': set(),
    'molecular_function': set(),
    'cellular_component': set()
}
obo_path = 'Kaggle/FunP/Train/go-basic.obo'
ia_path = 'Kaggle/FunP/IA.txt'

enc = preprocessing.LabelEncoder()
# MAXLEN = 7000
# MAXLEN = 6000
MAXLEN = 2048

tax_path = 'Kaggle/FunP/Train/train_taxonomy.tsv'


def extract_know(filepath):
    path = filepath
    train_id = []
    train_tax = []
    print(f"Start reading {path}")
    with open(path, "r") as lines:
        for _line in lines:
            if _line.startswith('EntryID'):
                continue
            _line = _line.strip()
            seqs = _line.split()
            _id = seqs[0]
            train_id.append(_id)
            _tax = seqs[1]
            train_tax.append(_tax)
    print("Read input complete")
    return train_id, train_tax


class TransformSize(nn.Module):
    def __init__(self, input_size, output_size):
        super(TransformSize, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        torch.nn.init.kaiming_normal_(self.fc.weight)
        self.fc.bias.data.fill_(0.01)

    def forward(self, x):
        return self.fc(x)


def embed_know(ground, extract, taxon):
    tax_set = set(taxon)
    tax_list = list(tax_set)
    enctax = preprocessing.LabelEncoder()
    taxspace = enctax.fit_transform(tax_list)
    tax_num = len(enctax.classes_)
    ground_taxon = [taxon[extract.index(_id)] for _id in ground if _id in extract]
    tax_list = []
    tax_fin = []
    for i in range(len(ground_taxon)):
        extract_tax = torch.zeros(tax_num)
        tax_list.append(extract_tax)
    # model_tax = TransformSize(tax_num, embed_dim).cuda()
    for i, _taxon in enumerate(ground_taxon):
        temp_tax = enctax.transform([_taxon])
        tax_list[i][temp_tax[0]] = 1.0
    for _tax in tax_list:
        # trans_tax = model_tax(_tax.cuda())
        tax_fin.append(_tax)
    # x = 0
    # for label in training_labels[key]:
    #     temp_labels = enc.transform(label)
    #     training_labels[key][x] = [1 if i in temp_labels else 0 for i in range(0, label_num)]
    #     # print(len(training_labels[x]))
    #     x += 1
    return tax_fin


# Parse the OBO file and creates a different graph for each namespace
def obo_graph(filepath, dict_path):
    ia_dict = None
    if dict_path is not None:
        ia_dict = ia_parser(dict_path)

    ontologies = []
    no_orphans = False
    for ns, terms_dict in obo_parser(filepath).items():
        ontologies.append(Graph(ns, terms_dict, ia_dict, not no_orphans))
        # logging.info("Ontology: {}, roots {}, leaves {}".format(ns, len(get_roots_idx(ontologies[-1].dag)), len(get_leafs_idx(ontologies[-1].dag))))
    return ontologies, ia_dict


onto, ia_dict = obo_graph(obo_path, ia_path)


# gts = gt_parser(gt_file, onto)

def parent(enc, key):
    onto_parent = {}
    label_num = len(enc.classes_)
    for i in range(label_num):
        _label = enc.inverse_transform([i])
        _tag = 'GO:' + str(_label[0])
        if i not in onto_parent.keys():
            onto_parent[i] = {
                'size': 0,
                'pos': []
            }
        for ont in onto:
            if ont.namespace != key:
                continue
            for term in ont.terms_list:
                if term['id'] == _tag:
                    ns = ont.namespace
                    parent_ids = term['adj']
                    if len(parent_ids) == 0:
                        continue
                    else:
                        for _parent in parent_ids:
                            for _key, val in ont.terms_dict.items():
                                if 'index' in val and val['index'] == _parent:
                                    poss_tags = _key[3:]
                                    if poss_tags not in label_space[
                                        key]:  # 'alt_id' is used in this version, has to exclude from ground-truth label space
                                        continue
                                    _pos = enc.transform([poss_tags])
                                    onto_parent[i]['size'] += 1
                                    onto_parent[i]['pos'].extend(_pos)
    return onto_parent


def preprocess_dataset(filepath, max_length=None):
    '''
        Args:
            sequences: list, the list which contains the protein primary sequences.
            labels: list, the list which contains the dataset labels.
            max_length, Integer, the maximum sequence length,
            if there is a sequence that is larger than the specified sequence length will be post-truncated.
    '''
    pro_id = []
    sequences = []
    labels = {
        'biological_process': [],
        'molecular_function': [],
        'cellular_component': []
    }
    multi_labels = {
        'biological_process': [],
        'molecular_function': [],
        'cellular_component': []
    }
    path = filepath
    print(f"Start reading {path}")
    with open(path, "r") as lines:
        for _line in lines:
            if _line.startswith('>'):
                _line = _line.strip()
                seqs = _line.split()
                _id = seqs[0][1:]
                pro_id.append(_id)
                tags = seqs[1].split(';')
                for tag in tags:
                    gene = 'GO:' + tag
                    for ont in onto:
                        ns = ont.namespace
                        if gene in ont.terms_dict.keys():
                            multi_labels[ns].append(tag)
                            label_space[ns].add(tag)
                            #     index = ont.terms_dict[gene]['index']
                            continue
                for key in multi_labels.keys():
                    labels[key].append(multi_labels[key])
                multi_labels = {
                    'biological_process': [],
                    'molecular_function': [],
                    'cellular_component': []
                }
            else:
                _line = _line.strip()
                if len(_line) > MAXLEN:
                    _line = _line[:MAXLEN]
                sequences.append(_line)
    print("Read input complete")
    # if max_length is None:
    #     max_length = len(max(sequences, key=lambda x: len(x)))
    # splitted_sequences = [list(seq[:max_length]) for seq in sequences]
    return pro_id, sequences, labels


# Initialization embedding
def embed_dataset(sample, taxon, nlp_embed, max_len=None):
    batch_converter = tokenizer.get_batch_converter()
    # embedding_list = []
    # weight_list = []
    # batch_size = 512
    # num_batches = len(sample) // batch_size + (1 if len(sample) % batch_size != 0 else 0)
    sample = sample[0]
    batch_labels, batch_strs, batch_tokens = batch_converter([("x", sample)])
    with torch.no_grad():
        batch_tokens = batch_tokens.cuda()
        results = model(batch_tokens, repr_layers=[num_layers])
        token_representations = results["representations"][num_layers]
        token_representations = token_representations.detach()
        plm_embed = token_representations[0, 1:1 + len(sample), :].clone().detach()
        if nlp_embed != None:
            embedding = clip_align(plm_embed, nlp_embed)
        else:
            embedding = plm_embed
        # import pdb; pdb.set_trace()
        embedding = torch.mean(embedding, dim=0)
        # embedding_list.append(embedding)
    tax_dim = taxon.size(0)
    model_tax = TransformSize(tax_dim, embed_dim).cuda()
    taxon = model_tax(taxon.cuda())
    embedding = embedding + 0.1 * taxon
    # mode = filepath.split("/")[-1].split("_src")[0]
    # embedding_path = "{}/esm2_t48_15B_UR50D_{}.out".format(embed_path, mode)
    # with open(embedding_path, 'wb') as embhandle:
    #     pickle.dump(inputs_embedding, embhandle)
    return embedding


# def embed_dataset(model, filepath):
#     with open(filepath, 'rb') as handle:
#         inputs_embedding = pickle.load(handle)
#     return inputs_embedding

# Extract sequences embeddings
class StabilitylandscapeDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __getitem__(self, idx):
        embedding = self.sequences[idx]
        label = self.labels[idx]
        return {'embed': embedding, 'labels': torch.as_tensor(label, dtype=torch.float32).clone().detach()}

    def __len__(self):
        return len(self.sequences)


def get_model_config(config):
    model_config = AutoConfig.from_pretrained(config.model, output_hidden_states=True)
    model_config.hidden_dropout = 0.0
    model_config.hidden_dropout_prob = 0.0
    model_config.attention_dropout = 0.0
    model_config.attention_probs_dropout_prob = 0.0
    return model_config


# class CustomModel(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(CustomModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, output_size)
#         torch.nn.init.kaiming_normal_(self.fc1.weight)
#         self.fc1.bias.data.fill_(0.01)
#         # self.relu1 = nn.ReLU()
#         # self.fc2 = nn.Linear(1024, 2048)
#         # self.relu2 = nn.ReLU()
#         # self.fc3 = nn.Linear(2048, 4096)
#         # self.relu3 = nn.ReLU()
#         # self.fc4 = nn.Linear(4096, output_size)
#         # torch.nn.init.kaiming_normal_(self.fc1.weight)
#         # torch.nn.init.kaiming_normal_(self.fc2.weight)
#         # torch.nn.init.kaiming_normal_(self.fc3.weight)
#         # torch.nn.init.kaiming_normal_(self.fc4.weight)
#         self.final_activation = torch.nn.Sigmoid()

#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)

#     def forward(self, x):
#         x = self.fc1(x)
#         return x

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        torch.nn.init.kaiming_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support.transpose(0, 1)).t()
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class CustomModel(nn.Module):
    def __init__(self, input_size, output_size, adj):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 10240)  # original 10240
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(10240, output_size)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)
        self.gc1 = GraphConvolution(output_size, output_size)
        # self.relu2 = nn.ReLU()
        # self.gc2 = GraphConvolution(5120, output_size)
        # self.final_activation = torch.nn.Sigmoid()
        self.adj = adj

    def _init_weights(self, module):
        std = math.sqrt(2. / module.weight.data.size()[1])
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        # x = self.relu2(self.gc1(x, self.adj))
        # x = nn.functional.dropout(x, 0.5, training=self.training)
        x = self.gc1(x, self.adj)
        return x


def calACC(hypo, gold):
    _correct = 0
    for i in range(0, len(hypo)):
        _hypo = np.array(hypo[i].cpu())
        _gold = np.array(gold[i].cpu())
        _hypo[_hypo >= 0.5] = 1
        _hypo[_hypo < 0.5] = 0
        correct = (_hypo == _gold).sum()
        total = _gold.shape[0]
        correct = correct / total
        _correct += correct
    total = len(gold)
    return _correct / total


def calROC(hypo, gold):
    # print([(h, g) for h, g in zip(hypo, gold)])
    roc = 0
    for i in range(0, len(hypo)):
        _hypo = np.array(hypo[i].cpu())
        _gold = np.array(gold[i].cpu())
        roc += roc_auc_score(_gold, _hypo)
    total = len(gold)
    return roc / total


def calMCC(hypo, gold):
    mcc = 0
    for i in range(0, len(hypo)):
        _hypo = np.array(hypo[i].cpu())
        _gold = np.array(gold[i].cpu())
        _hypo[_hypo >= 0.5] = 1
        _hypo[_hypo < 0.5] = -1
        _gold[_gold == 0] = -1
        mcc = matthews_corrcoef(_gold, _hypo)
    total = len(gold)
    return mcc / total


def calF(hypo, gold, return_all=False):
    b_f1 = ma_f1 = mi_f1 = 0
    b_p = b_r = 0
    for i in range(0, len(hypo)):
        _hypo = np.array(hypo[i].cpu())
        _gold = np.array(gold[i].cpu())
        _hypo[_hypo >= 0.5] = 1
        _hypo[_hypo < 0.5] = 0
        p, r, f, _ = precision_recall_fscore_support(_gold, _hypo, average='binary', zero_division=1)
        b_f1 += f
        b_p += p
        b_r += r
        p, r, f, _ = precision_recall_fscore_support(_gold, _hypo, average='macro', zero_division=1)
        ma_f1 += f
        p, r, f, _ = precision_recall_fscore_support(_gold, _hypo, average='micro', zero_division=1)
        mi_f1 += f
    total = len(gold)
    return b_f1 / total, b_p / total, b_r / total, ma_f1 / total, mi_f1 / total


pred_id = []


def pred_dataset(filepath):
    predict_seqs = []
    start = -1
    seqs = ''
    with open(filepath, "r") as lines:
        for _line in lines:
            if _line.startswith('>'):
                _id = _line.split()
                _id = _id[0][1:]
                pred_id.append(_id)
                start = -1
                continue
            if start == -1 and seqs != '':
                predict_seqs.append(seqs)
                seqs = ''
            start = 1
            seqs += _line.strip()
        predict_seqs.append(seqs)
    x_size = len(predict_seqs)
    max_length = len(max(predict_seqs, key=lambda x: len(x)))
    splitted_seqs = [list(seq[:max_length]) for seq in predict_seqs]
    predict_embeddings = embed_dataset(model, splitted_seqs, predict_path)
    return predict_embeddings


def PredictProcess(predict_embeddings, y_size):
    x_size = len(predict_embeddings)
    predict_labels = torch.zeros(x_size, y_size)
    dataset = StabilitylandscapeDataset(predict_embeddings, predict_labels)
    return dataset


def nlp_embedding(nlp_model, label_list, key):
    print(f"Start nlp model {nlp_path}")
    match_embedding = [None for _ in range(len(label_list))]
    for index, multi_tag in enumerate(label_list):
        if multi_tag == []:
            continue
        context = ''
        for _tag in multi_tag:
            for ont in onto:
                if ont.namespace != key:
                    continue
                _tag = 'GO:' + _tag
                if _tag in ont.terms_dict.keys():
                    # tag_context = ont.terms_dict[_tag]['def']
                    # tag_contents = re.findall(r'"(.*?)"', tag_context)
                    # if context == '':
                    #     context = context + tag_contents[0]
                    # else:
                    #     context = context + ' ' + tag_contents[0]
                    tag_context = ont.terms_dict[_tag]['name']
                    context = context + tag_context + ' '
        seq_len = 512
        num_seqs = len(context) // seq_len + (1 if len(context) % seq_len != 0 else 0)
        # if num_seqs == 1:
        #     last_hidden_states = torch.zeros(num_seqs, len(context), 1024)
        # else:
        #     last_hidden_states = torch.zeros(num_seqs, seq_len, 1024)
        last_embed = []
        for i in range(num_seqs):
            start_index = i * seq_len
            end_index = min((i + 1) * seq_len, len(context))
            context_sample = context[start_index:end_index]
            inputs = nlp_tokenizer(context_sample, return_tensors="pt")
            inputs = {k: v.cuda() for k, v in inputs.items()}
            # print(inputs)
            outputs = nlp_model(**inputs)
            last_hidden_states = outputs.last_hidden_state.squeeze(0).detach()
            # print(last_hidden_states.size())
            last_embed.append(last_hidden_states)
        embed = torch.cat(last_embed, dim=0)
        match_embedding[index] = embed
        # import pdb; pdb.set_trace()
    return match_embedding


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, hidden_dim, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
        pe = torch.zeros(seq_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultimodalTransformer(nn.Module):
    def __init__(self, text_embed, protein_embed, hidden_dim, num_heads, num_layers):
        super(MultimodalTransformer, self).__init__()
        self.text_embed = text_embed
        self.protein_embed = protein_embed
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.transformer_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layers, num_layers=num_layers)
        self.text_pos_encoder = PositionalEncoding(text_embed.size()[0], hidden_dim)
        self.protein_pos_encoder = PositionalEncoding(protein_embed.size()[0], hidden_dim)

    def forward(self, text_inputs, protein_inputs):
        # text_features = self.text_pos_encoder(self.text_embed)
        protein_features = self.protein_pos_encoder(self.protein_embed)
        # import pdb; pdb.set_trace()
        combined_features = torch.cat((self.text_embed, protein_features), dim=0)
        attn_output, _ = self.multihead_attn(combined_features, combined_features, combined_features)
        transformer_output = self.transformer_encoder(attn_output)
        return transformer_output


def clip_align(plm_embed, nlp_embed):
    ## cosine alignment
    # nlp_projection = TransformSize(nlp_dim, embed_dim).cuda()
    # plm_embed = nn.functional.normalize(plm_embed, p=2, dim=1)
    # for nlp_embed in nlp_list:
    #     nlp_embed = nn.functional.normalize(nlp_embed, p=2, dim=1).cuda()
    #     nlp_project = nlp_projection(nlp_embed)
    # similarity = nn.functional.cosine_similarity(nlp_project[0], plm_embed[0], dim=0)
    nlp_projection = TransformSize(nlp_dim, embed_dim).cuda()
    nlp_project = nlp_projection(nlp_embed.cuda())
    multimodal_transformer = MultimodalTransformer(
        nlp_project,
        plm_embed,
        hidden_dim=embed_dim,
        num_heads=8,
        num_layers=6,
    ).cuda()
    multimodal_transformer.train()
    plm_embed_normal = nn.functional.normalize(plm_embed, p=2, dim=1)
    nlp_project_normal = nn.functional.normalize(nlp_project, p=2, dim=1)
    similarity_matrix = plm_embed_normal @ nlp_project_normal.t()
    # import pdb; pdb.set_trace()
    mean_similarities = similarity_matrix.mean(dim=0)
    indices_to_remain = torch.where(mean_similarities > 0)[0]
    nlp_filtered = torch.index_select(nlp_project, dim=0, index=indices_to_remain)
    output_embed = multimodal_transformer(nlp_project, plm_embed)
    return output_embed


if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    ctime = datetime.now().strftime("%Y%m%d%H%M%S")
    print('Start running date:{}'.format(ctime))
    train_id, training_sequences, training_labels = preprocess_dataset(train_path)
    # validation_sequences, validation_labels = preprocess_dataset(validation_sequences, validation_labels)
    # valid_sequences, valid_labels = preprocess_dataset(valid_path)
    test_id, test_sequences, test_labels = preprocess_dataset(test_path)

    extract_id, extract_tax = extract_know(tax_path)
    train_tax = embed_know(train_id, extract_id, extract_tax)
    test_tax = embed_know(test_id, extract_id, extract_tax)
    # test_tax = []
    # for i in range(len(test_sequences)):
    #     temp_tax = torch.zeros(embed_dim)
    #     test_tax.append(temp_tax)

    # clip_align(train_plm, train_nlp)
    # clip_align(test_plm, test_nlp)

    pred_results = {}
    metrics_output_test = {}
    for key in label_space.keys():

        train_nlp = nlp_embedding(nlp_model, training_labels[key], key)
        test_nlp = []
        for x in test_id:
            test_nlp.append(None)

        label_list = list(label_space[key])
        labspace = enc.fit_transform(label_list)
        onto_parent = parent(enc, key)
        x = 0
        label_num = len(enc.classes_)
        for label in training_labels[key]:
            temp_labels = enc.transform(label)
            training_labels[key][x] = [1 if i in temp_labels else 0 for i in range(0, label_num)]
            # print(len(training_labels[x]))
            x += 1
        x = 0
        for label in test_labels[key]:
            temp_labels = enc.transform(label)
            test_labels[key][x] = [1 if i in temp_labels else 0 for i in range(0, label_num)]
            x += 1
        ia_list = torch.ones(1, label_num).cuda()
        for _tag, _value in ia_dict.items():
            _tag = _tag[3:]
            if _tag not in label_list:
                continue
            ia_id = enc.transform([_tag])
            if _value == 0.0:
                _value = 1.0
            ia_list[0, ia_id[0]] = _value
        adj_matrix = torch.zeros(label_num, label_num).cuda()
        for i in range(label_num):
            position = onto_parent[i]['pos'].copy()
            adj_matrix[i, i] = 1.0
            for j in position:
                adj_matrix[i, j] = 1.0
        adj_matrix.to_sparse()

        training_dataset = StabilitylandscapeDataset(training_sequences, training_labels[key])
        # validation_dataset = StabilitylandscapeDataset(validation_embeddings, validation_labels)
        # valid_dataset = StabilitylandscapeDataset(valid_embeddings, valid_labels)
        test_dataset = StabilitylandscapeDataset(test_sequences, test_labels[key])
        # pred_dataset = PredictProcess(predict_embeddings, label_num)

        train_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True)
        # valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        # pred_dataloader = DataLoader(pred_dataset, batch_size=1, shuffle=False)

        model_mlp = CustomModel(embed_dim, label_num, adj_matrix).cuda()
        # criterion = nn.MultiLabelSoftMarginLoss()
        criterion = nn.BCELoss()
        e = math.e
        optimizer = torch.optim.Adam(model_mlp.parameters(), lr=1e-4)  # 4e-5 150M 1e-5 3B
        epoch_num = 20
        # metrics_output_valid = {
        #     'acc':[],
        #     'auc':[],
        #     'mcc':[],
        #     'f1':[]
        # }
        if key not in metrics_output_test:
            metrics_output_test[key] = {
                'acc': [],
                'ma_f1': [],
                'mi_f1': [],
                'b_p': [],
                'b_r': [],
                'b_f1': [],
                'w_f1': []
            }
        best_f1 = 0
        best_model_weights = None
        optimizer_model_weights = None
        sigmoid = torch.nn.Sigmoid()

        for epoch in range(epoch_num):
            loss_mean = 0
            for i, (input_ids) in tqdm(enumerate(train_dataloader)):
                optimizer.zero_grad()
                embed = input_ids['embed']
                embed_fusion = embed_dataset(embed, train_tax[i], train_nlp[i]).unsqueeze(0).cuda()
                output = model_mlp(embed_fusion).cuda()
                output = sigmoid(output)
                labels = input_ids['labels'].cuda()
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                loss_mean += loss.item()
                if (i + 1) % 100 == 0:
                    print('{}  Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(key, epoch + 1, epoch_num, i + 1,
                                                                                 len(training_dataset) // 1,
                                                                                 loss_mean / (i + 1)))
            # _labels=[]
            # _preds=[]
            # with torch.no_grad():
            #     for input_ids in tqdm(valid_dataloader):
            #         embed = input_ids['embed'].cuda()
            #         labels = input_ids['labels'].cuda()
            #         output = model_mlp(embed).cuda()
            #         labels = labels.argmax(dim=1)
            #         predicted = output.argmax(dim=1)
            #         _labels.append(labels)
            #         _preds.append(predicted)
            # acc = calACC(torch.tensor(_preds).cpu(), torch.tensor(_labels).cpu())
            # auc = calROC(torch.tensor(_preds).cpu(), torch.tensor(_labels).cpu())
            # mcc = calMCC(torch.tensor(_preds).cpu(), torch.tensor(_labels).cpu())
            # f1 = calF(torch.tensor(_preds).cpu(), torch.tensor(_labels).cpu())
            # print('Epoch: {}, Val Accuracy: {:.2f}%, Val AUC:{:.2f}%, Val MCC:{:.2f}%, Val F1:{:.2f}%'.format(epoch+1, 100*acc, 100*auc, 100*mcc, 100*f1))
            # metrics_output_valid['acc'].append(acc)
            # metrics_output_valid['auc'].append(auc)
            # metrics_output_valid['mcc'].append(mcc)
            # metrics_output_valid['f1'].append(f1)
            # test
            _labels = []
            _preds = []
            weight_preds = []
            with torch.no_grad():
                for i, (input_ids) in tqdm(enumerate(test_dataloader)):
                    embed = input_ids['embed']
                    embed_fusion = embed_dataset(embed, test_tax[i], test_nlp[i]).unsqueeze(0).cuda()
                    labels = input_ids['labels'].squeeze(0)
                    output = model_mlp(embed_fusion)
                    w_output = output * ia_list
                    output = sigmoid(output).squeeze(0)
                    w_output = sigmoid(w_output).squeeze(0)
                    # predicted = [1 if i>0.5 else 0 for i in output]
                    _labels.append(labels)
                    _preds.append(output)
                    weight_preds.append(w_output)
            # import pdb; pdb.set_trace()
            acc = calACC(_preds, _labels)
            # auc = calROC(_preds, _labels)
            # mcc = calMCC(_preds, _labels)
            b_f1, b_p, b_r, ma_f1, mi_f1 = calF(_preds, _labels)
            wb_f1, wb_p, wb_r, wma_f1, wmi_f1 = calF(weight_preds, _labels)
            print(
            '{}  Epoch: {}, Test w-macro-F1: {:.2f}%, Test F1:{:.2f}%, Test weight-F1:{:.2f}%'.format(key, epoch + 1,
                                                                                                      100 * wma_f1,
                                                                                                      100 * b_f1,
                                                                                                      100 * wb_f1))
            metrics_output_test[key]['acc'].append(acc)
            metrics_output_test[key]['ma_f1'].append(ma_f1)
            metrics_output_test[key]['mi_f1'].append(mi_f1)
            metrics_output_test[key]['b_f1'].append(b_f1)
            metrics_output_test[key]['b_p'].append(b_p)
            metrics_output_test[key]['b_r'].append(b_r)
            metrics_output_test[key]['w_f1'].append(wb_f1)
            f1 = b_f1
            if f1 > best_f1:
                best_f1 = f1
                best_model_weights = model_mlp.state_dict().copy()
                optimizer_model_weights = optimizer.state_dict().copy()
                #     model_mlp.load_state_dict(best_model_weights)

            ckpt_path = '/ckpt/cafa5/'
            ckpt_path = ckpt_path + "{}_1e-4_quickGO_BioLink_sub_esm2_t30_150M_UR50D_{}.pt".format(ctime, key)
            checkpoint = {
                'model_state_dict': best_model_weights,
                'optimizer_state_dict': optimizer_model_weights
            }
            torch.save(checkpoint, ckpt_path)
    # count = 0
    #     with torch.no_grad():
    #         for input_ids in tqdm(pred_dataloader):
    #             embed = input_ids['embed'].cuda()
    #             output = model_mlp(embed).squeeze(0)
    #             output = sigmoid(output)
    #             _pred = np.array(output.cpu())
    #             _pred[_pred>=0.3] = 1
    #             _pred[_pred<0.3] = 0
    #             pred_pos = [i for i, x in enumerate(_pred) if x == 1]
    #             for _pos in pred_pos:
    #                 pre_id = pred_id[count]
    #                 _label = enc.inverse_transform([_pos])
    #                 gene_tag = 'GO:' + str(_label[0])
    #                 if pre_id not in pred_results:  # check if the key is in the dictionary
    #                     pred_results[pre_id] = {'GO': [], 'score': []}  # if not, create a new key-value pair
    #                 pred_results[pre_id]['GO'].append(gene_tag)
    #                 score = output[_pos].item()
    #                 # score = (score-0.1) /0.9
    #                 score = round(score, 3)
    #                 pred_results[pre_id]['score'].append(str(score))
    #             count += 1

    # for key in pred_results.keys():
    # # zip 'GO' and 'score' lists, sort them by 'score', and unzip them
    #     go_score_pair = list(zip(pred_results[key]['GO'], pred_results[key]['score']))
    #     go_score_pair.sort(key=lambda x: x[1], reverse=True)  # Sort by 'score' in descending order
    #     pred_results[key]['GO'], pred_results[key]['score'] = zip(*go_score_pair)  # Unzip the pairs

    # output_results=[]
    # for key in pred_results.keys():
    #     for go, score in zip(pred_results[key]['GO'], pred_results[key]['score']):
    #         sample = str(key)+' '+str(go)+' '+ str(score)
    #         output_results.append(sample)

    # with open(output_path+"/results/weight_orgin_2hidden_sub_esm2_t36_3B_UR50D_BCE_prediction.tsv", 'w') as file_prec:
    #     for i in range(len(output_results)):
    #         file_prec.write("{}\n".format(output_results[i]))

    with open(output_path + "/1e-4_quickGO_BioLink_sub_esm2_t30_150M_UR50D.txt", 'w') as file_prec:
        for key in metrics_output_test.keys():
            for i in range(epoch_num):
                # file_prec.write("Epoch={}; Val Accuracy={}; Val AUC={}; Val MCC={}; Val F1={}\n".format(i+1, metrics_output_valid['acc'][i], metrics_output_valid['auc'][i], metrics_output_valid['mcc'][i], metrics_output_valid['f1'][i]))
                file_prec.write(
                    "{} Epoch={}; Val Accuracy={}; Val Precision={}; Val Recall ={}; Val F1={}; Val macro-F1={}; Val micro-F1={}; Val weight-F1={}\n".
                    format(key, i + 1, metrics_output_test[key]['acc'][i], metrics_output_test[key]['b_p'][i],
                           metrics_output_test[key]['b_r'][i], metrics_output_test[key]['b_f1'][i],
                           metrics_output_test[key]['ma_f1'][i], metrics_output_test[key]['mi_f1'][i],
                           metrics_output_test[key]['w_f1'][i]))