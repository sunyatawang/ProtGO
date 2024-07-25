# Setup
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["WANDB_DISABLED"] = "true"
# os.environ["TRANSFORMERS_CACHE"]
# os.environ['TORCH_HOME']

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
from collections import Counter

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
# import wandb
# wandb.init(mode='offline')
from sklearn import preprocessing
from sklearn.metrics import roc_curve,auc,roc_auc_score,matthews_corrcoef,\
    multilabel_confusion_matrix,precision_recall_fscore_support


from graph import Graph, Prediction, propagate
from parser import obo_parser, gt_parser, pred_parser, ia_parser
from evaluation import get_leafs_idx, get_roots_idx, evaluate_prediction

from transformers import AutoTokenizer, AutoModel

# nlp_path = '/BioLinkBERT-large/' # nlp_dim = 1024
nlp_path = '/BiomedNLP-BiomedBERT-base-uncased-abstract/' # nlp_dim = 768
nlp_tokenizer = AutoTokenizer.from_pretrained(nlp_path)
nlp_model = AutoModel.from_pretrained(nlp_path)
nlp_dim = 768
nlp_model.cuda()
nlp_model.eval()


# Select the available device
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    print("Available device:", torch.cuda.current_device())
else:
    print('Available device: CPU')


# Load pre-trained PLMs
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

train_path = "data/cafa5_train.txt" # merge cafa5_train_1 and cafa5_train_2 into a file
test_path = "data/cafa5_test.txt"
output_path = "eval/function_linear"
tax_path = 'data/Original/train_taxonomy.tsv'
obo_path = 'data/Original/go-basic.obo'
ia_path = 'data/Original/IA.txt'

label_space = {
    'biological_process': [],
    'molecular_function': [],
    'cellular_component': []
}

enc = preprocessing.LabelEncoder()
MAXLEN = 2048

#taxonomy inputs
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

#taxonomy encoding
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
    for i, _taxon in enumerate(ground_taxon):
        temp_tax = enctax.transform([_taxon])
        tax_list[i][temp_tax[0]] = 1.0
    for _tax in tax_list:
        tax_fin.append(_tax)
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
    return ontologies, ia_dict


onto, ia_dict = obo_graph(obo_path, ia_path) # OG knowledgebase inputs


def parent(enc, key, label_list):
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
                                    if poss_tags not in label_list:
                                        continue
                                    _pos = enc.transform([poss_tags])
                                    onto_parent[i]['size'] += 1
                                    onto_parent[i]['pos'].extend(_pos)
    return onto_parent


def preprocess_dataset(filepath, max_length=MAXLEN):
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
                            label_space[ns].append(tag)
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
def embed_dataset(sample, taxon, nlp_embed, t_s_nlp, t_s_tax, multimodal_transformer, max_len=None):
    batch_converter = tokenizer.get_batch_converter()
    embedding_list = []
    # weight_list = []
    # batch_size = 512
    # num_batches = len(sample) // batch_size + (1 if len(sample) % batch_size != 0 else 0)
    batch_size = len(sample)
    for i in range(batch_size):
        batch_labels, batch_strs, batch_tokens = batch_converter([("x", sample[i])])
        with torch.no_grad():
            batch_tokens = batch_tokens.cuda()
            results = model(batch_tokens, repr_layers=[num_layers])
            token_representations = results["representations"][num_layers]
            token_representations = token_representations.detach()
            plm_embed = token_representations[0, 1:1 + len(sample[i]), :].clone().detach()
            if nlp_embed != None:
                embedding = clip_align(plm_embed, nlp_embed, t_s_nlp, multimodal_transformer).cuda()
            else:
                embedding = plm_embed
            embedding = torch.mean(embedding, dim=0)
        taxon_t = t_s_tax(taxon.cuda())
        embedding = embedding + 0.1 * taxon_t
        embedding_list.append(embedding)
    # mode = filepath.split("/")[-1].split("_src")[0]
    # embedding_path = "{}/esm2_t48_15B_UR50D_{}.out".format(embed_path, mode)
    # with open(embedding_path, 'wb') as embhandle:
    #     pickle.dump(inputs_embedding, embhandle)
    embedding_batch = torch.stack(embedding_list, dim=0)
    return embedding_batch

# Organize sequences embeddings
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

# GNN for GO graph
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

#Classifier
class CustomModel(nn.Module):
    def __init__(self, input_size, output_size, adj):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 2*input_size)  # original 10240
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2*input_size, output_size)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)
        self.gc1 = GraphConvolution(output_size, output_size)
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
        x = self.gc1(x, self.adj)
        return x

#evaluation metrics
def calACC(hypo, gold):
    _correct = 0
    for i in range(0,len(hypo)):
        _hypo = np.array(hypo[i].cpu())
        _gold = np.array(gold[i].cpu())
        _hypo[_hypo>=0.5] = 1
        _hypo[_hypo<0.5] = 0
        correct = (_hypo == _gold).sum()
        total = _gold.shape[0]
        correct = correct/total
        _correct += correct
    total = len(gold)
    return _correct/total

def calROC(hypo, gold):
    roc_auc = 0
    for i in range(0,len(hypo)):
        _hypo = np.array(hypo[i].cpu())
        _gold = np.array(gold[i].cpu())
        if np.max(_gold)== 0 or np.max(_hypo)==0:
            continue
        fpr, tpr, _ = roc_curve(_gold, _hypo)
        sample_auc = auc(fpr, tpr)
        roc_auc += sample_auc
    total = len(hypo)
    return roc_auc/ total

def calMCC(hypo, gold):
    mcc = 0
    for i in range(0,len(hypo)):
        _hypo = np.array(hypo[i].cpu())
        _gold = np.array(gold[i].cpu())
        _hypo[_hypo>=0.5] = 1
        _hypo[_hypo<0.5] = 0
        if np.max(_hypo) == 0 or np.max(_gold) == 0:
            continue
        mcc_sample = matthews_corrcoef(_gold, _hypo)
        mcc += mcc_sample
    total = len(hypo)
    return mcc/ total

def calF(hypo, gold, return_all=False):
    b_f1 = ma_f1 = mi_f1 = 0
    b_p = b_r= 0
    for i in range(0,len(hypo)):
        _hypo = np.array(hypo[i].cpu())
        _gold = np.array(gold[i].cpu())
        _hypo[_hypo>=0.5] = 1
        _hypo[_hypo<0.5] = 0
        p,r,f,_ = precision_recall_fscore_support(_gold, _hypo, average='binary', zero_division=1)
        b_f1 += f
        b_p += p
        b_r += r
        p,r,f,_ = precision_recall_fscore_support(_gold, _hypo, average='macro', zero_division=1)
        ma_f1 += f
        p,r,f,_ = precision_recall_fscore_support(_gold, _hypo, average='micro', zero_division=1)
        mi_f1 += f
    total = len(hypo)
    return b_f1/total, b_p/total, b_r/total, ma_f1/total, mi_f1/total

def evaluate_annotations(gold, hypo):
    """
    Computes Fmax
    """
    total = 0
    p = 0.0
    r = 0.0
    p_total= 0
    prec_list=[0]
    rec_list=[0]
    for i in range(0,len(hypo)):
        _hypo = np.array(hypo[i].cpu())
        _gold = np.array(gold[i].cpu())
        real_num = np.sum(_gold == 1)
        _hypo[_hypo>=0.5] = 1
        _hypo[_hypo<0.5] = 0
        pred_num = np.sum(_hypo == 1)
        if real_num == 0 or pred_num == 0:
            continue
        tpn = np.sum((_gold == 1) & (_hypo == 1))
        fpn = np.sum((_gold == 0) & (_hypo == 1))
        fnn = np.sum((_gold == 1) & (_hypo == 0))
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        if pred_num > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
        if i % 100 == 0 or i == len(hypo)-1:
            if p_total > 0 and total > 0:
                prec_list.append(p/ p_total)
                rec_list.append(r/ total)
    if total != 0:
        r /= total
    if p_total > 0:
        p /= p_total
    f = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
    prec_list = np.array(prec_list)
    rec_list = np.array(rec_list)
    sorted_index = np.argsort(rec_list)
    rec_list = rec_list[sorted_index]
    prec_list = prec_list[sorted_index]
    aupr = np.trapz(prec_list, rec_list)
    return f, p, r, aupr


#pre-trained NLP models embedding
def nlp_embedding(nlp_model, label_list, key, top_list):
    print(f"Start nlp model {nlp_path}")
    match_embedding = [None for _ in range(len(label_list))]
    for index, multi_tag in enumerate(label_list):
        if multi_tag == []:
            continue
        context = ''
        for _tag in multi_tag:
            if _tag not in top_list:
                continue
            for ont in onto:
                if ont.namespace != key:
                    continue
                _tag = 'GO:' + _tag
                if _tag in ont.terms_dict.keys():
                    # tag_context = ont.terms_dict[_tag]['def'] #if count(_tag) <=10
                    # tag_contents = re.findall(r'"(.*?)"', tag_context)
                    # if context == '':
                    #     context = context + tag_contents[0]
                    # else:
                    #     context = context + ' ' + tag_contents[0]
                    tag_context = ont.terms_dict[_tag]['name']
                    context = context + tag_context + ' '
        seq_len = 512
        max_len = MAXLEN // 2
        if len(context) > max_len:
            context = context[:max_len]
        num_seqs = len(context) // seq_len + (1 if len(context) % seq_len != 0 else 0)
        last_embed = []
        for i in range(num_seqs):
            start_index = i * seq_len
            end_index = min((i + 1) * seq_len, len(context))
            context_sample = context[start_index:end_index]
            inputs = nlp_tokenizer(context_sample, return_tensors="pt")
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = nlp_model(**inputs)
            last_hidden_states = outputs.last_hidden_state.squeeze(0).detach()
            last_embed.append(last_hidden_states)
        embed = torch.cat(last_embed, dim=0)
        match_embedding[index] = embed
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

#Text-protein alignments
class MultimodalTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers):
        super(MultimodalTransformer, self).__init__()
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.transformer_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layers, num_layers=num_layers)
        self.text_pos_encoder = PositionalEncoding(MAXLEN, hidden_dim)
        self.protein_pos_encoder = PositionalEncoding(MAXLEN, hidden_dim)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, text_inputs, protein_inputs):
        text_features = self.text_pos_encoder(text_inputs)
        protein_features = self.protein_pos_encoder(protein_inputs)
        text_features = self.dropout(text_features)
        combined_features = torch.cat((text_features, protein_features), dim=0)
        attn_output, _ = self.multihead_attn(combined_features, combined_features, combined_features)
        transformer_output = self.transformer_encoder(attn_output)
        return transformer_output

def clip_align(plm_embed, nlp_embed, t_s_nlp, multimodal_transformer):
    nlp_project = t_s_nlp(nlp_embed.cuda())
    plm_embed_normal = nn.functional.normalize(plm_embed, p=2, dim=1)
    nlp_project_normal = nn.functional.normalize(nlp_project, p=2, dim=1)
    similarity_matrix = plm_embed_normal @ nlp_project_normal.t()
    mean_similarities = similarity_matrix.mean(dim=0)
    indices_to_remain = torch.where(mean_similarities > 0)[0]
    nlp_filtered = torch.index_select(nlp_project, dim=0, index=indices_to_remain)
    output_embed = multimodal_transformer(nlp_filtered.cuda(), plm_embed.cuda())
    return output_embed


if __name__ == "__main__":
    ctime = datetime.now().strftime("%Y%m%d%H%M%S")
    print('Start running date:{}'.format(ctime))
    train_id, training_sequences, training_labels = preprocess_dataset(train_path)
    test_id, test_sequences, test_labels = preprocess_dataset(test_path)

    extract_id, extract_tax = extract_know(tax_path)
    train_tax = embed_know(train_id, extract_id, extract_tax)
    test_tax = embed_know(test_id, extract_id, extract_tax)

    pred_results = {}
    metrics_output_test = {}
    for key in label_space.keys():
        label_tops = Counter(label_space[key])
        top_labels = [label for label in set(label_space[key]) if label_tops[label] > 21]  # 0.015%
        print('Top label numbers:{}'.format(len(top_labels))) # Pruning labels
        label_list = top_labels

        train_nlp = nlp_embedding(nlp_model, training_labels[key], key, label_list)
        test_nlp = []
        for x in test_id:
            test_nlp.append(None)

        labspace = enc.fit_transform(label_list)
        onto_parent = parent(enc, key, label_list)
        x = 0
        label_num = len(enc.classes_)
        for label in training_labels[key]:
            filtered_label = [item for item in label if item in label_list]
            if len(filtered_label) == 0:
                training_labels[key][x] = [0] * label_num
            else:
                temp_labels = enc.transform(filtered_label)
                training_labels[key][x] = [1 if i in temp_labels else 0 for i in range(0, label_num)]
            x += 1
        x = 0
        for label in test_labels[key]:
            filtered_label = [item for item in label if item in label_list]
            if len(filtered_label) == 0:
                test_labels[key][x] = [0] * label_num
            else:
                temp_labels = enc.transform(filtered_label)
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
        test_dataset = StabilitylandscapeDataset(test_sequences, test_labels[key])

        train_dataloader = DataLoader(training_dataset, batch_size=16, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model_mlp = CustomModel(embed_dim, label_num, adj_matrix).cuda()
        nlp_projection = TransformSize(nlp_dim, embed_dim).cuda()
        tax_dim = train_tax[0].size(0)
        model_tax = TransformSize(tax_dim, embed_dim).cuda()
        multimodal_transformer = MultimodalTransformer(
            hidden_dim=embed_dim,
            num_heads=8,
            num_layers=6,
        ).cuda()

        criterion = nn.BCELoss()
        e = math.e
        optimizer = torch.optim.Adam(model_mlp.parameters(), lr=4e-5)  # 4e-5 150M 1e-5 3B
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.6)
        epoch_num = 30
        if key not in metrics_output_test:
            metrics_output_test[key] = {
                'acc':[],
                'ma_f1':[],
                'mi_f1':[],
                'b_p':[],
                'b_r':[],
                'b_f1':[],
                'w_f1':[],
                'p':[],
                'r':[],
                'f1':[],
                'mcc':[],
                'aupr':[],
                'roc':[]
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
                embed_fusion = embed_dataset(embed, train_tax[i], train_nlp[i], nlp_projection, model_tax,
                                             multimodal_transformer).cuda()
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
            scheduler.step()

            _labels = []
            _preds = []
            weight_preds = []
            with torch.no_grad():
                for i, (input_ids) in tqdm(enumerate(test_dataloader)):
                    embed = input_ids['embed']
                    embed_fusion = embed_dataset(embed, test_tax[i], test_nlp[i], nlp_projection, model_tax,
                                                 multimodal_transformer).cuda()
                    labels = input_ids['labels'].squeeze(0)
                    output = model_mlp(embed_fusion)
                    w_output = output * ia_list
                    output = sigmoid(output).squeeze(0)
                    w_output = sigmoid(w_output).squeeze(0)
                    _labels.append(labels.cpu())
                    _preds.append(output.cpu())
                    weight_preds.append(w_output.cpu())
            acc = calACC(_preds, _labels)
            roc = calROC(_preds, _labels)
            mcc = calMCC(_preds, _labels)
            b_f1, b_p, b_r, ma_f1, mi_f1 = calF(_preds, _labels)
            wb_f1, wb_p, wb_r, wma_f1, wmi_f1 = calF(weight_preds, _labels)
            f, p, r, aupr = evaluate_annotations(_labels, _preds)
            print(
            '{}  Epoch: {}, Test w-macro-F1: {:.2f}%, Test F1:{:.2f}%, Test avg-F1:{:.2f}%, Test weight-F1:{:.2f}%, Test AUPR:{:.2f}%'.
            format(key, epoch + 1, 100 * wma_f1, 100 * b_f1, 100 * f, 100 * wb_f1, 100 * aupr))
            metrics_output_test[key]['acc'].append(acc)
            metrics_output_test[key]['ma_f1'].append(ma_f1)
            metrics_output_test[key]['mi_f1'].append(mi_f1)
            metrics_output_test[key]['b_f1'].append(b_f1)
            metrics_output_test[key]['b_p'].append(b_p)
            metrics_output_test[key]['b_r'].append(b_r)
            metrics_output_test[key]['w_f1'].append(wb_f1)
            metrics_output_test[key]['p'].append(p)
            metrics_output_test[key]['r'].append(r)
            metrics_output_test[key]['f1'].append(f)
            metrics_output_test[key]['mcc'].append(mcc)
            metrics_output_test[key]['roc'].append(roc)
            metrics_output_test[key]['aupr'].append(aupr)
            f1 = f
            if f1 > best_f1:
                best_f1 = f1
                best_model_weights = model_mlp.state_dict().copy()
                optimizer_model_weights = optimizer.state_dict().copy()
                #     model_mlp.load_state_dict(best_model_weights)

            ckpt_path = '/ckpt/cafa5/linear/'
            ckpt_path = ckpt_path + "{}_ProtGO_BioMed_esm2_t30_150M_UR50D_{}.pt".format(ctime, key)
            checkpoint = {
                'model_state_dict': best_model_weights,
                'optimizer_state_dict': optimizer_model_weights
            }
            torch.save(checkpoint, ckpt_path)


    with open(output_path + "/ProtGO_BioMed_esm2_t30_150M_UR50D.txt", 'w') as file_prec:
        for key in metrics_output_test.keys():
            for i in range(epoch_num):
                file_prec.write("{} Epoch={}; Val Accuracy={}; Val Precision={}; Val Recall ={}; Val F1={}; Val macro-F1={}; Val micro-F1={}; Val weight-F1={}; Val avg-precision={}; Val avg-recall={}; Val avg-F1={}; Val AUC={}; Val MCC={}; Val AUPR={}\n".
                format(key, i+1, metrics_output_test[key]['acc'][i], metrics_output_test[key]['b_p'][i], metrics_output_test[key]['b_r'][i], metrics_output_test[key]['b_f1'][i], metrics_output_test[key]['ma_f1'][i], metrics_output_test[key]['mi_f1'][i], metrics_output_test[key]['w_f1'][i], metrics_output_test[key]['p'][i], metrics_output_test[key]['r'][i], metrics_output_test[key]['f1'][i], metrics_output_test[key]['roc'][i], metrics_output_test[key]['mcc'][i], metrics_output_test[key]['aupr'][i]))