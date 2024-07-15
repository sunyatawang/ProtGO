# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
from typing import Union
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import esm
import math
from esm.modules import ContactPredictionHead, ESM1bLayerNorm, RobertaLMHead, TransformerLayer
from LoRA_modules import my_TransformerLayer


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        torch.nn.init.kaiming_normal_(self.linear1.weight)
        self.linear1.bias.data.fill_(0.01)
        self.linear2 = nn.Linear(hidden_size, output_size)
        torch.nn.init.kaiming_normal_(self.linear2.weight)
        self.linear2.bias.data.fill_(0.01)
        self.final_activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.linear2(x)
        return self.final_activation(x)


class FC_Net_Conv(torch.nn.Module):
    def __init__(self, layer_size_list):
        super(FC_Net_Conv, self).__init__()
        self.top_layer_list = torch.nn.ModuleList()
        for i in range(len(layer_size_list) - 2):
            self.top_layer_list.append(torch.nn.Conv1d(layer_size_list[i], layer_size_list[i + 1], kernel_size=1))
            self.top_layer_list.append(torch.nn.ReLU())
        self.final_layer = torch.nn.Conv1d(layer_size_list[-2], layer_size_list[-1], kernel_size=1)

    def forward(self, x):
        for layer in self.top_layer_list:
            x = layer(x)
        x = self.final_layer(x)
        return x


class FC_Net_Linear(torch.nn.Module):
    def __init__(self, layer_size_list):
        super(FC_Net_Linear, self).__init__()
        self.top_layer_list = torch.nn.ModuleList()
        for i in range(len(layer_size_list) - 2):
            self.top_layer_list.append(torch.nn.Linear(layer_size_list[i], layer_size_list[i + 1]))
            self.top_layer_list.append(torch.nn.ReLU())
        self.final_layer = torch.nn.Linear(layer_size_list[-2], layer_size_list[-1])

    def forward(self, x):
        for layer in self.top_layer_list:
            x = layer(x)
        x = self.final_layer(x)
        return x


class TransformSize(nn.Module):
    def __init__(self, input_size, output_size):
        super(TransformSize, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


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


class quick_GO_GCN(nn.Module):
    def __init__(self, input_size, output_size, adj):
        # import pdb; pdb.set_trace()
        super(quick_GO_GCN, self).__init__()
        # self.fc = nn.Linear(tax_size, input_size)
        self.fc1 = nn.Linear(input_size, 5120)
        self._init_weights(self.fc1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(5120, output_size)
        self._init_weights(self.fc2)
        self.gc1 = GraphConvolution(output_size, output_size)
        # self.relu2 = nn.ReLU()
        # self.gc2 = GraphConvolution(5120, output_size)
        # self.final_activation = torch.nn.Sigmoid()
        self.adj = adj
        self.final_activation = torch.nn.Sigmoid()

    def _init_weights(self, module):
        std_w = math.sqrt(2. / module.weight.data.size()[1])
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std_w)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std_w)
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
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, text_inputs, protein_inputs):
        text_features = self.text_pos_encoder(text_inputs)
        protein_features = self.protein_pos_encoder(protein_inputs)
        text_features = self.dropout(text_features)
        # import pdb; pdb.set_trace()
        combined_features = torch.cat((text_features, protein_features), dim=0)
        attn_output, _ = self.multihead_attn(combined_features, combined_features, combined_features)
        transformer_output = self.transformer_encoder(attn_output)
        return transformer_output


class freeze_nlp(nn.Module):
    def clip_align(plm_embed, nlp_embed):
        nlp_dim = nlp_embed.size(-1)
        embed_dim = plm_embed.size(-1)
        plm_embed = plm_embed.squeeze(0)
        nlp_projection = TransformSize(nlp_dim, embed_dim).cuda()
        nlp_project = nlp_projection(nlp_embed)
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
        mean_similarities = similarity_matrix.mean(dim=0)
        indices_to_remain = torch.where(mean_similarities > 0)[0]
        nlp_filtered = torch.index_select(nlp_project, dim=0, index=indices_to_remain)

        output_embed = multimodal_transformer(nlp_filtered, plm_embed)
        return output_embed.unsqueeze(0)


class ProtGO_GCN(nn.Module):
    def __init__(self, input_size, output_size, adj):
        # import pdb; pdb.set_trace()
        super(ProtGO_GCN, self).__init__()
        # self.fc = nn.Linear(tax_size, input_size)
        self.fc1 = nn.Linear(input_size, 2 * input_size)
        self._init_weights(self.fc1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2 * input_size, output_size)
        self._init_weights(self.fc2)
        self.gc1 = GraphConvolution(output_size, output_size)
        # self.relu2 = nn.ReLU()
        # self.gc2 = GraphConvolution(5120, output_size)
        # self.final_activation = torch.nn.Sigmoid()
        self.adj = adj
        self.final_activation = torch.nn.Sigmoid()

    def _init_weights(self, module):
        std_w = math.sqrt(2. / module.weight.data.size()[1])
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std_w)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std_w)
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


class CustomModel(nn.Module):
    def __init__(self, input_size, output_size):
        # import pdb; pdb.set_trace()
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self._init_weights(self.fc1)
        self.final_activation = torch.nn.Sigmoid()

    def _init_weights(self, module):
        std_w = math.sqrt(2. / module.weight.data.size()[1])
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std_w)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std_w)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.fc1(x)
        return self.final_activation(x)


class my_ESM2(nn.Module):
    def __init__(
            self,
            num_layers: int = 33,
            embed_dim: int = 1280,
            attention_heads: int = 20,
            alphabet: Union[esm.data.Alphabet, str] = "ESM-1b",
            token_dropout: bool = True,
            lora_r: int = 0,
            lora_alpha: int = 1,
            lora_dropout_ratio: float = 0.,
            mlp_hidden_size: int = 128,
            use_checkpoint: bool = False,
            predictor_type: str = 'multi',  # pair residual sequence
            class_num: int = 2,
            tax_dim: int = 1,
            adj_matrix = torch.zeros(1,1),
    ):
    super().__init__()
    self.num_layers = num_layers
    self.embed_dim = embed_dim
    self.attention_heads = attention_heads
    if not isinstance(alphabet, esm.data.Alphabet):
        alphabet = esm.data.Alphabet.from_architecture(alphabet)
    self.alphabet = alphabet
    self.batch_converter = self.alphabet.get_batch_converter()
    self.alphabet_size = len(alphabet)
    self.padding_idx = alphabet.padding_idx
    self.mask_idx = alphabet.mask_idx
    self.cls_idx = alphabet.cls_idx
    self.eos_idx = alphabet.eos_idx
    self.prepend_bos = alphabet.prepend_bos
    self.append_eos = alphabet.append_eos
    self.token_dropout = token_dropout
    self.adj_matrix = adj_matrix
    self.tax_dim = tax_dim
    if predictor_type == 'pair':
        self.lora_predictor = MLP(embed_dim * 2, mlp_hidden_size, 1)
    elif predictor_type == 'residual':
        self.lora_predictor = FC_Net_Conv([embed_dim, mlp_hidden_size, class_num])
    elif predictor_type == 'protein':
        self.lora_predictor = FC_Net_Linear([embed_dim, mlp_hidden_size, class_num])
    elif predictor_type == 'multi':
        self.lora_predictor = CustomModel(embed_dim, class_num)
    elif predictor_type == 'quickGO':
        self.TransformSize = TransformSize(tax_dim, embed_dim)
        self.lora_predictor = quick_GO_GCN(embed_dim, class_num, adj_matrix)
    elif predictor_type == 'ProtGO':
        self.TransformSize = TransformSize(tax_dim, embed_dim)
        self.lora_predictor = ProtGO_GCN(embed_dim, class_num, adj_matrix)
    else:
        raise NotImplementedError("{} is not implemented".format(predictor_type))
    self.use_checkpoint = use_checkpoint
    self._init_submodules(lora_r, lora_alpha, lora_dropout_ratio)


def _init_submodules(self, lora_r, lora_alpha, lora_dropout_ratio):
    self.embed_scale = 1
    self.embed_tokens = nn.Embedding(
        self.alphabet_size,
        self.embed_dim,
        padding_idx=self.padding_idx,
    )

    self.layers = nn.ModuleList(
        [
            my_TransformerLayer(
                self.embed_dim,
                4 * self.embed_dim,
                self.attention_heads,
                add_bias_kv=False,
                use_esm1b_layer_norm=True,
                use_rotary_embeddings=True,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout_ratio=lora_dropout_ratio,
            )
            for _ in range(self.num_layers)
        ]
    )

    self.contact_head = ContactPredictionHead(
        self.num_layers * self.attention_heads,
        self.prepend_bos,
        self.append_eos,
        eos_idx=self.eos_idx,
    )
    self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)

    self.lm_head = RobertaLMHead(
        embed_dim=self.embed_dim,
        output_dim=self.alphabet_size,
        weight=self.embed_tokens.weight,
    )


def forward(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False):
    if return_contacts:
        need_head_weights = True

    assert tokens.ndim == 2
    padding_mask = tokens.eq(self.padding_idx)  # B, T

    x = self.embed_scale * self.embed_tokens(tokens)

    if self.token_dropout:
        x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
        # x: B x T x C
        mask_ratio_train = 0.15 * 0.8
        src_lengths = (~padding_mask).sum(-1)
        mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
        x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

    if padding_mask is not None:
        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

    repr_layers = set(repr_layers)
    hidden_representations = {}
    if 0 in repr_layers:
        hidden_representations[0] = x

    if need_head_weights:
        attn_weights = []

    # (B, T, E) => (T, B, E)
    x = x.transpose(0, 1)

    if not padding_mask.any():
        padding_mask = None

    for layer_idx, layer in enumerate(self.layers):
        # if self.use_checkpoint:
        #     x, attn = checkpoint(lambda t: layer(t, self_attn_padding_mask=padding_mask, need_head_weights=need_head_weights), x)
        # else:
        x, attn = layer(
            x,
            self_attn_padding_mask=padding_mask,
            need_head_weights=need_head_weights,
        )
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x.transpose(0, 1)
        if need_head_weights:
            # (H, B, T, T) => (B, H, T, T)
            attn_weights.append(attn.transpose(1, 0))

    x = self.emb_layer_norm_after(x)
    x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

    # last hidden representation should have layer norm applied
    if (layer_idx + 1) in repr_layers:
        hidden_representations[layer_idx + 1] = x
    x = self.lm_head(x)

    result = {"logits": x, "representations": hidden_representations}
    if need_head_weights:
        # attentions: B x L x H x T x T
        attentions = torch.stack(attn_weights, 1)
        if padding_mask is not None:
            attention_mask = 1 - padding_mask.type_as(attentions)
            attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
            attentions = attentions * attention_mask[:, None, None, :, :]
        result["attentions"] = attentions
        if return_contacts:
            contacts = self.contact_head(tokens, attentions)
            result["contacts"] = contacts

    return result


def predict_contacts(self, tokens):
    return self(tokens, return_contacts=True)["contacts"]


def outer_concat(self, x, pair_method='symmetry_concat'):
    # breakpoint()
    x = x.permute(0, 2, 1)
    x_1 = x[:, :, :, None]
    x_2 = x[:, :, None, :]
    x_1 = x_1.repeat(1, 1, 1, x.shape[-1])
    x_2 = x_2.repeat(1, 1, x.shape[-1], 1)
    if pair_method == 'nonsymmetry_concat':
        x = torch.cat((x_1, x_2), dim=1)
    elif pair_method == 'symmetry_concat':
        x = torch.cat((x_1, x_2), dim=1)
        I, J = torch.tril_indices(x.shape[-1], x.shape[-1], -1)
        x[:, :, I, J] = x[:, :, J, I]  # symmetrization
    elif pair_method == 'add':
        x = x_1 + x_2
    elif pair_method == 'abs_minus':
        x = torch.abs(x_1 - x_2)
    elif pair_method == 'dot':
        x = x_1 * x_2
    else:
        sys.exit("{} not implement".format(pair_method))
    return x.permute(0, 2, 3, 1).contiguous()


def forward_contact(self, seq):
    # print(seq)
    batch_labels, batch_strs, batch_tokens = self.batch_converter([("x", seq[0])])
    batch_tokens = batch_tokens.to(self.lora_predictor.linear1.weight.device)
    # print(self.lora_mlp.linear1.weight.device)
    results = self.forward(batch_tokens, repr_layers=[self.num_layers])
    token_representations = results["representations"][self.num_layers][:, 1:-1, :]
    # print(token_representations.shape)
    x = self.outer_concat(token_representations)
    # print(x)
    # print(x.shape)
    # print(f"x.grad_fn: {x.grad_fn}")
    # print(x.shape)
    if self.training and self.use_checkpoint:
        print("use_checkpoint")
        y = checkpoint(self.lora_predictor, x)
    else:
        y = self.lora_predictor(x)
    # y = self.lora_predictor(x)
    # print(y.shape)
    return y


def forward_residual(self, seq):
    batch_labels, batch_strs, batch_tokens = self.batch_converter([("x", seq[0])])
    batch_tokens = batch_tokens.to(self.lora_predictor.final_layer.weight.device)
    # print(self.lora_mlp.linear1.weight.device)
    results = self.forward(batch_tokens, repr_layers=[self.num_layers])
    x = results["representations"][self.num_layers][:, 1:-1, :]
    x = torch.permute(x, [0, 2, 1])
    # print(x.shape)
    if self.training and self.use_checkpoint:
        y = checkpoint(self.lora_predictor, x)
    else:
        y = self.lora_predictor(x)
    # print(y.shape)
    return y


def forward_protein(self, seq):
    batch_labels, batch_strs, batch_tokens = self.batch_converter([("x", seq[0])])
    batch_tokens = batch_tokens.to(self.lora_predictor.final_layer.weight.device)
    # print(self.lora_mlp.linear1.weight.device)
    results = self.forward(batch_tokens, repr_layers=[self.num_layers])
    x = results["representations"][self.num_layers][:, 1:-1, :]
    # print(x.shape)
    x = torch.mean(x, dim=-2)
    # print(x.shape)
    if self.training and self.use_checkpoint:
        y = checkpoint(self.lora_predictor, x)
    else:
        y = self.lora_predictor(x)
    # print(y.shape)
    return y


def forward_long_protein(self, seq):
    embedding_list = []
    weight_list = []
    batch_size = 512
    seq = seq[0]
    # print(len(seq))
    num_batches = len(seq) // batch_size + (1 if len(seq) % batch_size != 0 else 0)
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(seq))
        batch_sample = seq[start_index:end_index]
        batch_labels, batch_strs, batch_tokens = self.batch_converter([("x", batch_sample)])
        batch_tokens = batch_tokens.to(self.lora_predictor.fc1.weight.device)
        # batch_tokens = batch_tokens.to(self.lora_predictor.fc1.weight.device).cpu()
        # print(self.lora_mlp.linear1.weight.device)
        results = self.forward(batch_tokens, repr_layers=[self.num_layers])
        x = results["representations"][self.num_layers][:, 1:-1, :]
        # print(x.shape)
        embedding = torch.mean(x, dim=-2)
        embedding_list.append(embedding)
        weight = len(batch_sample)
        weight_list.append(weight)
        # print(x.shape)
    numer = weight = 0
    for i in range(len(weight_list)):
        numer += embedding_list[i] * weight_list[i]
        weight += weight_list[i]
    embedding = numer / weight
    if self.training and self.use_checkpoint:
        y = checkpoint(self.lora_predictor, embedding)
    else:
        y = self.lora_predictor(embedding)
    # print(y.shape)
    # import pdb; pdb.set_trace()
    return y


def forward_quickgo_protein(self, seq, taxon):
    embedding_list = []
    weight_list = []
    batch_size = 512
    seq = seq[0]
    # print(len(seq))
    num_batches = len(seq) // batch_size + (1 if len(seq) % batch_size != 0 else 0)
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(seq))
        batch_sample = seq[start_index:end_index]
        batch_labels, batch_strs, batch_tokens = self.batch_converter([("x", batch_sample)])
        batch_tokens = batch_tokens.to(self.lora_predictor.gc1.weight.device)
        # batch_tokens = batch_tokens.to(self.lora_predictor.fc1.weight.device).cpu()
        # print(self.lora_mlp.linear1.weight.device)
        results = self.forward(batch_tokens, repr_layers=[self.num_layers])
        x = results["representations"][self.num_layers][:, 1:-1, :]
        # print(x.shape)
        embedding = torch.mean(x, dim=-2)
        embedding_list.append(embedding)
        weight = len(batch_sample)
        weight_list.append(weight)
        # print(x.shape)
    numer = weight = 0
    for i in range(len(weight_list)):
        numer += embedding_list[i] * weight_list[i]
        weight += weight_list[i]
    embedding = numer / weight
    taxon.cuda()
    trans_taxon = self.TransformSize(taxon)
    embedding = embedding + 0.1 * trans_taxon
    if self.training and self.use_checkpoint:
        y = checkpoint(self.lora_predictor, embedding)
    else:
        y = self.lora_predictor(embedding)
    # print(y.shape)
    # import pdb; pdb.set_trace()
    return y


def forward_protgo_protein(self, seq, taxon, nlp_embed):
    embedding_list = []
    weight_list = []
    batch_size = 512
    seq = seq[0]
    # print(len(seq))
    num_batches = len(seq) // batch_size + (1 if len(seq) % batch_size != 0 else 0)
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(seq))
        batch_sample = seq[start_index:end_index]
        batch_labels, batch_strs, batch_tokens = self.batch_converter([("x", batch_sample)])
        batch_tokens = batch_tokens.to(self.lora_predictor.gc1.weight.device)
        # batch_tokens = batch_tokens.to(self.lora_predictor.fc1.weight.device).cpu()
        # print(self.lora_mlp.linear1.weight.device)
        results = self.forward(batch_tokens, repr_layers=[self.num_layers])
        x = results["representations"][self.num_layers][:, 1:-1, :]
        # print(x.shape)
        if nlp_embed != None:
            embedding = freeze_nlp.clip_align(x, nlp_embed)
        else:
            embedding = x
        embedding = torch.mean(x, dim=-2)
        embedding_list.append(embedding)
        weight = len(batch_sample)
        weight_list.append(weight)
        # print(x.shape)
    numer = weight = 0
    for i in range(len(weight_list)):
        numer += embedding_list[i] * weight_list[i]
        weight += weight_list[i]
    embedding = numer / weight
    taxon.cuda()
    trans_taxon = self.TransformSize(taxon)
    embedding = embedding + 0.1 * trans_taxon
    if self.training and self.use_checkpoint:
        y = checkpoint(self.lora_predictor, embedding)
    else:
        y = self.lora_predictor(embedding)
    # print(y.shape)
    return y