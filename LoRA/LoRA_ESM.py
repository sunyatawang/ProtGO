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
        for i in range(len(layer_size_list)-2):
            self.top_layer_list.append(torch.nn.Conv1d(layer_size_list[i], layer_size_list[i+1], kernel_size=1))
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
        for i in range(len(layer_size_list)-2):
            self.top_layer_list.append(torch.nn.Linear(layer_size_list[i], layer_size_list[i+1]))
            self.top_layer_list.append(torch.nn.ReLU())
        self.final_layer = torch.nn.Linear(layer_size_list[-2], layer_size_list[-1])
    def forward(self, x):
        for layer in self.top_layer_list:
            x = layer(x)
        x = self.final_layer(x)
        return x


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
            predictor_type: str = 'protein',  # pair residual sequence
            class_num: int = 2,
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
        if predictor_type == 'pair':
            self.lora_predictor = MLP(embed_dim*2, mlp_hidden_size, 1)
        elif predictor_type == 'residual':
            self.lora_predictor = FC_Net_Conv([embed_dim, mlp_hidden_size, class_num])
        elif predictor_type == 'protein':
            self.lora_predictor = FC_Net_Linear([embed_dim, mlp_hidden_size, class_num])
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
        x_1 = x_1.repeat(1, 1, 1,  x.shape[-1])
        x_2 = x_2.repeat(1, 1, x.shape[-1],  1)
        if pair_method == 'nonsymmetry_concat':
            x = torch.cat((x_1, x_2), dim=1)
        elif pair_method == 'symmetry_concat':
            x = torch.cat((x_1, x_2), dim=1)
            I, J = torch.tril_indices(x.shape[-1], x.shape[-1], -1)
            x[:, :, I, J] = x[:, :, J, I]                # symmetrization
        elif pair_method == 'add':
            x = x_1+x_2
        elif pair_method == 'abs_minus':
            x = torch.abs(x_1-x_2)
        elif pair_method == 'dot':
            x = x_1*x_2
        else:
            sys.exit("{} not implement".format(pair_method))
        return x.permute(0, 2, 3, 1).contiguous()

    def forward_contact(self, seq):
        # print(seq)
        batch_labels, batch_strs, batch_tokens = self.batch_converter ([("x", seq[0])])
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
        batch_labels, batch_strs, batch_tokens = self.batch_converter ([("x", seq[0])])
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
        batch_labels, batch_strs, batch_tokens = self.batch_converter ([("x", seq[0])])
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