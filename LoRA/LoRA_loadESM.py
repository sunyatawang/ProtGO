from LoRA_ESM import my_ESM2
import esm
import torch
import re
from esm.model.esm2 import ESM2
import os
os.environ['TORCH_HOME']


# print(model_data.keys())
# print(regression_data.keys())
# print(type(model_data['model']))
# print(type(regression_data['model']))
# print(model_data['model'].keys())
# print(regression_data['model'].keys())
# model_name = "esm2_t48_15B_UR50D"
# num_layers = 48
# model_name = "esm2_t36_3B_UR50D"
# num_layers = 36
# model_name = "esm2_t33_650M_UR50D"
# num_layers = 33
# model_name = "esm2_t30_150M_UR50D"
# num_layers = 30

def upgrade_state_dict(state_dict):
    """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
    prefixes = ["encoder.sentence_encoder.", "encoder."]
    pattern = re.compile("^" + "|".join(prefixes))
    state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
    return state_dict

def load_esm_model(model_name, lora_r, lora_alpha, use_checkpoint=False, predictor_type='pair', class_num=2, tax_dim = 1, adj_matrix=torch.zeros(1,1)):
    model_data, regression_data = esm.pretrained._download_model_and_regression_data(model_name)
    cfg = model_data["cfg"]["model"]
    state_dict = model_data["model"]
    model_state = upgrade_state_dict(state_dict)
    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    my_model = my_ESM2(
        num_layers=cfg.encoder_layers,
        embed_dim=cfg.encoder_embed_dim,
        attention_heads=cfg.encoder_attention_heads,
        alphabet=alphabet,
        token_dropout=cfg.token_dropout,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        # use_checkpoint=use_checkpoint,
        tax_dim = tax_dim,
        predictor_type=predictor_type,
        class_num=class_num,
        adj_matrix=adj_matrix,
    )
    my_model.load_state_dict(model_state, strict=False)
    trainable_size = 0
    for name, param in my_model.named_parameters():
        if param.requires_grad and not ('lora_' in name):
            param.requires_grad = False
        else:
            trainable_size += param.numel()
 
    print(trainable_size)
    return my_model, alphabet, trainable_size
