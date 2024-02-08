# ProtGO: Multi-Modal Protein Function Prediction using Large-Scale Protein Language Models

## Requirements
```shell
pip install fair-esm
pip install git+https://github.com/facebookresearch/esm.git
python -m pip install ankh
pip install SwissArmyTransformer
```

## Usage
* Linear Probing with MLP:
```python
python model/ProtGO_linear.py
```

* Fine-tuning with LoRA:
```python
python model/ProtGO_LoRA.py
```