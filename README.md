# ProtGO: Multi-Modal Protein Function Prediction using Large-Scale Protein Language Models

## Requirements
```shell
pip install fair-esm
pip install git+https://github.com/facebookresearch/esm.git
python -m pip install ankh
pip install SwissArmyTransformer
```

## Datasets
* Train set:
/main/data/cafa5_train_1.txt + /main/data/cafa5_train_2.txt
* Test set:
/main/data/cafa5_test.txt

## Usage
* Linear Probing with MLP:
```python
python model/ProtGO_linear.py
```

* Fine-tuning with LoRA:
```python
python model/ProtGO_LoRA.py
```