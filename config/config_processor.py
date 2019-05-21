
from typing import NamedTuple
import json

class task_config(NamedTuple):
    """ Config for classification """
    task: str = "agnews"
    mode: str = "train"
    seed: int = 12345
    cfg_data: str = "config/agnews_data.json"
    cfg_model: str = "config/bert_base.json"
    cfg_optim: str = "config/finetune/agnews/optim.json"
    model_file: str = ""
    pretrain_file: str = "../uncased_L-12_H-768_A-12/bert_model.ckpt"
    save_dir: str = "../exp/bert/finetune/agnews"
    comments: str = [] # for comments in json file

class data_config(NamedTuple):
    """ Config for classification dataset """
    vocab_file: str = "../uncased_L-12_H-768_A-12/vocab.txt"
    data_file: dict = {"train": "../agnews/train.csv",
                       "eval": "../agnews/test.csv"}
    max_len: int = 128
    comments: list = [] # for comments in json file

class model_config(NamedTuple):
    """ Configuration for BERT model """
    vocab_size: int = None # Size of Vocabulary
    dim: int = 768 # Dimension of Hidden Layer in Transformer Encoder
    n_layers: int = 12 # Numher of Hidden Layers
    n_heads: int = 12 # Numher of Heads in Multi-Headed Attention Layers
    dim_ff: int = 768*4 # Dimension of Intermediate Layers in Positionwise Feedforward Net
    #activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    p_drop_hidden: float = 0.1 # Probability of Dropout of various Hidden Layers
    p_drop_attn: float = 0.1 # Probability of Dropout of Attention Layers
    max_len: int = 512 # Maximum Length for Positional Embeddings
    n_segments: int = 2 # Number of Sentence Segments
    comments: list = [] # for comments in json file


class optim_config(NamedTuple):
    """ Hyperparameters for optimization """
    seed: int = 3431 # random seed
    batch_size: int = 32
    lr: int = 5e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup: float = 0.1
    save_steps: int = 100 # interval for saving model
    total_steps: int = 100000 # total number of steps to train
    data_parallel: bool = False
    comments: str = "" # for comments in json file


def read_config(config='config/finetune/agnews/train.json'):
    cfg = task_config(**json.load(open(config, "r")))
    cfg_data = data_config(**json.load(open(cfg.cfg_data, "r")))
    cfg_model = model_config(**json.load(open(cfg.cfg_model, "r")))
    cfg_optim = optim_config(**json.load(open(cfg.cfg_optim, "r")))

    return cfg, cfg_data, cfg_model, cfg_optim