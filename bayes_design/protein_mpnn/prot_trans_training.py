# TODO: verify environment will provide necessary packages
# TODO: add correct path to os.chdir()

def main():
    #bayes_design specific imports
    from dataset import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset
    from torch.utils.data import DataLoader

    #import rest of prot_trans training dependencies
    import os.path
    
    # Not necessary to move directories yet
    # os.chdir("set a path here")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
    from torch.utils.data import DataLoader

    import re
    import numpy as np
    import pandas as pd
    import copy

    import transformers
    from transformers.modeling_outputs import TokenClassifierOutput
    from transformers.models.t5.modeling_t5 import T5Config, T5PreTrainedModel, T5Stack
    from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
    from transformers import T5EncoderModel, T5Tokenizer
    from transformers import TrainingArguments, Trainer, set_seed

    #DataCollator
    from transformers.data.data_collator import DataCollatorMixin
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.utils import PaddingStrategy

    import random
    import warnings
    from collections.abc import Mapping
    from dataclasses import dataclass
    from random import randint
    from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

    from evaluate import load

    from tqdm import tqdm
    import random

    from scipy import stats
    from sklearn.metrics import accuracy_score

    import matplotlib.pyplot as plt

    # Set environment variables to run Deepspeed from a notebook
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "9993"  # modify if RuntimeError: Address already in use
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    # prepping dataframe from pdb data

    scaler = torch.cuda.amp.GradScaler()
        
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # data_path = "/data/bayes_design/pdb_2021aug02/" # when running script outside docker container
    data_path = "/home/pdb_2021aug02/"
    params = {
        "LIST"    : f"{data_path}/list.csv", 
        "VAL"     : f"{data_path}/valid_clusters.txt",
        "TEST"    : f"{data_path}/test_clusters.txt",
        "DIR"     : f"{data_path}",
        "DATCUT"  : "2030-Jan-01",
        "RESCUT"  : 3.5, #resolution cutoff for PDBs. Potentially add flexibility with argparser
        "HOMO"    : 0.70 #min seq.id. to detect homo chains
    }


    LOAD_PARAM = {'batch_size': 1,
                    'shuffle': True,
                    'pin_memory':False,
                    'num_workers': 4}

    print("About to build training clusters")
    train, valid, test = build_training_clusters(params, False)
    print("Finished building training clusters")
        
    train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    train_loader = torch.utils.data.DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    # Potential for validation sets; consider adding argparser to switch modes
    # valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    # valid_loader = torch.utils.data.DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    print("train_loader/valid_loader initialized")

    pdb_dict_train = get_pdbs(train_loader, 1)
    # pdb_dict_valid = get_pdbs(valid_loader, 1)

    sequences = []
    input_dict = {}
    for entry in pdb_dict_train:
        sequences.append([entry.name, entry.sequence])
        input_dict[entry.name] = entry.sequence
        
        
    print(sequences)
    df = pd.DataFrame(sequences, columns=["name", "sequence"])
    print(df.head(5))
    return df

    # next the finetuning script generates a list containing a list of chezod scores per sequence
    # SETH_1.py accepts a dictionary where key:value == name:sequence

if __name__ == "__main__":
    main()