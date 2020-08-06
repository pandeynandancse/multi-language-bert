import config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

#--------------------------------------------------
#for tpu setting (else not required)
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xla_multiprocessing
import torch_xla.distributed.parallel_loader as pl
#--------------------------------------------------



def run():
    df1 = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-train.csv", usecols = ['comment_text','toxic'])
    df1 = pd.read_csv("../input/jigsaw-unintended-bias-train.csv", usecols = ['comment_text','toxic'])
    
    #combined df1 and df2 and made big dataframe
    df_train = pd.concat([df1,df2],axis=0).reset_index(drop=True)

    #validation dataframe has been given by kaggle
    df_valid - pd.read_csv("../input/validation.csv")

    train_dataset = dataset.BERTDataset(
        comment_text=df_train.comment_text.values, target=df_train.toxic.values
    )


    #--------------------------------------
    #write sampler if using tpu else not 
    train_sampler = torch.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas = xm.xrt_world_size(), 
            rank = xm.get_ordinal(),
            shuffle = True
        )
    #----------------------------------------



    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.TRAIN_BATCH_SIZE, 
        num_workers=4,
        sampler = train_sampler,
        #problem with tpu when using torch_xla is that if batch size is not equal then it's going to crash , so use drop_last 
        drop_last = True

    )



    valid_dataset = dataset.BERTDataset(
        comment_text=df_valid.comment_text.values, target=df_valid.toxic.values
    )



    #--------------------------------------
    #write sampler if using tpu else not 
    valid_sampler = torch.data.distributed.DistributedSampler(
            valid_dataset,
            num_replicas = xm.xrt_world_size(), 
            rank = xm.get_ordinal(),
            shuffle = True
        )
    #----------------------------------------------



    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=config.VALID_BATCH_SIZE, 
        num_workers=1,
        sampler = valid_sampler,
        #no need of drop_last here
    )



    
    device = xm.xla_device()  #xla_device means tpu
    model = BERTBaseUncased()
    # model.to(device)  #no need to move data on device
    
    #specify what parameters you want to train
    param_optimizer = list(model.named_parameters())
	
    #we don't want any deacy for these layer names such as bias and othr following things
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    
    optimizer_parameters = [
        {
	   #don't decay weight for above no_decay list else decay
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE / xm.xrt_world_size() * config.EPOCHS)

    lr = 3e-5 * xm.xrt_world_size()
    #experiment with lr
    optimizer = AdamW(optimizer_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    
    

    best_accuracy = 0
    for epoch in range(config.EPOCHS):


        #parallel loader for tpus
        para_loader = pl.ParallelLoader(train_data_loader,[device])
        engine.train_fn(para_loader.per_device_loader(device), model, optimizer, device, scheduler)
        

        parallel_loader = pl.ParallelLoader(valid_data_loader,[device])
        outputs, targets = engine.eval_fn(para_loader.per_device_loader(device), model, device)
        

        #threshold the target instead of output 
        targets = np.array(targets) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score = {accuracy}")
        if accuracy > best_accuracy:

            #instead of torch.save use xm.save
            xm.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy


if __name__ == "__main__":
    run()
