'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

from functools import partial

import os
import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

import numpy as np

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data,
    load_mlm_data,
    MLMDataset
)

from evaluation import model_eval_sst, model_eval_paraphrase, model_eval_sts, model_eval_lin, model_eval_multitask, model_eval_test_multitask

from minlora import (
    LoRAParametrization,
    add_lora,
    apply_to_lora,
    merge_lora,
)

from minlora import add_lora, apply_to_lora, disable_lora, enable_lora, get_lora_params, merge_lora, name_is_lora, remove_lora, load_multiple_lora, select_lora

TQDM_DISABLE=False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Pretrain mode does not require updating BERT paramters.
        self.set_grad(config)
        # You will want to add layers here to perform the downstream tasks.

        self.shared_layer = torch.nn.Linear(self.bert.config.hidden_size, 256)
        self.sentiment_linear2 = torch.nn.Linear(256, 5)
        self.paraphrase_linear2 = torch.nn.Linear(256, 1)
        self.similarity_linear2 = torch.nn.Linear(256, 1)
        self.linguistic_linear2 = torch.nn.Linear(256, 1)
        self.initialize_weights()

        if config.lora:
            self.add_lora(config.lora)

    def set_grad(self, config):
        for param in self.bert.parameters():
            if config.option == 'finetune':
                param.requires_grad = True
            elif config.option == 'pretrain':
                param.requires_grad = False

    def add_lora(self, r):
        lora_config = {
            nn.Embedding: {
                "weight": partial(LoRAParametrization.from_embedding, rank=r),
            },
            nn.Linear: {
                "weight": partial(LoRAParametrization.from_linear, rank=r),
            },
        }
        add_lora(self.bert, lora_config=lora_config)

        for param in self.bert.parameters():
            param.requires_grad = False
        for param in get_lora_params(self.bert):
            param.requires_grad = True

    def initialize_weights(self):
        init_method = torch.nn.init.xavier_uniform_
        init_method(self.shared_layer.weight)
        init_method(self.sentiment_linear2.weight)
        init_method(self.paraphrase_linear2.weight)
        init_method(self.similarity_linear2.weight)
        init_method(self.linguistic_linear2.weight)

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO

        return self.bert.forward(input_ids, attention_mask)

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO

        bert_out = self.bert.forward(input_ids, attention_mask)["pooler_output"]
        x = self.shared_layer(bert_out)
        x = F.gelu(x)
        x = self.sentiment_linear2(x)
        return x

    def predict_paraphrase(self, input_ids, attention_mask):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO

        bert_out = self.bert.forward(input_ids, attention_mask)["pooler_output"]
        x = self.shared_layer(bert_out)
        x = F.gelu(x)
        x = self.paraphrase_linear2(x)
        return x

    def predict_similarity(self, input_ids, attention_mask):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO

        bert_out = self.bert.forward(input_ids, attention_mask)["pooler_output"]
        x = self.shared_layer(bert_out)
        x = F.gelu(x)
        x = self.similarity_linear2(x)
        return x

    def predict_linguistic(self, input_ids, attention_mask):
        bert_out = self.bert.forward(input_ids, attention_mask)["pooler_output"]
        x = self.shared_layer(bert_out)
        x = F.gelu(x)
        x = self.linguistic_linear2(x)
        return x

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def get_optimizer(args, model):
    if args.enable_per_layer_finetune:
        DECAY_RATE = args.enable_per_layer_finetune

        parameters = []

        # 14 bert layers, 1 embedding, 12 attention, 1 pooler
        per_layer_lr = []
        for i in range(14):
            per_layer_lr.append(args.lr * 0.95**i)
        per_layer_lr = per_layer_lr[::-1]

        layer_names = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                layer_names.append(name)

        for idx, name in enumerate(layer_names):
            layer_lr = args.lr
            if "bert" in name and "embed" in name:
                layer_lr = per_layer_lr[0]
            elif "bert_layers" in name:
                layer_lr = per_layer_lr[int(name.split(".")[2]) + 1]
            elif "bert.pooler" in name:
                layer_lr = per_layer_lr[13]
            parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
                            'lr':     layer_lr}]

        return AdamW(parameters)
    else:
        return AdamW(model.parameters(), lr=args.lr)

def train_sst(args, model, device, config):
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data, lin_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train,args.lin_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data, lin_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,args.lin_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    lr = args.lr
    optimizer = get_optimizer(args, model)
    best_dev_acc = 0

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_sentiment(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        with open(args.acc_out, "a") as f:
            f.write(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}\n")

def train_paraphrase(args, model, device, config):
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data, lin_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train,args.lin_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data, lin_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,args.lin_dev, split ='train')

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn)

    lr = args.lr
    optimizer = get_optimizer(args, model)
    best_dev_acc = 0

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            (b_ids, b_mask,
             b_labels, b_sent_ids) = (batch['token_ids'], batch['attention_mask'],
                          batch['labels'], batch['sent_ids'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            b_labels = b_labels.float().unsqueeze(-1).to(device)

            optimizer.zero_grad()
            logits = model.predict_paraphrase(b_ids, b_mask)
            loss = F.binary_cross_entropy_with_logits(logits, b_labels) / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, *_ = model_eval_paraphrase(para_train_dataloader, model, device)
        dev_acc, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        with open(args.acc_out, "a") as f:
            f.write(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}\n")

def train_sts(args, model, device, config):
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data, lin_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train,args.lin_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data, lin_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,args.lin_dev, split ='train')

    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    lr = args.lr
    optimizer = get_optimizer(args, model)
    best_dev_corr = 0

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            (b_ids, b_mask,
             b_labels, b_sent_ids) = (batch['token_ids'], batch['attention_mask'],
                          batch['labels'], batch['sent_ids'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            b_labels = b_labels.float().unsqueeze(-1).to(device)

            optimizer.zero_grad()
            logits = model.predict_similarity(b_ids, b_mask)
            loss = F.mse_loss(logits, b_labels) / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_corr, *_ = model_eval_sts(sts_train_dataloader, model, device)
        dev_corr, *_ = model_eval_sts(sts_dev_dataloader, model, device)

        if dev_corr > best_dev_corr:
            best_dev_corr = dev_corr
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train corr :: {train_corr :.3f}, dev corr :: {dev_corr :.3f}")
        with open(args.acc_out, "a") as f:
            f.write(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train corr :: {train_corr :.3f}, dev corr :: {dev_corr :.3f}\n")

def train_lin(args, model, device, config):
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data, lin_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train,args.lin_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data, lin_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,args.lin_dev, split ='train')

    lin_train_data = SentenceClassificationDataset(lin_train_data, args)
    lin_dev_data = SentenceClassificationDataset(lin_dev_data, args)

    lin_train_dataloader = DataLoader(lin_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=lin_train_data.collate_fn)
    lin_dev_dataloader = DataLoader(lin_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=lin_dev_data.collate_fn)

    lr = args.lr
    optimizer = get_optimizer(args, model)
    best_dev_acc = 0

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(lin_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels, b_sent_ids = batch['token_ids'], batch['attention_mask'], batch['labels'], batch['sent_ids']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            b_labels = b_labels.float().unsqueeze(-1).to(device)

            optimizer.zero_grad()
            logits = model.predict_linguistic(b_ids, b_mask)
            loss = F.binary_cross_entropy_with_logits(logits, b_labels) / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, *_ = model_eval_lin(lin_train_dataloader, model, device)
        dev_acc, *_ = model_eval_lin(lin_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        with open(args.acc_out, "a") as f:
            f.write(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}\n")

def train_pretraining(args, model, device, config):
    all_sentences = load_mlm_data(args.sst_train,args.para_train,args.sts_train,args.lin_train, args.pretrain_dataset)

    mlm_dataset = MLMDataset(all_sentences, {})
    mlm_dataloader = torch.utils.data.DataLoader (
        mlm_dataset, batch_size=args.pretrain_batch_size, collate_fn=mlm_dataset.collate_fn, shuffle=True
    )

    from transformers import BertForMaskedLM

    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    if os.path.isfile(args.enable_pretrain):
      model.load_state_dict(torch.load(args.enable_pretrain))
      print(f"Loaded cached further pretraining MLM model {args.enable_pretrain}")
    for param in model.parameters():
        param.requires_grad = True
    model.to(device)

    optimizer = get_optimizer(args, model)

    # Run for the specified number of epochs.
    for epoch in range(args.pretrain_epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(mlm_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = batch['input_ids'], batch['attention_mask'], batch['labels']

            b_ids = b_ids.squeeze(1).to(device)
            b_mask = b_mask.squeeze(1).to(device)

            b_labels = b_labels.squeeze(1).to(device)

            optimizer.zero_grad()
            out = model(b_ids, b_mask)
            logits = out.logits
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), b_labels.view(-1)) / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        print(f"Pretrain epoch {epoch}: train loss :: {train_loss :.3f}")
        with open(args.acc_out, "a") as f:
            f.write(f"Pretrain epoch {epoch}: train loss :: {train_loss :.3f}\n")

        fname = f"epoch_{epoch}_" + args.enable_pretrain
        torch.save(model.state_dict(), fname)
        print(f"Saved pre-trained BERT to {fname}")

    return model

def convert_state_dict(path):
    state_dict = torch.load(path)
    # Convert old format to new format if needed from a PyTorch state_dict
    old_keys = []
    new_keys = []
    m = {'embeddings.word_embeddings': 'word_embedding',
         'embeddings.position_embeddings': 'pos_embedding',
         'embeddings.token_type_embeddings': 'tk_type_embedding',
         'embeddings.LayerNorm': 'embed_layer_norm',
         'embeddings.dropout': 'embed_dropout',
         'encoder.layer': 'bert_layers',
         'pooler.dense': 'pooler_dense',
         'pooler.activation': 'pooler_af',
         'attention.self': "self_attention",
         'attention.output.dense': 'attention_dense',
         'attention.output.LayerNorm': 'attention_layer_norm',
         'attention.output.dropout': 'attention_dropout',
         'intermediate.dense': 'interm_dense',
         'intermediate.intermediate_act_fn': 'interm_af',
         'output.dense': 'out_dense',
         'output.LayerNorm': 'out_layer_norm',
         'output.dropout': 'out_dropout'}

    for key in state_dict.keys():
      new_key = None
      if "gamma" in key:
        new_key = key.replace("gamma", "weight")
      if "beta" in key:
        new_key = key.replace("beta", "bias")
      for x, y in m.items():
        if new_key is not None:
          _key = new_key
        else:
          _key = key
        if x in key:
          new_key = _key.replace(x, y)
      if new_key:
        old_keys.append(key)
        if new_key.startswith("bert."):
            new_key = new_key[5:]
        new_keys.append(new_key)

    new_state_dict = {}
    for old_key, new_key in zip(old_keys, new_keys):
      # print(old_key, new_key)
      new_state_dict[new_key] = state_dict.pop(old_key)

    return new_state_dict

def train_multi(args, model, device, config):
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data, lin_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train,args.lin_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data, lin_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,args.lin_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn)

    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    lin_train_data = SentenceClassificationDataset(lin_train_data, args)
    lin_dev_data = SentenceClassificationDataset(lin_dev_data, args)

    lin_train_dataloader = DataLoader(lin_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=lin_train_data.collate_fn)
    lin_dev_dataloader = DataLoader(lin_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=lin_dev_data.collate_fn)

    sst_train_iter = iter(sst_train_dataloader)
    para_train_iter = iter(para_train_dataloader)
    sts_train_iter = iter(sts_train_dataloader)
    lin_train_iter = iter(lin_train_dataloader)
    iters = [sst_train_iter, para_train_iter, sts_train_iter, lin_train_iter]

    lr = args.lr
    optimizer = get_optimizer(args, model)
    best_dev_acc = 0

    iters_per_epoch = 1000

    # Run for the specified number of epochs.
    for epoch in range(args.multi_epochs):
        model.train()
        train_loss = 0

        for num_batches in tqdm(range(iters_per_epoch), desc=f'train-{epoch}', disable=TQDM_DISABLE):
            dataset = np.random.randint(0, len(iters) - 1)
            try:
                batch = next(iters[dataset])
            except StopIteration:
                if dataset == 0: iters[dataset] = iter(sst_train_dataloader)
                if dataset == 1: iters[dataset] = iter(para_train_dataloader)
                if dataset == 2: iters[dataset] = iter(sts_train_dataloader)
                if dataset == 3: iters[dataset] = iter(lin_train_dataloader)
                batch = next(iters[dataset])

            if dataset == 0:
                b_ids, b_mask, b_labels = (batch['token_ids'],
                               batch['attention_mask'], batch['labels'])

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_sentiment(b_ids, b_mask)
                loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            elif dataset == 1:
                (b_ids, b_mask,
                 b_labels, b_sent_ids) = (batch['token_ids'], batch['attention_mask'],
                              batch['labels'], batch['sent_ids'])

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)

                b_labels = b_labels.float().unsqueeze(-1).to(device)

                optimizer.zero_grad()
                logits = model.predict_paraphrase(b_ids, b_mask)
                loss = F.binary_cross_entropy_with_logits(logits, b_labels) / args.batch_size

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            elif dataset == 2:
                (b_ids, b_mask,
                 b_labels, b_sent_ids) = (batch['token_ids'], batch['attention_mask'],
                              batch['labels'], batch['sent_ids'])

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)

                b_labels = b_labels.float().unsqueeze(-1).to(device)

                optimizer.zero_grad()
                logits = model.predict_similarity(b_ids, b_mask)
                loss = F.mse_loss(logits, b_labels) / args.batch_size

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            elif dataset == 3:
                b_ids, b_mask, b_labels = batch['input_ids'], batch['attention_mask'], batch['labels']

                b_ids = b_ids.squeeze(1).to(device)
                b_mask = b_mask.squeeze(1).to(device)

                b_labels = b_labels.squeeze(1).to(device)

                optimizer.zero_grad()
                out = model(b_ids, b_mask)
                logits = out.logits
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), b_labels.view(-1)) / args.batch_size

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

        train_loss = train_loss / iters_per_epoch

        save_model(model, optimizer, args, config, args.filepath)
        if args.multitask_filepath:
            save_model(model, optimizer, args, config, args.multitask_filepath)
"""
        sst_train_acc, *_ = model_eval_sst(sst_train_dataloader, model, device)
        sst_dev_acc, *_ = model_eval_sst(sst_dev_dataloader, model, device)
        para_train_acc, *_ = model_eval_paraphrase(para_train_dataloader, model, device)
        para_dev_acc, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)
        sts_train_acc, *_ = model_eval_sts(sts_train_dataloader, model, device)
        sts_dev_acc, *_ = model_eval_sts(sts_dev_dataloader, model, device)
        lin_train_acc, *_ = model_eval_lin(lin_train_dataloader, model, device)
        lin_dev_acc, *_ = model_eval_lin(lin_dev_dataloader, model, device)

        average_acc = (sst_dev_acc + para_dev_acc + sts_dev_acc + lin_dev_acc) / 3

        if average_acc > best_dev_acc:
            best_dev_acc = average_acc

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, sst train acc :: {sst_train_acc :.3f}, sst dev acc :: {sst_dev_acc :.3f}")
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, para train acc :: {para_train_acc :.3f}, para dev acc :: {para_dev_acc :.3f}")
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, sts train acc :: {sts_train_acc :.3f}, sts dev acc :: {sts_dev_acc :.3f}")
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, lin train acc :: {lin_train_acc :.3f}, lin dev acc :: {lin_dev_acc :.3f}")
        print(f"Epoch {epoch}: average_acc:: {average_acc :.3f}")
        with open(args.acc_out, "a") as f:
            f.write(f"Epoch {epoch}: train loss :: {train_loss :.3f}, sst train acc :: {sst_train_acc :.3f}, sst dev acc :: {sst_dev_acc :.3f}\n")
            f.write(f"Epoch {epoch}: train loss :: {train_loss :.3f}, para train acc :: {para_train_acc :.3f}, para dev acc :: {para_dev_acc :.3f}\n")
            f.write(f"Epoch {epoch}: train loss :: {train_loss :.3f}, sts train acc :: {sts_train_acc :.3f}, sts dev acc :: {sts_dev_acc :.3f}\n")
            f.write(f"Epoch {epoch}: train loss :: {train_loss :.3f}, lin train acc :: {lin_train_acc :.3f}, lin dev acc :: {lin_dev_acc :.3f}\n")
            f.write(f"Epoch {epoch}: average_acc:: {average_acc :.3f}\n")
"""


def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''

    device = torch.device(args.device) if args.use_gpu else torch.device('cpu')

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              # 'num_labels': num_labels,
              'lora': args.lora,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}
    config = SimpleNamespace(**config)
    model = MultitaskBERT(config)

    lr = args.lr
    model = model.to(device)

    if args.load_pretrain:
        state_dict = convert_state_dict(args.load_pretrain)
        if args.lora:
            model = model.to("cpu")
            remove_lora(model)
            model = model.to(device)
        state_dict["position_ids"] = model.state_dict()["bert.position_ids"]
        state_dict["pooler_dense.weight"] = model.state_dict()["bert.pooler_dense.weight"]
        state_dict["pooler_dense.bias"] = model.state_dict()["bert.pooler_dense.bias"]
        model.bert.load_state_dict(state_dict)
        model.set_grad(config)
        if args.lora:
            model = model.to("cpu")
            model.add_lora(args.lora)
            model = model.to(device)
        print(f"Loaded pre-trained BERT from {args.load_pretrain}")

    if args.enable_pretrain:
        assert args.option == "finetune"
        assert not args.lora
        pretrained_model = train_pretraining(args, model, device, config)
        torch.save(pretrained_model.state_dict(), args.enable_pretrain)
        print(f"Saved pre-trained BERT to {args.enable_pretrain}")

        state_dict = convert_state_dict(args.enable_pretrain)
        state_dict["position_ids"] = model.state_dict()["bert.position_ids"]
        state_dict["pooler_dense.weight"] = model.state_dict()["bert.pooler_dense.weight"]
        state_dict["pooler_dense.bias"] = model.state_dict()["bert.pooler_dense.bias"]
        model.bert.load_state_dict(state_dict)
        model.set_grad(config)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_parameters}")

    if args.multitask:
        if os.path.isfile(args.multitask_filepath):
            print(f"Loaded saved multitask from {args.multitask_filepath}")
            saved = torch.load(args.multitask_filepath)
            model.load_state_dict(saved['model'])
            model = model.to(device)
        else:
            train_multi(args, model, device, config)
    if args.task == 'sst':
        train_sst(args, model, device, config)
    elif args.task == 'para':
        train_paraphrase(args, model, device, config)
    elif args.task == 'sts':
        train_sts(args, model, device, config)
    elif args.task == 'lin':
        train_lin(args, model, device, config)

def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device(args.device) if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data, lin_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, args.lin_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data, lin_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,args.lin_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        lin_test_data = SentenceClassificationTestDataset(lin_test_data, args)
        lin_dev_data = SentenceClassificationDataset(lin_dev_data, args)

        lin_test_dataloader = DataLoader(lin_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=lin_test_data.collate_fn)
        lin_dev_dataloader = DataLoader(lin_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=lin_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids, \
            dev_linguistic_accuracy, dev_lin_y_pred, dev_lin_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader,
                                                                    lin_dev_dataloader, model, device)

        if args.test:
            test_sst_y_pred, \
                test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids, test_lin_y_pred, test_lin_sent_ids = \
                    model_eval_test_multitask(sst_test_dataloader,
                                              para_test_dataloader,
                                              sts_test_dataloader, lin_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        if args.test:
            with open(args.sst_test_out, "w+") as f:
                f.write(f"id \t Predicted_Sentiment \n")
                for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                    f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        if args.test:
            with open(args.para_test_out, "w+") as f:
                f.write(f"id \t Predicted_Is_Paraphrase \n")
                for p, s in zip(test_para_sent_ids, test_para_y_pred):
                    f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        if args.test:
            with open(args.sts_test_out, "w+") as f:
                f.write(f"id \t Predicted_Similiary \n")
                for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                    f.write(f"{p} , {s} \n")

        with open(args.lin_dev_out, "w+") as f:
            print(f"dev linguistic acc :: {dev_linguistic_accuracy :.3f}")
            f.write(f"id \t Predicted_Linguistic \n")
            for p, s in zip(dev_lin_sent_ids, dev_lin_y_pred):
                f.write(f"{p} , {s} \n")

        if args.test:
            with open(args.lin_test_out, "w+") as f:
                f.write(f"id \t Predicted_Linguistic \n")
                for p, s in zip(test_lin_sent_ids, test_lin_y_pred):
                    f.write(f"{p} , {s} \n")

        with open(args.acc_out, "a") as f:
            f.write(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}\n")
            f.write(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}\n")
            f.write(f"dev sts corr :: {dev_sts_corr :.3f}\n")
            f.write(f"dev linguistic acc :: {dev_linguistic_accuracy :.3f}\n")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--lin_train", type=str, default="cola_public/raw/in_domain_train.tsv")
    parser.add_argument("--lin_dev", type=str, default="cola_public/raw/in_domain_dev.tsv")
    parser.add_argument("--lin_test", type=str, default="cola_public/raw/cola_in_domain_test.tsv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--task", type=str,
                        choices=('sst', 'para', 'sts', 'lin', 'none'), default="none")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--lin_dev_out", type=str, default="predictions/lin-dev-output.csv")
    parser.add_argument("--lin_test_out", type=str, default="predictions/lin-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    parser.add_argument("--f", type=str, default="")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--test", default=False, const=True, nargs="?")

	# Experiments
    parser.add_argument("--load_pretrain", const="pretrain.pt", nargs="?", default=False)
    parser.add_argument("--enable_pretrain", const="pretrain.pt", nargs="?", default=False)
    parser.add_argument("--pretrain_epochs", type=int, default=20)
    parser.add_argument("--pretrain_dataset", type=str, default="sst-para-sts-lin")
    parser.add_argument("--pretrain_batch_size", type=int, default=8)

    parser.add_argument("--enable_per_layer_finetune", const=0.95, nargs="?", default=False)
    parser.add_argument("--enable_multitask_finetune", action="store_true")

    parser.add_argument("--lora", const=4, type=int, default=False, nargs="?")

    parser.add_argument("--multitask", default=False, const=True, nargs="?")
    parser.add_argument("--multitask_filepath", default=False, const="multitask.pt", nargs="?")
    parser.add_argument("--multi_epochs", default=3, type=int)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-{args.f}-multitask.pt' # Save path.
    args.acc_out = f'output/{args.f}_{args.task}_acc.txt'
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    if args.task != "none":
        test_multitask(args)
