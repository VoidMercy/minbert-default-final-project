#!/usr/bin/env python3

'''
This module contains our Dataset classes and functions that load the three datasets
for training and evaluating multitask BERT.

Feel free to edit code in this file if you wish to modify the way in which the data
examples are preprocessed.
'''

import csv
import hashlib

import torch
from torch.utils.data import Dataset
from tokenizer import BertTokenizer
import numpy as np
import config
from transformers import DataCollatorForLanguageModeling

def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())

class SentenceClassificationDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):

        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data


# Unlike SentenceClassificationDataset, we do not load labels in SentenceClassificationTestDataset.
class SentenceClassificationTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        return token_ids, attention_mask, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data


class SentencePairDataset(Dataset):
    def __init__(self, dataset, args, isRegression=False):
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression 
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1['input_ids'])
        attention_mask = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids = torch.LongTensor(encoding1['token_type_ids'])

        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])
        if self.isRegression:
            labels = torch.DoubleTensor(labels)
        else:
            labels = torch.LongTensor(labels)

        return (token_ids, token_type_ids, attention_mask,
                token_ids2, token_type_ids2, attention_mask2,
                labels,sent_ids)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask,
         token_ids2, token_type_ids2, attention_mask2,
         labels, sent_ids) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'token_type_ids_1': token_type_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids2,
                'token_type_ids_2': token_type_ids2,
                'attention_mask_2': attention_mask2,
                'labels': labels,
                'sent_ids': sent_ids
            }

        return batched_data


# Unlike SentencePairDataset, we do not load labels in SentencePairTestDataset.
class SentencePairTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1['input_ids'])
        attention_mask = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids = torch.LongTensor(encoding1['token_type_ids'])

        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])


        return (token_ids, token_type_ids, attention_mask,
                token_ids2, token_type_ids2, attention_mask2,
               sent_ids)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask,
         token_ids2, token_type_ids2, attention_mask2,
         sent_ids) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'token_type_ids_1': token_type_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids2,
                'token_type_ids_2': token_type_ids2,
                'attention_mask_2': attention_mask2,
                'sent_ids': sent_ids
            }

        return batched_data

class MLMDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.collate_fn = DataCollatorForLanguageModeling (
                                tokenizer=self.tokenizer,
                                mlm=True,
                                mlm_probability=0.15,
                                return_tensors="pt",
                            )

        self.column_names = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # output = self.tokenizer.encode(self.dataset[idx], return_tensors='pt', padding=True, truncation=True, return_special_tokens_mask=True)[0, ...]
        # return output
        output = self.tokenizer(self.dataset[idx], return_tensors='pt', padding="max_length", truncation=True, return_special_tokens_mask=True)
        return output

    # def collate_fn(self, all_data):
    #     encoding = self.tokenizer(all_data, return_tensors='pt', padding=True, truncation=True)
    #     token_ids = torch.LongTensor(encoding["input_ids"])
    #     attention_mask = torch.LongTensor(encoding["attention_mask"])
    #     token_type_ids = torch.LongTensor(encoding["token_type_ids"])

    #     special_tokens = set(self.tokenizer.all_special_ids)
    #     print(self.tokenizer.mask_token_id)
    #     for sent in range(token_ids.shape[0]):
    #         for token in range(token_ids.shape[1]):
    #             if token_ids[sent, token] not in special_tokens and attention_mask[sent, token] == 1:
    #                 if np.random.random() <= 0.15:
    #                     # Mask token
    #                     temp = np.random.random()
    #                     if temp <= 0.8: 
    #                         token_ids[sent, token] = self.tokenizer.mask_token_id
    #                     elif temp <= 0.9:
    #                         # random token
    #                         rand_token = 0
    #                         while rand_token in special_tokens:
    #                             rand_token = np.random.randint(0, 30522-1)
    #                         token_ids[sent, token] = rand_token

    #     batched_data = {
    #         "token_ids" : token_ids,
    #         "attention_mask" : attention_mask,
    #         "token_type_ids" : token_type_ids
    #     }
        
    #     return batched_data

def load_multitask_data(sentiment_filename,paraphrase_filename,similarity_filename,linguistic_filename,split='train'):
    sentiment_data = []
    num_labels = {}
    if split == 'test':
        with open(sentiment_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                sentiment_data.append((sent,sent_id))
    else:
        with open(sentiment_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                sentiment_data.append((sent, label,sent_id))

    print(f"Loaded {len(sentiment_data)} {split} examples from {sentiment_filename}")

    paraphrase_data = []
    if split == 'test':
        with open(paraphrase_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                paraphrase_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        sent_id))

    else:
        with open(paraphrase_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                try:
                    sent_id = record['id'].lower().strip()
                    paraphrase_data.append((preprocess_string(record['sentence1']),
                                            preprocess_string(record['sentence2']),
                                            int(float(record['is_duplicate'])),sent_id))
                except:
                    pass

    print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")

    similarity_data = []
    if split == 'test':
        with open(similarity_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2'])
                                        ,sent_id))
    else:
        with open(similarity_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        float(record['similarity']),sent_id))

    print(f"Loaded {len(similarity_data)} {split} examples from {similarity_filename}")

    linguistic_data = []
    if split == 'test':
        with open(linguistic_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = int(record['Id'].lower().strip())
                linguistic_data.append((preprocess_string(record['Sentence']),
                                        sent_id))
    else:
        with open(linguistic_filename, 'r') as fp:
            data = fp.read().strip().split("\n")
            for record in data:
                _, annotation, _, sentence = record.split("\t")
                sent_id = hashlib.md5(sentence.encode()).hexdigest()
                linguistic_data.append((preprocess_string(sentence), int(annotation), sent_id))

    print(f"Loaded {len(linguistic_data)} {split} examples from {linguistic_filename}")

    return sentiment_data, num_labels, paraphrase_data, similarity_data, linguistic_data

def load_mlm_data(sentiment_filename,paraphrase_filename,similarity_filename,linguistic_filename, datasets):
    sentiment_data, num_labels, paraphrase_data, similarity_data, linguistic_data = load_multitask_data(sentiment_filename,paraphrase_filename,similarity_filename,linguistic_filename, split="train")
    all_sentences = []
    if "sst" in datasets:
        for record in sentiment_data:
            all_sentences.append(record[0])
    if "para" in datasets:
        for record in paraphrase_data:
            all_sentences.append(record[0])
            all_sentences.append(record[1])
    if "sts" in datasets:
        for record in similarity_data:
            all_sentences.append(record[0])
            all_sentences.append(record[1])
    if "lin" in datasets:
        for record in similarity_data:
            all_sentences.append(record[0])

    return all_sentences

if __name__ == "__main__":
    sentiment_data, num_labels, paraphrase_data, similarity_data, linguistic_data = load_multitask_data("data/ids-sst-train.csv", "data/quora-train.csv", "data/sts-train.csv", "cola_public/raw/in_domain_train.tsv", split="train")
    all_sentences = load_mlm_data("data/ids-sst-train.csv", "data/quora-train.csv", "data/sts-train.csv", "cola_public/raw/in_domain_train.tsv", "sst-para-sts-lin")
    print("Num MLM:", len(all_sentences))
    dataset = MLMDataset(all_sentences, {})
    dataloader = torch.utils.data.DataLoader (
        dataset, batch_size=4, collate_fn=dataset.collate_fn, shuffle=True
    )

    from transformers import BertForMaskedLM, Trainer, TrainingArguments
    import datasets

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    for batch in dataloader:
        print(batch)
        print(batch["input_ids"].shape)
        print(batch["labels"].shape)
        exit()