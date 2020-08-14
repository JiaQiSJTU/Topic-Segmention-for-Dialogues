# encoding = 'utf-8'
# from torch.utils.data import Dataset, DataLoader
import json
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np


class WeiboDataset(Dataset):
    '''
    Reads in the Weibo TopicSegmentation dataset
    '''
    def __init__(self, dataset_folder, file_type):
        self.dataset_foleder = dataset_folder
        self.file_type = file_type
        self.all_samples = self.load_examples(os.path.join(dataset_folder,'weibo_'+self.file_type+'.txt'))
        self.len = len(self.all_samples)



    def __getitem__(self, index):
        sample = self.all_samples[index]
        dialogue_id = sample['dialogue_id']
        utterances = sample['utterances']
        roles = sample['roles']
        labels = sample['labels']
        conversation_length = sample['conversation_length']

        return dialogue_id, utterances, roles,labels, conversation_length


    def __len__(self):
        return self.len

    def load_examples(self,filename):

        examples = []
        with open(filename,'r',encoding = 'utf-8') as f:
            line_num = 0
            for line in f:

                labels, utterances = line.strip('\n').split('+++$+++')
                utterances = utterances.split('\t')
                labels = [int(t) for t in labels.split(',')]
                assert len(utterances)==len(labels)

                roles = [1] * len(labels)
                topic_starts_idx = [i for i in range(len(labels)) if labels[i]==1]


                examples.append({"dialogue_id":line_num, "utterances":utterances, "roles":roles,"labels":labels,"conversation_length":len(utterances),"topic_starts_idx":topic_starts_idx})
                line_num+=1
            print("load {} samples from {}".format(len(examples),filename))


        return examples

    def get_labels(self):
        return {"B":1, "I":0}

    def map_label(self,label):
        return self.get_labels()[label.strip()]

















