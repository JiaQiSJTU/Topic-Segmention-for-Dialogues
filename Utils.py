# encoding = 'utf-8'
import os
import torch
import datetime
from transformers import *
from torch.autograd import Variable
from config import args
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from nltk.metrics.segmentation import pk, windowdiff



#网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def collate_fn(data):
    '''
    Collate list of data in to batch
    :param data: list of dicts ({"dialogue_id":id, "utterances":utterances, "roles":roles,"labels":labels,"conversation_length":len(utterances)})
    :return:
        Batch of each feature
        -
    '''

    dialogue_id, utterances, roles, labels, conversation_length = zip(*data)
    return dialogue_id, utterances, roles, labels, conversation_length


def process_batch_data(batch_data, tokenizer):
    '''
    utterance pad到max_seq_len, conversation pad到batch中最长对话长度
    :param batch_data:
    :param tokenizer:
    :return:
    '''
    dialogue_ids, utterances, roles, labels, conversation_length = batch_data

    padded_utterances = []
    utterances_mask = []

    longest_conversation_in_batch = max(conversation_length)
    # batchsize = len(conversation_length)
    conversation_mask = []

    padded_roles = []

    padded_labels = []

    index_decoder_X = []
    index_decoder_Y = []

    for i in range(len(dialogue_ids)):

        encoded_dicts = [tokenizer.encode_plus(utterance, max_length=args.max_seq_len, pad_to_max_length=True,
                                               add_special_tokens=True, truncation='longest_first') for utterance in
                         utterances[i]]

        padded_utterances += [encoded_dict['input_ids'] for encoded_dict in encoded_dicts]
        utterances_mask += [encoded_dict['attention_mask'] for encoded_dict in encoded_dicts]

        conversation_mask += [
            [1] * conversation_length[i] + [0] * (longest_conversation_in_batch - conversation_length[i])]
        padded_roles += [roles[i] + [0] * (longest_conversation_in_batch - conversation_length[i])]

        padded_labels += [labels[i] + [2] * (longest_conversation_in_batch - conversation_length[i])]


    return dialogue_ids, padded_utterances, utterances_mask, padded_roles, conversation_length, conversation_mask, padded_labels


def write_test_predictions(test_batches, test_scores):
    print("Begin writting predictions into files...")
    path = os.path.join("log_" + str(args.logid), "predicts", args.corpus)
    if not os.path.exists(path):
        os.makedirs(path)
    count = 0
    for batch_data in test_batches:
        dialogue_ids, utterances, roles, labels, conversation_length = batch_data

        for dialogue_id, utterance, role, label in zip(dialogue_ids, utterances, roles, labels):
            test_score = test_scores[count]

            with open(path + '/' + str(dialogue_id), 'a+', encoding='utf-8') as f:

                for utt, rol, pred, truth in zip(utterance, role, test_score, label):
                    f.write(str(rol) + '\t' + str(truth) + '\t' + str(pred) + '\t' + utt + '\n')

            count += 1
    print("Finish writting predictions into files!!!")
    return


def save_model(model, epoch, path='result', **kwargs):
    """
    默认保留所有模型
    :param model: 模型
    :param path: 保存路径
    :param loss: 校验损失
    :param last_loss: 最佳epoch损失
    :param kwargs: every_epoch or best_epoch
    :return:
    """
    path = os.path.join("log_" + str(args.logid), path, args.corpus)

    if not os.path.exists(path):
        os.makedirs(path)
        print("mkdir {}".format(path))
    if kwargs.get('name', None) is None:
        cur_time = datetime.datetime.now().strftime('%Y-%m-%d#%H:%M:%S')
        name = cur_time + '--epoch:{}'.format(epoch)
        full_name = os.path.join(path, name)
        torch.save(model.state_dict(), full_name)
        print('Saved model at epoch {} successfully'.format(epoch))
        with open('{}/checkpoint'.format(path), 'w') as file:
            file.write(name)
            print('Write to checkpoint')


def load_model(model, path='result'):

    path = os.path.join("log_"+str(args.logid),path, args.corpus)

    with open('{}/checkpoint'.format(path)) as file:
        content = file.read().strip()
        name = os.path.join(path, content)

    model.load_state_dict(torch.load(name, map_location=lambda storage, loc: storage))
    print('load model {} successfully'.format(name))
    return model


def pad(tensor, length):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            padded_var = torch.cat([var, torch.zeros((length - var.size(0), *var.size()[1:]), dtype=var.dtype).cuda()])
            return padded_var
        else:
            return var
    else:
        if length > tensor.size(0):
            return torch.cat(
                [tensor, torch.zeros((length - tensor.size(0), *tensor.size()[1:]), dtype=tensor.dtype).cuda()])
        else:
            return tensor


def score(predicts, labels, windowsize=2, type=1):
    '''
    sample_num * conversation_length
    list of numpy arrays
    :param predicts:
    :param masks:
    :param labels:
    :param type: 1 -- origianl dataset, the windowsize is appropriate; 0 -- augmented datset, the windowsize may be wrong
    :return: windowdiff pk F1-macro
    '''

    acc = 0
    f1_macro = 0
    f1_micro = 0
    windiff = 0
    pkk = 0
    for i in range(len(predicts)):

        predict_str = ''.join(str(x) for x in list(predicts[i]))
        label_str = ''.join(str(x) for x in list(labels[i]))
        acc += np.sum(np.equal(predicts[i], labels[i])) / len(predicts[i])
        f1_macro += f1_score(labels[i], predicts[i], average='macro')
        f1_micro += f1_score(labels[i], predicts[i], average='micro')

        if type:
            windiff += windowdiff(label_str, predict_str, windowsize)
            pkk += pk(label_str, predict_str, windowsize)

    acc = acc / len(predicts)
    f1_macro = f1_macro / len(predicts)
    f1_micro = f1_micro / len(predicts)
    if type:
        windiff = windiff / len(predicts)
        pkk = pkk / len(predicts)

    score = {"windowdiff": windiff, "pk": pkk, "F1-macro": f1_macro,"acc":acc}

    return score
