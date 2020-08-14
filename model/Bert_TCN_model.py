import torch
from torch import nn
import sys
sys.path.append("../../")
from model.TCN import TemporalConvNet
from transformers import BertModel
from Utils import pad
from model.focalLoss import FocalLoss


class Bert_TCN(nn.Module):

    def __init__(self, args):
        super(Bert_TCN, self).__init__()

        self.batch_size = args.batch_size

        # Bert
        self.bert_config = args.load_pretrained_model_path
        self.bert_batch_size = args.bert_batch_size

        # TCN
        self.input_size = args.bert_emb
        self.output_size = args.label_num
        self.num_channels = [args.nhid] * args.levels
        self.kernel_size = args.kernel_size
        self.dropout = args.dropout
        self.emb_dropout = args.emb_dropout

        # Loss
        self.learning_algorithm = args.loss_function
        self.label_paddingId = args.label_paddingId
        self.gamma = args.gamma


        # Layers
        self.bert_layer = BertModel.from_pretrained(self.bert_config, output_hidden_states=True)

        if args.freeze_bert:
            for name, param in self.bert_layer.named_parameters():
                param.requires_grad = False

        self.drop = nn.Dropout(self.emb_dropout)
        self.tcn = TemporalConvNet(self.input_size, self.num_channels, self.kernel_size, dropout=self.dropout)


        self.linear_layer = nn.Linear(self.num_channels[-1], self.output_size)

        # add
        self.sig = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        self.linear_layer.bias.data.fill_(0)
        self.linear_layer.weight.data.normal_(0, 0.01)

    def forward(self, utterances, utterances_mask, roles, conversation_mask, conversation_length):
        '''

        :param input:
            - utterances = [batch_size * utt_num] * max_seq_len
            - utterances_mask = [batchsize * utt_num] * max_seq_len
            - roles = batch_size * max_turn_len
            - conversation_mask = batch_size * max_turn_len
            - conversation_length = batch_size

        :return: output: batchsize * max_turn_len * output_size
        '''

        '''
        bert layer
        since [batch_size * utt_num] may be too large, divide into bert_batch_size * max_seq_len pieces to go through the bert
        '''
        num_sentences = utterances.size(0)

        input_utterances = [utterances[i:i + self.bert_batch_size, :] for i in
                            range(0, num_sentences, self.bert_batch_size)]
        input_utterances_masks = [utterances_mask[i:i + self.bert_batch_size, :] for i in
                                  range(0, num_sentences, self.bert_batch_size)]

        bert_encoded_embeds = []  # [batch_size utt_num] * max_seq_len * hidden_size

        for input_utterances_piece, input_utterances_masks_piece in zip(input_utterances, input_utterances_masks):
            _, _, bert_encoded_piece = self.bert_layer(input_utterances_masks_piece, input_utterances_masks_piece)
            # extract output tensor from the second-to-last layer and average pooling
            # use the [CLS] for sentence embedding
            # bert_batch_size * hidden_size
            bert_encoded_piece = torch.stack(tuple(bert_encoded_piece[i] for i in range(2, len(bert_encoded_piece), 1)),
                                             dim=0)

            bert_encoded_piece = torch.mean(bert_encoded_piece[:, :, 0, :], dim=0)

            bert_encoded_embeds += bert_encoded_piece.tolist()

        # recover to batch_size * max_turn_len * hidden_size
        max_turn_len = conversation_mask.size(1)

        start = torch.cumsum(torch.cat((conversation_length.new(1).zero_(), conversation_length[:-1])), 0)

        bert_encoded_embeds = torch.stack(
            [pad(torch.tensor(bert_encoded_embeds, dtype=torch.float32).cuda().narrow(0, s, l), max_turn_len) for s, l
             in zip(start.data.tolist(), conversation_length.data.tolist())],
            0).cuda()  # batchsize * max_turn_len * hidden_size

        '''TCN
        Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)
        Ouput batch_size * max_turn_len * output_size
        '''
        # emb = self.drop(bert_encoded_embeds)
        emb = bert_encoded_embeds


        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.linear_layer(y)

        # add
        y = self.sig(y)

        return y


    def loss(self, preds, mask, labels):
        """
        preds: size=(batch_size, max_turn_len, tag_size)
            mask: size=(batch_size, seq_len)
            labels: size=(batch_size, seq_len)
        :return:
        """

        loss = self._calculate_loss(preds, mask, labels)

        return loss

    def _loss(self, learning_algorithm, label_paddingId):
        """
        :param learning_algorithm:
        :param label_paddingId:
        :param use_crf:
        :return:
        """

        if learning_algorithm == "SGD":
            loss_function = nn.CrossEntropyLoss(ignore_index=label_paddingId, reduction="sum")
            return loss_function
        elif learning_algorithm == "CrossEntropy":
            loss_function = nn.CrossEntropyLoss(ignore_index=label_paddingId, reduction="mean")
            return loss_function
        elif learning_algorithm == "FocalLoss":
            # loss_function = FocalLoss(self.output_size,  gamma=self.gamma, size_average=True, ignore_index=label_paddingId)
            loss_function = FocalLoss(gamma=self.gamma, alpha = torch.tensor([0.25,0.75]), size_average=False, ignore_index=label_paddingId)
            return loss_function

    def _calculate_loss(self, feats, mask, tags):
        """
        Args:
            feats: size = (batch_size, seq_len, tag_size)
            mask: size = (batch_size, seq_len)
            tags: size = (batch_size, seq_len)
        """
        loss_function = self._loss(learning_algorithm=self.learning_algorithm,
                                   label_paddingId=self.label_paddingId)
        batch_size, max_len = feats.size(0), feats.size(1)

        feats = feats.view(batch_size * max_len, -1)


        tags = tags.view(-1)

        return loss_function(feats, tags)

