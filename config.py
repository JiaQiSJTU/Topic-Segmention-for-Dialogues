# encoding = 'utf-8'
import argparse

parser = argparse.ArgumentParser(description='Topic Segmentation with Bert-TCN model')

parser.add_argument("--model", type=str, default='TCN')

parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size (default: 16)')
parser.add_argument('--bert_batch_size', type=int, default=128, help='the batchsize for sentence encoder')
parser.add_argument('--cuda', type=bool, default=True,
                    help='use CUDA (default: True)')
# parser.add_argument('--load_pretrained_model', action='store_true')
parser.add_argument('--load_pretrained_model_path', type=str, default='./model/bert-base-chinese/',
                    help='the location of the pretrained model')
parser.add_argument('--model_path', type=bool, default=False, help="load_trained_model")

parser.add_argument('--data_path', type=str,
                    default='./Dataset',
                    help='location of the Weibo data for topic segmentation provided from the paper')

parser.add_argument('--logid', type=str, default='weibo')

parser.add_argument('--corpus', type=str, default='Weibo',
                    help='')
parser.add_argument('--label_num', type=int, default=2, help='the number of labels')
parser.add_argument('--shuffle_train', type=bool, default=True, help='whether shuffle training samples')
parser.add_argument('--clip', type=float, default=0.2,
                    help='gradient clip, -1 means no clip (default: 0.35)')
parser.add_argument("--max_seq_len", type=int, default=50)
parser.add_argument("--optim", type=str, default="Adam")
# parser.add_argument("--log_interval", type=int, default=30, help="print log")
parser.add_argument("--early_stop", type=int, default=3)
parser.add_argument("--label_paddingId", type=int, default=2)

parser.add_argument("--loss_function", type=str, default='FocalLoss', help='')
parser.add_argument("--gamma", type=int, default=2, help="parameter for FocusLoss")

parser.add_argument("--lr", type=float, default= 1e-4, help="original: 0.0001")
# parser.add_argument("--lr_decay", type=float, default=0.00001, help="original: 0.00001")
# parser.add_argument("--weight_decay", type=float, default=0.00005, help='original:0.00005')
parser.add_argument("--seed",type = int, default=1111,help="random seed")


parser.add_argument("--eval_window", type=int, default=4,
                    help='parameter for evalutation metric: set to half the average segment length in the reference segmentation')
parser.add_argument("--topic_augment_size", type=int, default=1,
                    help='the data augmentation size of training data for topic segmentation')

# Bert settings
parser.add_argument('--bert_emb', type=int, default=768, help="the hidden size of bert model")
parser.add_argument('--freeze_bert', type=bool, default='True')

# TCN settings
parser.add_argument('--forwardTCN', type=bool, default=False, help='TCN only for forward turns')
parser.add_argument('--dropout', type=float, default=0.25,
                    help='dropout applied to layers (default: 0.25)')
parser.add_argument('--emb_dropout', type=float, default=0.25,
                    help='dropout applied to the embedded layer (default: 0.25)')
parser.add_argument('--kernel_size', type=int, default=5,
                    help='kernel size (default: 3)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
parser.add_argument('--nhid',type=int, default=150,help='number of hidden units per layer (default: 150)')

args = parser.parse_args()
print(args)
