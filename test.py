# encoding = 'utf-8'
# from CNNDataset import CNNDMTSDataset
from Dataset.WeiboDataset import WeiboDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from model.Bert_TCN_model import Bert_TCN
from Utils import *
import os
from main import eval

os.environ["CUDA_VISIBLE_DEVICES"] = "3,2,1"


tokenizer = BertTokenizer.from_pretrained(args.load_pretrained_model_path)
model = Bert_TCN(args).cuda()
print(get_parameter_number(model))

if args.model_path:
    model = load_model(model)
    print("successfully load pre-trained model")
else:
    print("no pre-trained model")
    exit(0)

print("===loading test data==")
test_dataset = WeiboDataset(args.data_path, file_type='test_simplified')
test_batches = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
print("test:length={},batch_num={}".format(test_dataset.__len__(), len(test_batches)))


optimizer = getattr(optim, args.optim)
optimizer = optimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


print('begin test evaluation')
test_loss, test_scores,test_predicts = eval(args,model, tokenizer, test_batches)
write_test_predictions(test_batches,test_predicts)
