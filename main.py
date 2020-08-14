# encoding = 'utf-8'
# from CNNDataset import CNNDMTSDataset
from Dataset.WeiboDataset import WeiboDataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from model.Bert_TCN_model import Bert_TCN
from Utils import *
import time
import os
import numpy

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.manual_seed(args.seed)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():

    tokenizer = BertTokenizer.from_pretrained(args.load_pretrained_model_path)
    model = Bert_TCN(args).cuda()
    print(get_parameter_number(model))

    if args.model_path:
        model = load_model(model)

    print("===loading training data==")
    train_dataset = WeiboDataset(args.data_path, file_type='train')
    if args.shuffle_train:
        train_batches = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                   collate_fn=collate_fn)
    else:
        train_batches = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False,
                                   collate_fn=collate_fn)
    print("train:length={},batch_num={}".format(train_dataset.__len__(), len(train_batches)))

    print("===loading dev data==")
    dev_dataset = WeiboDataset(args.data_path, file_type = 'dev')
    dev_batches = DataLoader(dataset = dev_dataset, batch_size = args.batch_size, shuffle=False,collate_fn = collate_fn)
    print("dev:length={},batch_num={}".format(dev_dataset.__len__(), len(dev_batches)))
    print("===loading test data==")
    test_dataset = WeiboDataset(args.data_path, file_type='test')
    test_batches = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    print("test:length={},batch_num={}".format(test_dataset.__len__(), len(test_batches)))


    optimizer = getattr(optim, args.optim)
    # optimizer = optimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr = args.lr
    optimizer = optimizer(model.parameters(), lr=lr)


    best_dev_loss = 100000
    step = 0
    no_improvement = 0


    for epoch in range(args.epochs):
        model.train()

        start_time = time.time()
        total_loss = 0
        count = 0

        train_predict = []
        train_groundtruth = []

        for i, train_batch in enumerate(train_batches):
            step += 1
            model.zero_grad()

            # 读出一个batch的数据,进行tokenize和生成mask
            dialogue_ids, padded_utterances, utterances_mask, padded_roles, conversation_length, conversation_mask, padded_labels = process_batch_data(train_batch, tokenizer)

            y = numpy.array(padded_labels)


            utterances = torch.LongTensor(padded_utterances).cuda()
            utterances_mask = torch.LongTensor(utterances_mask).cuda()
            roles = torch.LongTensor(padded_roles).cuda()
            conversation_length = torch.LongTensor(conversation_length).cuda()
            conversation_mask = torch.LongTensor(conversation_mask).cuda()
            padded_labels = torch.LongTensor(padded_labels).cuda()


            predicts= model(utterances,utterances_mask,roles,conversation_mask,conversation_length)
            train_loss = model.loss(predicts, conversation_mask, padded_labels)
            train_loss.backward()



            print(train_loss)

            total_loss += train_loss * len(dialogue_ids)
            count += predicts.size(0)


            y_hat = torch.argmax(F.softmax(predicts,dim=-1),dim=-1).cpu().numpy()


            for i in range(conversation_length.size(0)):

                train_predict.append(y_hat[i,:conversation_length[i]])
                train_groundtruth.append(y[i,:conversation_length[i]])


            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            if step%100==0:
                print("epoch: {}/{} | step: {} | total loss: {}".format(epoch,args.epochs,step,total_loss))


        scores = score(train_predict, train_groundtruth, windowsize=args.eval_window, type=1)
        print('epoch: {}/{} | time elapse: {} |  loss: {} | windowdiff: {} | pk: {} | F1-macro:{} | Acc:{}'.format(epoch, args.epochs,time.time()-start_time, total_loss/count*1.0, scores['windowdiff'], scores['pk'],scores['F1-macro'],scores['acc']))

        print("begin dev evaluation")
        dev_loss, dev_scores ,dev_predicts= eval(args, model,tokenizer, dev_batches,epoch )

        if dev_loss < best_dev_loss:
            print("new best loss: {}".format(dev_loss))
            best_dev_loss = dev_loss
            save_model(model, epoch)
            no_improvement=0
        else:
            no_improvement+=1
            print("didn't beat the best loss, impatience:{}".format(no_improvement))

        if epoch > 10 and no_improvement>=args.early_stop:

            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


        if no_improvement==5 or epoch==(args.epochs-1):
        # if no_improvement==args.early_stop or epoch==(args.epochs-1):
            print("Ran out of patience, stop training")
            print('begin test evaluation')
        #     break

    # test
    # print("Ran out of patience, stop training")
    # print('begin test evaluation')
    model = load_model(model) # load the best model
    test_loss, test_scores,test_predicts = eval(args,model, tokenizer, test_batches)
    write_test_predictions(test_batches,test_predicts)



def eval(args,model,tokenizer, dev_batches, epoch=None):

    model.eval()
    start_eval_time = time.time()
    eval_predict = []
    eval_groundtruth = []
    # masks = []
    # length = 0
    count = 0
    total_eval_loss = 0
    with torch.no_grad():
        for i, dev_batch in enumerate(dev_batches):
            dialogue_ids, padded_utterances, utterances_mask, padded_roles, conversation_length, conversation_mask, labels = process_batch_data(dev_batch, tokenizer)

            # truth += labels
            # masks += conversation_mask
            y = numpy.array(labels)

            utterances = torch.LongTensor(padded_utterances).cuda()
            utterances_mask = torch.LongTensor(utterances_mask).cuda()
            roles = torch.LongTensor(padded_roles).cuda()
            conversation_length = torch.LongTensor(conversation_length).cuda()
            conversation_mask = torch.LongTensor(conversation_mask).cuda()
            labels = torch.LongTensor(labels).cuda()

            # length += len(dialogue_ids)

            predicts = model(utterances, utterances_mask, roles, conversation_mask, conversation_length)
            eval_loss = model.loss(predicts, conversation_mask, labels)

            total_eval_loss += eval_loss * len(dialogue_ids)
            count += predicts.size(0)



            y_hat = torch.argmax(F.softmax(predicts,dim=-1),dim=-1).cpu().numpy()

            for i in range(conversation_length.size(0)):

                eval_predict.append(y_hat[i,:conversation_length[i]])
                eval_groundtruth.append(y[i,:conversation_length[i]])


    val_scores = score(eval_predict, eval_groundtruth,windowsize=args.eval_window, type = 1)
    val_loss = total_eval_loss / count * 1.0

    print('eval: epoch: {}/{} | time elapse: {} |  loss: {} | windowdiff: {} | pk: {} | F1-macro:{} | Acc:{}'.format(epoch, args.epochs, time.time() - start_eval_time, val_loss,val_scores['windowdiff'],val_scores['pk'],val_scores['F1-macro'],val_scores['acc']))



    return val_loss, val_scores,eval_predict









if __name__ == '__main__':
    train()