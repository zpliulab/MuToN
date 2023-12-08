import time, os, sys
from Arguments import parser
from dataloader.dataloader import DataLoader as mutation_loader
from dataloader.dataloader_leaveoneout import DataLoader as complex_loader
from models.MuToN import Main_model
from models.MuToN import EarlyStopping

from sklearn.metrics import precision_recall_curve, auc, mean_squared_error
from default_config.dir_options import dir_opts
from tqdm import tqdm
import numpy as np
import pandas as pd

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total: ', total_num, 'Trainable: ', trainable_num)

def printf(args, *content):
    file = sys.stdout
    f_handler = open(os.path.join(args.checkpoints_dir, 'log.txt'), 'a+')
    sys.stdout = f_handler
    print(' '.join(content))
    f_handler.close()
    sys.stdout = file
    print(' '.join(content))

def aupr(true, proba):
    precision, recall, thresholds = precision_recall_curve(true, proba)
    result = auc(recall, precision)
    return result

def train_batch(data, model):
    model.set_input(data)
    out = model.optimize_parameters()
    loss = model.loss
    return loss

def val_batch(data, model):
    model.set_input(data)
    loss, true, pre = model.test()
    return loss, true, pre

if __name__ == '__main__':
    args = parser.parse_args()
    checkpoints_dir = args.checkpoints_dir
    for fold in range(0, 10):
        setattr(args, 'fold', fold)
        model = Main_model(args)
        get_parameter_number(model)
        args.checkpoints_dir = checkpoints_dir + '_' + str(args.fold)
        if not os.path.exists(args.checkpoints_dir):
            os.makedirs(args.checkpoints_dir)
        opts = dir_opts()
        setattr(args, 'dir_opts', opts)
        DataLoader = mutation_loader if args.splitting == 'mutation' else complex_loader
        setattr(args, 'subset', 'train')
        dataset_train = DataLoader(args)
        setattr(args, 'subset', 'val')
        dataset_val = DataLoader(args)
        setattr(args, 'subset', 'test')
        dataset_test = DataLoader(args)

        early_stop = EarlyStopping(opt=args, path=os.path.join(args.checkpoints_dir, 'best.pth'))
        total_steps = 0

        auc_best = 0
        time1 = time.time()
        curve = []
        for epoch in range(100):
            epoch_start_time = time.time()
            epoch_iter = 0

            loss_train = []; proba_train = []; true_train = []
            for i, data in tqdm(enumerate(dataset_train)):
                loss_ = train_batch(data, model)
                loss_train.append(loss_.detach().cpu().numpy())

            loss_train = []; proba_train = []; true_train = []
            for i, data in enumerate(dataset_val):
                loss_, true_, proba_ = val_batch(data, model)
                loss_train.append(np.sqrt(loss_.detach().cpu().numpy()))
                proba_train.append(proba_.detach().cpu().numpy())
                true_train.append(true_.detach().cpu().numpy())

            loss_val = []; proba_val = []; true_val = []
            for i, data in enumerate(dataset_val):
                loss_, true_, proba_ = val_batch(data, model)
                loss_val.append(np.sqrt(loss_.detach().cpu().numpy()))
                proba_val.append(proba_.detach().cpu().numpy())
                true_val.append(true_.detach().cpu().numpy())

            loss_test = []; proba_test = []; true_test = []
            for i, data in enumerate(dataset_test):
                loss_, true_, proba_ = val_batch(data, model)
                loss_test.append(np.sqrt(loss_.detach().cpu().numpy()))
                proba_test.append(proba_.detach().cpu().numpy())
                true_test.append(true_.detach().cpu().numpy())

            loss_train = np.average(np.array(loss_train))
            pre_train = np.hstack(proba_train)
            true_train = np.hstack(true_train)
            rmse_train = mean_squared_error(true_train, pre_train, squared=False)
            pcc_train = np.corrcoef(pre_train, true_train)[0, 1]

            pre_val = np.hstack(proba_val)
            true_val = np.hstack(true_val)
            rmse_val = mean_squared_error(true_val, pre_val, squared=False)
            pcc_val = np.corrcoef(pre_val, true_val)[0, 1]

            pre_test = np.hstack(proba_test)
            true_test = np.hstack(true_test)
            rmse_test = mean_squared_error(true_test, pre_test, squared=False)
            pcc_test = np.corrcoef(pre_test, true_test)[0, 1]

            early_stop(pcc_val, model)
            printf(args, 'Epoch: ', str(epoch),
                   'Loss_train: ', str(loss_train),
                   'Loss_val: ', str(rmse_val),
                   'Rp_val:', str(pcc_val),
                   'Loss_test: ', str(rmse_test),
                   'Rp_test:', str(pcc_test))
            curve.append([rmse_train, pcc_train, rmse_val, pcc_val, rmse_test, pcc_test])
            if early_stop.early_stop == True:
                break
        curve = pd.DataFrame(curve)
        curve.to_csv(os.path.join(args.checkpoints_dir, 'curve.csv'))