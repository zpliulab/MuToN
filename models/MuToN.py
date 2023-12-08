import torch
import os, sys
from torch import nn
import numpy as np
from models.function import scatter_sum
from torch.nn.functional import pad
from models.Attention import Geo_attention, Interface_attention, Interface

def printf(args, *content):
    file = sys.stdout
    f_handler = open(os.path.join(args.checkpoints_dir, 'log.txt'), 'a+')
    sys.stdout = f_handler
    print(' '.join(content))
    f_handler.close()
    sys.stdout = file
    print(' '.join(content))

class EarlyStopping:
    def __init__(self, opt, patience_stop=10, patience_lr=5, verbose=False, delta=0.001, path='check1.pth'):
        self.opt = opt
        self.stop_patience = patience_stop
        self.lr_patience = patience_lr
        self.verbose = verbose
        self.counter = 0
        self.best_fitness = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_fitness, model):
        if self.best_fitness is None:
            self.best_fitness = val_fitness
            self.save_checkpoint(model)
            printf(self.opt, 'saving best model...')
            return True
        elif val_fitness <= self.best_fitness + self.delta:
            self.counter += 1
            if self.counter == self.lr_patience:
                self.adjust_lr(model)

            if self.counter >= self.stop_patience:
                self.early_stop = True
            return False
        else:
            self.best_fitness = val_fitness
            self.save_checkpoint(model)
            self.counter = 0
            printf(self.opt, 'saving best model...')
            return False

    def adjust_lr(self, model):
        lr = model.optimizer.param_groups[0]['lr']
        lr = lr/10
        for param_group in model.optimizer.param_groups:
            param_group['lr'] = lr
        model.load_state_dict(torch.load(self.path))
        printf(self.opt, 'loading best model, changing learning rate to %.7f' % lr)

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

class Main_model(nn.Module):

    def __init__(self, opt):
        super(Main_model, self).__init__()
        self.opt = opt

        self.gpu_ids = opt.device
        self.device = torch.device('{}'.format(self.gpu_ids)) if self.gpu_ids else torch.device('cpu')
        in_channels = 21
        I = in_channels #input
        E = opt.emb_dims # Embedding dims per head

        self.lr = opt.lr

        self.embed_tokens = nn.Embedding(
            21, E//2
        ).to(self.device)
        # # self.embed_tokens = nn.Sequential(
        # #     nn.Linear(21, E//2),
        # #     nn.ELU(),
        # # ).to(self.device)
        self.embed_llm = nn.Sequential(
            nn.LayerNorm(1280),
            nn.Linear(1280, E),
            nn.ELU(),
            nn.Linear(E, E//2),
            nn.ELU(),

        ).to(self.device)

        self.stru_net = nn.ModuleList([Geo_attention(E, E, 4, opt.radius_structure) for i in range(opt.n_layers_structure)]
                                        ).to(self.device)

        self.inter_net = nn.ModuleList([Interface_attention(E, E, 4, opt.radius_interface) for i in range(1)]
                                             ).to(self.device)

        self.net_out = nn.Sequential(
            nn.Linear(E, E, bias=False),
            nn.Linear(E, E, bias=False),
            nn.Linear(E, 1, bias=False),
            ).to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)
        self.criterion = torch.nn.MSELoss().to(self.device)


    def load_stru_net(self):
        save_path = 'Pretrained_model/lsm/best_4_64.pth'
        pretrained_dict = torch.load(save_path)
        model_dict = self.state_dict()

        pretrained_dict2 = {'.'.join(key.split('.')[1:]): value for key, value in pretrained_dict.items() if key in model_dict}

        model_dict.update(pretrained_dict2)
        self.load_state_dict(pretrained_dict, strict=False)
        pass
        for param in self.named_parameters():
            param[1].requires_grad = False
            pass

    def set_input(self, data):
        self.P1 = self.process_single(data, chain_idx='1')
        self.P2 = self.process_single(data, chain_idx='2')
        self.P3 = self.process_single(data, chain_idx='3')

    def collect_topk(self, topk, batch1, batch2=None):
        if torch.sum(batch1==1) == 0:
            return torch.nn.functional.pad(topk, (0,0,1,0), value=1)
        if batch2 is None:
            batch2 = batch1
        mask = topk==0
        indices1 = torch.nonzero(torch.eq(batch1[1:] - batch1[:-1], 1)).squeeze() + 1
        indices2 = torch.nonzero(torch.eq(batch2[1:] - batch2[:-1], 1)).squeeze() + 1
        if len(indices1.shape) == 0:
            indices1 = indices1.unsqueeze(0)
            indices2 = indices2.unsqueeze(0)
        for item in range(indices1.shape[0]-1):
            topk[indices1[item]:indices1[item+1]] += indices2[item]
        topk[indices1[-1]:] += indices2[-1]
        topk[mask] = 0
        topk = torch.nn.functional.pad(topk, (0,0,1,0), value=1)
        return topk

    def process_single(self, protein_pair, chain_idx='1'):
        P = {}
        P['llm'] = protein_pair.get('llm_p{}'.format(chain_idx)).to(self.device)
        P['llm'] = pad(P['llm'], (0,0,1,0), value=0).to(self.device)

        P['token'] = protein_pair.get('token_p{}'.format(chain_idx))
        P['token'] = pad(P['token'], (1,0), value=0).to(self.device)
        # P['token'] = protein_pair.get('token_p{}'.format(chain_idx)).unsqueeze(-1)
        # P['token'] = torch.zeros([P['token'].shape[0], 21]).scatter_(1,  P['token'].type(torch.int64), 1)
        # P['token'] = pad(P['token'], (0, 0, 1, 0), value=0).to(self.device)

        P['xyz'] = protein_pair.get('xyz_p{}'.format(chain_idx)).to(self.device)
        P['xyz'] = pad(P['xyz'], (0,0,1,0), value=0).to(self.device)

        P['nuv'] = protein_pair.get('nuv_p{}'.format(chain_idx)).to(self.device)
        P['nuv'] = torch.nn.functional.pad(P['nuv'], (0,0,0,0,1,0), value=0).to(self.device)

        P['y'] = protein_pair.get('y_p{}'.format(chain_idx)).to(self.device)
        P['batch'] = protein_pair.get('xyz_p{}_batch'.format(chain_idx)).to(self.device)

        topk = protein_pair.get('topk_p{}'.format(chain_idx)).to(self.device)
        P['topk'] = self.collect_topk(topk, P['batch'])
        if chain_idx in ['2', '3']:
            topk_interface = protein_pair.get('topk_p1_p{}'.format(chain_idx)).to(self.device)
            P['topk_interface'] = self.collect_topk(topk_interface, self.P1['batch'], P['batch'])
            # P['topk_interface'] = self.collect_topk(topk_interface, P['batch'], self.P1['batch'])
        # P['mt_info'] = protein_pair.get('mt_info_p{}'.format(chain_idx)).to(self.device)

        return P

    def token_onehot(self, seq, dims):
        return torch.zeros([seq.shape[0], dims]).to(self.device).scatter_(1, seq[:, None], 1)


    def embed_inter(self, P1, P2):#P1: wild, P2: mutated
        feature = P1['features']
        for encoder in self.inter_net:
            feature = encoder(P1['features'], P2['features'],
                                     P1['xyz'], P2['xyz'],
                                     P1['nuv'], P2['nuv'], P2['topk_interface'])
        return feature[1:]

    def embed_stru(self):
        for encoder in self.stru_net:
            self.P1['features'] = encoder(self.P1['features'], self.P1['xyz'], self.P1['nuv'], self.P1['topk'])
            self.P2['features'] = encoder(self.P2['features'], self.P2['xyz'], self.P2['nuv'], self.P2['topk'])
            self.P3['features'] = encoder(self.P3['features'], self.P3['xyz'], self.P3['nuv'], self.P3['topk'])

    def forward(self):
        # self.P1['features'] = self.embed_tokens(self.P1['token'])
        # self.P2['features'] = self.embed_tokens(self.P2['token'])
        # self.P3['features'] = self.embed_tokens(self.P3['token'])
        self.P1['f_token'] = self.embed_tokens(self.P1['token'])
        self.P2['f_token'] = self.embed_tokens(self.P2['token'])
        self.P3['f_token'] = self.embed_tokens(self.P3['token'])
        self.P1['f_llm'] = self.embed_llm(self.P1['llm'])
        self.P2['f_llm'] = self.embed_llm(self.P2['llm'])
        self.P3['f_llm'] = self.embed_llm(self.P3['llm'])

        self.P1['features'] = torch.hstack([self.P1['f_token'], self.P1['f_llm']])
        self.P2['features'] = torch.hstack([self.P2['f_token'], self.P2['f_llm']])
        self.P3['features'] = torch.hstack([self.P3['f_token'], self.P3['f_llm']])

        self.embed_stru()

        f1 = self.embed_inter(self.P1, self.P2)
        f2 = self.embed_inter(self.P1, self.P3)

        f1 = scatter_sum(f1, self.P1['batch'], dim=0)
        f2 = scatter_sum(f2, self.P1['batch'], dim=0)
        ddG = self.net_out(f1-f2).squeeze()
        self.P1["iface_preds"] = ddG

        pass

    def optimize_parameters(self):
        self.train()
        self.forward()
        self.loss = self.compute_loss()
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return self.loss

    def test(self):
        self.eval()
        with torch.no_grad():
            self.forward()
            self.loss = self.compute_loss()
        return self.loss, self.P1['y'], self.P1["iface_preds"]

    def compute_loss(self):
        loss = self.criterion(self.P1["iface_preds"], self.P1['y'])
        return loss

    def load_network(self, which_epoch='best'):
        save_filename = '%s.pth' % which_epoch
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        self.load_state_dict(torch.load(save_path))
        printf(self.opt, 'best model loaded:', save_path)

    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = '%s_net.pth' % which_epoch
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        if torch.cuda.is_available():
            torch.save(self.cpu().state_dict(), save_path)
            self.cuda(self.device)
        else:
            torch.save(self.cpu().state_dict(), save_path)

    def load_network_downsteam(self, which_epoch='best'):
        save_filename = '%s.pth' % which_epoch
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        pretrained_dict = torch.load(save_path)
        model_dict = self.state_dict()

        pretrained_dict = {key: value for key, value in pretrained_dict.items() if
                         (value.shape == model_dict[key].shape)}

        model_dict.update(pretrained_dict)
