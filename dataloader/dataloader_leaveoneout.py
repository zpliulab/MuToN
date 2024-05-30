import random, os, tqdm
import numpy as np
from joblib import Parallel, delayed, cpu_count
from torch.utils.data import Dataset
from dataloader.extract_multimer import extractPDB, call_modeller, read_PDB, call_foldx
import torch, esm
from dataloader.ESM_encoder import ESM_encoder
tensor = torch.FloatTensor
inttensor = torch.LongTensor
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()


def extract_topology(X1, X2=None, num_nn=16):
    if X2 is None:
        X2 = X1
    R = X1[None,:,:] - X2[:,None,:]
    D = np.linalg.norm(R, axis=2)
    knn = min(num_nn, D.shape[1])
    ids_topk = np.argpartition(D, knn-1, axis=1)
    dis = np.zeros_like(ids_topk[:,:knn], dtype=float)
    for i in range(len(ids_topk)):
        for j in range(knn):
            dis[i,j] = D[i, ids_topk[:,:knn][i,j]]
    ids_topk = ids_topk[:,:knn] + 1
    padding_num = num_nn - D.shape[1] if num_nn > D.shape[1] else 0
    ids_topk = np.pad(ids_topk,((0,0), (0, padding_num)), constant_values=0)
    return ids_topk

def extract_interface(X1, X2, distance=8.0):
    if X2 is None:
        X2 = X1
    R = X1[None,:,:] - X2[:,None,:]
    D = np.linalg.norm(R, axis=2)
    inter_res = D<distance
    inter_res = np.sum(inter_res, axis=1)
    return np.nonzero(inter_res)[0]

def pack(mutation=None,
            token_p1=None,
            token_p2=None,
            token_p3=None,
            xyz_p1=None,
            xyz_p2=None,
            xyz_p3=None,
            nuv_p1=None,
            nuv_p2=None,
            nuv_p3=None,
            y_p1=None,
            y_p2=None,
            y_p3=None,
            llm_p1=None,
            llm_p2=None,
            llm_p3=None,
            topk_p1=None,
            topk_p2=None,
            topk_p3=None,
            topk_p1_p1=None,
            topk_p1_p2=None,
            topk_p1_p3=None):
    triplet = {}
    triplet['token_p1']=token_p1
    triplet['token_p2']=token_p2
    triplet['token_p3']=token_p3

    triplet['xyz_p1']=xyz_p1
    triplet['xyz_p2']=xyz_p2
    triplet['xyz_p3']=xyz_p3

    triplet['nuv_p1']=nuv_p1
    triplet['nuv_p2']=nuv_p2
    triplet['nuv_p3']=nuv_p3

    triplet['y_p1']=y_p1
    triplet['y_p2']=y_p2
    triplet['y_p3']=y_p3

    triplet['llm_p1']=llm_p1
    triplet['llm_p2']=llm_p2
    triplet['llm_p3']=llm_p3

    triplet['topk_p1']=topk_p1
    triplet['topk_p2']=topk_p2
    triplet['topk_p3']=topk_p3

    triplet['topk_p1_p2']=topk_p1_p2
    triplet['topk_p1_p3']=topk_p1_p3
    triplet['mutation'] = mutation
    return triplet

def load_triplet(args, file):
    pdb_id, wild_chains, mutated_chain, res_id, wild_rescode, mutated_rescode, dG1, dG2, ddG = file
    #For receptor side, multimer formation is supported
    token = []; xyz = []; nuv = []; llm = []
    for wild_chain in wild_chains:
        extractPDB(args['complex_dir'], args['single_dir'], pdb_id + '_' + wild_chain)
        token_, xyz_, nuv_, fasta_ = read_PDB(args['single_dir'], pdb_id + '_' + wild_chain)
        llm_ = ESM_encoder(args['llm_dir'], pdb_id + '_' + wild_chain, fasta_, model, alphabet, batch_converter)
        token.append(token_)
        xyz.append(xyz_)
        nuv.append(nuv_)
        llm.append(llm_)
    p1_token = np.hstack(token)
    p1_xyz = np.vstack(xyz)
    p1_nuv = np.vstack(nuv)
    p1_llm = torch.vstack(llm)
    p1_topk = extract_topology(p1_xyz)

    #For ligand side with wild formation
    extractPDB(args['complex_dir'], args['single_dir'], pdb_id + '_' + mutated_chain)
    p2_token, p2_xyz, p2_nuv, p2_fasta = read_PDB(args['single_dir'], pdb_id + '_' + mutated_chain)
    p2_llm = ESM_encoder(args['llm_dir'], pdb_id + '_' + mutated_chain,
                          p2_fasta, model, alphabet, batch_converter)
    p2_topk = extract_topology(p2_xyz)
    p2_topk2_1 = extract_topology(p2_xyz, p1_xyz)
    # topk2_1 = extract_topology(xyz, xyz2)


    #For ligand side with mutated formation
    call_modeller(args['single_dir'], pdb_id + '_' + mutated_chain, res_id, mutated_rescode, wild_rescode)

    #call_foldx(args['single_dir'], pdb_id + '_' + mutated_chain, res_id, mutated_rescode, wild_rescode)
    p3_token, p3_xyz, p3_nuv, p3_fasta = read_PDB(args['single_dir'], '{}.mut.{}_{}'.format(pdb_id + '_' + mutated_chain,res_id, mutated_rescode))
    p3_llm = ESM_encoder(args['llm_dir'], '{}.mut.{}_{}'.format(pdb_id + '_' + mutated_chain,res_id, mutated_rescode),
                          p3_fasta, model, alphabet, batch_converter)
    p3_topk = extract_topology(p3_xyz)
    p3_topk3_1 = extract_topology(p3_xyz, p1_xyz)
    # topk3_1 = extract_topology(xyz, xyz3)

    # if p1['token'].shape[0]!=p1['llm'].shape[0] or p2['token'].shape[0]!=p2['llm'].shape[0] or p3['token'].shape[0]!=p3['llm'].shape[0]:
    #     print('hello')
    Triplet = pack(
            token_p1=inttensor(p1_token),
            token_p2=inttensor(p2_token),
            token_p3=inttensor(p3_token),
            xyz_p1=tensor(p1_xyz),
            xyz_p2=tensor(p2_xyz),
            xyz_p3=tensor(p3_xyz),
            nuv_p1=tensor(p1_nuv),
            nuv_p2=tensor(p2_nuv),
            nuv_p3=tensor(p3_nuv),
            llm_p1=tensor(p1_llm),
            llm_p2=tensor(p2_llm),
            llm_p3=tensor(p3_llm),
            topk_p1=inttensor(p1_topk),
            topk_p2=inttensor(p2_topk),
            topk_p3=inttensor(p3_topk),
            topk_p1_p2=inttensor(p2_topk2_1),
            topk_p1_p3=inttensor(p3_topk3_1),
            y_p1=tensor([ddG, ]),
            y_p2=tensor([dG1, ]),
            y_p3=tensor([dG2, ]),
            mutation='{}_{}_{}_{}{}{}'.format(pdb_id, wild_chains, mutated_chain, wild_rescode, res_id, mutated_rescode))
    return Triplet

def load_pdb(args, files, parallelize=False):
    print('Loading pdbs, paralleling: ', parallelize)
    if parallelize == False:
        pdbs_including_wrong = []
        for file in tqdm.tqdm(files):
            try:
                a = load_triplet(args, file)
                pdbs_including_wrong.append(a)
            except:
                continue
    else:
        n_jobs = cpu_count() - 1
        n_jobs = 16
        pdbs_including_wrong = Parallel(n_jobs=n_jobs, verbose=False, timeout=None)(
            delayed(load_triplet)(args, file) for i, file in enumerate(files)
        )
    pdbs = []
    for item in pdbs_including_wrong:
        if len(item['token_p1']) >= 1 and len(item['token_p2']) >= 1 and len(item['token_p3']) >=1:
            pdbs.append(item)
    return pdbs


def load_SKEMPI2():
    with open('dataset/skempi_v2.csv', 'r') as pid:
        lines = pid.readlines()[1:]
    non_redundant = set()
    # record = []
    pdbid2sites = {}
    pdbid = []
    for line in lines:
        line = line.split(';')
        # if len(line[2].split(','))==1:
        #     continue
        mutated_rescode = []
        wild_rescode = []
        mutated_resid = []
        mutated_chain = []
        for item in line[2].split(','):
            mutated_rescode.append(item[-1])
            wild_rescode.append(item[0])
            mutated_resid.append(item[2:-1])
            mutated_chain.append(item[1])
        if len(set(mutated_chain)) != 1:
            continue
        mutated_chain = mutated_chain[0]
        mutated_rescode = '_'.join(mutated_rescode)
        wild_rescode = '_'.join(wild_rescode)
        mutated_resid = '_'.join(mutated_resid)
        pdb_id, p1, p2 = line[0].split('_')
        mutated_chain = line[2][1]
        wild_chain = p1 if mutated_chain in p2 else p2
        # mutated_resid = line[2][2:-1]
        # wild_rescode= line[2][0]
        # mutated_rescode = line[2][-1]
        # R = 8.314/4184
        R = 0.001985
        try:
            T = float(line[13][:3])
            a = np.float32(298 * R * np.log(float(line[8])))
            b = np.float32(298 * R * np.log(float(line[7])))
        except:
            continue
        dG1 = np.float32(T * R * np.log(float(line[8]))) #wild
        dG2 = np.float32(T * R * np.log(float(line[7])))#mutated
        # print(dG1-dG2)
        rec = '{},{},{}:{}\n'.format(pdb_id, wild_chain+'_'+mutated_chain, mutated_chain, wild_rescode+str(mutated_resid)+mutated_rescode)
        # pdb_id_wild_chain = pdb_id +'_'+ wild_chain
        if rec not in non_redundant:
            if pdb_id in pdbid2sites.keys():
                pdbid2sites[pdb_id].append(
                    [pdb_id, wild_chain, mutated_chain, mutated_resid, wild_rescode, mutated_rescode, dG1, dG2, dG1-dG2])
            else:
                pdbid2sites[pdb_id]=[
                    [pdb_id, wild_chain, mutated_chain, mutated_resid, wild_rescode, mutated_rescode, dG1, dG2, dG1-dG2], ]
                pdbid.append(pdb_id)
            non_redundant.add(rec)
        else:
            pass
    return pdbid2sites, pdbid

def load_S1131():
    with open('data/SKEMPI/S1131.csv', 'r') as pid:
        lines = pid.readlines()
    non_redundant = set()
    # record = []
    pdbid2sites = {}
    pdbid = []
    for line in lines[1:]:
        line = line.split(',')
        pdb_id = line[0]
        info = line[4]
        mutated_chain = info.split(':')[0]
        wild_rescode = info.split(':')[1][0]
        mutated_resid = info.split(':')[1][1:-1]
        mutated_rescode = info.split(':')[1][-1]

        p1_chain = line[1].split('_')
        wild_chain = p1_chain[0] if p1_chain[1] in mutated_chain else p1_chain[1]
        dG1 = 0  # wild
        dG2 = 0
        ddG = float(line[5])
        # print(dG1-dG2)
        rec='{},{},{}:{}\n'.format(pdb_id, wild_chain + '_' + mutated_chain, mutated_chain,
                                   wild_rescode + str(mutated_resid) + mutated_rescode)
        if rec not in non_redundant:
            if pdb_id in pdbid2sites.keys():
                pdbid2sites[pdb_id].append(
                    [pdb_id, wild_chain, mutated_chain, mutated_resid, wild_rescode, mutated_rescode, dG1, dG2,
                     ddG])
            else:
                pdbid2sites[pdb_id]=[
                    [pdb_id, wild_chain, mutated_chain, mutated_resid, wild_rescode, mutated_rescode, dG1, dG2,
                     ddG], ]
                pdbid.append(pdb_id)
            non_redundant.add(rec)
        else:
            pass
    return pdbid2sites, pdbid

def load_S4169():
    with open('data/SKEMPI/S4169.txt', 'r') as pid:
        lines = pid.readlines()
    non_redundant = set()
    # record = []
    pdbid2sites = {}
    pdbid = []
    for line in lines[1:]:
        line = line.split(',')
        pdb_id = line[0]
        info = line[4]
        mutated_chain = info.split(':')[0]
        wild_rescode = info.split(':')[1][0]
        mutated_resid = info.split(':')[1][1:-1]
        mutated_rescode = info.split(':')[1][-1]

        p1_chain = line[1].split('_')
        wild_chain = p1_chain[0] if p1_chain[1] in mutated_chain else p1_chain[1]
        dG1 = 0  # wild
        dG2 = 0
        ddG = float(line[5])
        # print(dG1-dG2)
        rec='{},{},{}:{}\n'.format(pdb_id, wild_chain + '_' + mutated_chain, mutated_chain,
                                   wild_rescode + str(mutated_resid) + mutated_rescode)
        if rec not in non_redundant:
            if pdb_id in pdbid2sites.keys():
                pdbid2sites[pdb_id].append(
                    [pdb_id, wild_chain, mutated_chain, mutated_resid, wild_rescode, mutated_rescode, dG1, dG2,
                     ddG])
            else:
                pdbid2sites[pdb_id]=[
                    [pdb_id, wild_chain, mutated_chain, mutated_resid, wild_rescode, mutated_rescode, dG1, dG2,
                     ddG], ]
                pdbid.append(pdb_id)
            non_redundant.add(rec)
        else:
            pass
    return pdbid2sites, pdbid

def load_M1101():
    import pandas as pd
    lines = pd.read_csv('data/SKEMPI/M1101.csv')

    non_redundant = set()
    # record = []
    pdbid2sites = {}
    pdbid = []
    for line in lines.values:
        pdb_id = line[0]
        info = line[4]

        mutated_rescode = []
        wild_rescode = []
        mutated_resid = []
        mutated_chain = []
        for item in info.split(','):
            mutated_rescode.append(item[-1])
            wild_rescode.append(item[2])
            mutated_resid.append(item[3:-1])
            mutated_chain.append(item[0])
        if len(set(mutated_chain)) != 1:
            continue
        mutated_chain = mutated_chain[0]
        mutated_rescode = '_'.join(mutated_rescode)
        wild_rescode = '_'.join(wild_rescode)
        mutated_resid = '_'.join(mutated_resid)

        p1_chain = line[1].split('_')
        wild_chain = p1_chain[0] if p1_chain[1] in mutated_chain else p1_chain[1]
        dG1 = 0  # wild
        dG2 = 0
        ddG = float(line[5])
        # print(dG1-dG2)
        rec='{},{},{}:{}\n'.format(pdb_id, wild_chain + '_' + mutated_chain, mutated_chain,
                                   wild_rescode + str(mutated_resid) + mutated_rescode)
        if rec not in non_redundant:
            if pdb_id in pdbid2sites.keys():
                pdbid2sites[pdb_id].append(
                    [pdb_id, wild_chain, mutated_chain, mutated_resid, wild_rescode, mutated_rescode, dG1, dG2,
                     ddG])
            else:
                pdbid2sites[pdb_id]=[
                    [pdb_id, wild_chain, mutated_chain, mutated_resid, wild_rescode, mutated_rescode, dG1, dG2,
                     ddG], ]
                pdbid.append(pdb_id)
            non_redundant.add(rec)
        else:
            pass
    return pdbid2sites, pdbid

def collate_fn(batch):
    """Creates mini-batch tensors
    We should build custom collate_fn rather than using default collate_fn
    """
    following_batch = ['xyz_p1', 'xyz_p2', 'xyz_p3']
    meta = {}
    # meta['']
    keys = batch[0].keys()
    for key in keys:
        if key == 'mutation':
            meta.update({key: [d[key] for d in batch]})
        else:
            meta.update({key: torch.concat([d[key] for d in batch])})
            if key in following_batch:
                meta.update({key+'_batch': torch.concat([torch.ones(d[key].shape[0],dtype=torch.int64)*i for i, d in enumerate(batch)])})

    return meta

class DataLoader:
    def __init__(self, opt):
        self.opt = opt
        self.dataset = self.CreateDataset()

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            collate_fn=collate_fn)

    def CreateDataset(self):
        pdbid2sites, pdbid = eval('load_'+self.opt.dataset)()
        random.seed(2023)
        random.shuffle(pdbid)
        fold_frac_test = 0.1
        train_val_pdbs = pdbid[:int(len(pdbid)*fold_frac_test)*self.opt.fold] + pdbid[int(len(pdbid)*fold_frac_test)*(self.opt.fold+1):]
        test_pdbs = pdbid[int(len(pdbid)*fold_frac_test)*self.opt.fold: int(len(pdbid)*fold_frac_test)*(self.opt.fold+1)]
        test_records = [site for pdb in test_pdbs for site in pdbid2sites[pdb]]
        train_val_records = [site for pdb in train_val_pdbs for site in pdbid2sites[pdb]]
        random.shuffle(train_val_records)
        train_records = train_val_records[int(len(train_val_records)*0.1):]
        val_records = train_val_records[:int(len(train_val_records)*0.1)]
        if self.opt.subset == 'train':
            records = train_records
        if self.opt.subset == 'val':
            records = val_records
        if self.opt.subset == 'test':
            records = test_records
        records2 = []
        for item in records:
            if len(item[3].split('_'))==1:
                continue
            records2.append(item)
        # records = records [0:10]

        loaded_pdbs = load_pdb(self.opt.dir_opts, records, parallelize=True)
        print(self.opt.dataset, ': {}'.format(len(loaded_pdbs)))
        return loaded_pdbs

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= len(self.dataset):
                break
            yield data
