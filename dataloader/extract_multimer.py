import os, shutil
from Bio.PDB import *
import numpy as np
from numpy import linalg as LA
import math
from subprocess import Popen, PIPE
from Bio.SeqUtils import seq1, seq3
Amino_acid_type = [
"ILE",
"VAL",
"LEU",
"PHE",
"CYS",
"MET",
"ALA",
"GLY",
"THR",
"SER",
"TRP",
"TYR",
"PRO",
"HIS",
"GLU",
"GLN",
"ASP",
"ASN",
"LYS",
"ARG",
]
token_dict = {item:index+1 for index, item in enumerate(Amino_acid_type)}
# token_dict = {'C':0, 'N':1, 'O':2, 'S':3, 'H':4, 'X':5}

def text_save(file, string):
    with open(file, 'w') as pid:
        pid.write(string)
def text_read(file):
    with open(file, 'r') as pid:
        return pid.read()
class NotDisordered(Select):
    def accept_atom(self, atom):
        return not atom.is_disordered() or atom.get_altloc() == "A"  or atom.get_altloc() == "1"
# Exclude disordered atoms.
def read_PDB(in_pdb_dir, pdb_name, mutated_resid=None,  wild_rescode=None):
    pdb_file = os.path.join(in_pdb_dir, pdb_name+'.pdb')
    try:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure(pdb_file, pdb_file)
        Residues = Selection.unfold_entities(struct, "R")
    except:
        info = (np.empty(shape=(0)), np.empty(shape=(0, 3)), np.empty(shape=(0,3,3)), '', 0)
    else:
        xyz = []
        token = []
        nuv = []
        seq = ''
        mutate_pos=10000
        for index, residue in enumerate(Residues):
            if residue.get_resname() not in Amino_acid_type:
                # print(residue)
                continue
            if mutated_resid != None:
                if str(residue.id[1]) == mutated_resid:
                    mutate_pos = index
                    if seq1(residue.get_resname())== wild_rescode:
                        pass
                    else:
                        print(pdb_name, 'mutated not correct')
                        assert seq1(residue.get_resname()) == wild_rescode
            if 'CA' in residue.child_dict.keys():
            #from EQUIDOCK
                alphaC_loc = residue.child_dict['CA'].coord
                C_loc = residue.child_dict['C'].coord
                if 'N' not in residue.child_dict.keys():
                    print(residue)
                N_loc = residue.child_dict['N'].coord
                u_i = (N_loc - alphaC_loc) / LA.norm(N_loc - alphaC_loc)
                t_i = (C_loc - alphaC_loc) / LA.norm(C_loc - alphaC_loc)
                n_i = np.cross(u_i, t_i) / LA.norm(np.cross(u_i, t_i))
                v_i = np.cross(n_i, u_i)
                assert (math.fabs(LA.norm(v_i) - 1.) < 1e-5), "protein utils protein_to_graph_dips, v_i norm larger than 1"
                token.append(token_dict[residue.get_resname()])
                xyz.append(alphaC_loc)
                nuv.append(np.row_stack([u_i, n_i, v_i]))
                seq += seq1(residue.get_resname())
        if len(token) == 0 or (mutate_pos == 10000 and mutated_resid!=None):
            print('res_id not in PDB')
            info = (np.empty(shape=(0)), np.empty(shape=(0, 3)), np.empty(shape=(0,3,3)), '', 0)
        else:
            mutate_position = np.zeros(len(seq), dtype=float)
            if mutate_pos != 10000:
                mutate_position[mutate_pos] = 1.0
            info = (np.array(token), np.stack(xyz, axis=0), np.stack(nuv, axis=0), seq, mutate_position)

    return info[0], info[1], info[2], info[3]

def extractPDB(infile_dir, outfile_dir, pdbid_chainid, aug=False):
    if not os.path.exists(outfile_dir):
        os.mkdir(outfile_dir)
    pdb_id, chain_id = pdbid_chainid.split('_')

    infilename = os.path.join(infile_dir, pdb_id + '.pdb')
    outfilename = os.path.join(outfile_dir, pdbid_chainid+'.pdb')
    if not os.path.exists(infilename):
        return
    if os.path.exists(outfilename):
        return
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(infilename, infilename)
    model = Selection.unfold_entities(struct, "M")[0]
    chain = model.child_dict[chain_id]

    # Select residues to extract and build new structure
    structBuild = StructureBuilder.StructureBuilder()
    structBuild.init_structure("output")
    structBuild.init_seg(" ")
    structBuild.init_model(0)
    outputStruct = structBuild.get_structure()

    structBuild.init_chain(chain_id[-1])
    for residue in chain:
        if residue.get_resname().upper() not in Amino_acid_type:
            continue
        if 'CA' in residue.child_dict.keys() and 'C' in residue.child_dict.keys() and 'N' in residue.child_dict.keys():
            outputStruct[0][chain_id[-1]].add(residue)

    # Output the selected residues
    pdbio = PDBIO()
    pdbio.set_structure(outputStruct)
    pid = open(outfilename,'w')
    pdbio.save(pid, select=NotDisordered())
    pid.close()


def call_modeller(infile_dir, pdbid_chainid, res_id, mutated_rescode, wild_rescode=None):
    outfilename = os.path.join(infile_dir, '{}.mut.{}_{}.pdb'.format(pdbid_chainid,res_id, mutated_rescode))
    if os.path.exists(outfilename):
        return
    args = ['bash', './bin/run_modeller.sh', '../'+infile_dir,
            '{}_{}'.format(res_id, mutated_rescode), pdbid_chainid, str(res_id),
            seq3(mutated_rescode).upper(), pdbid_chainid.split('_')[1]]
    if not os.path.exists(outfilename):
        p2 = Popen(args, stdout=PIPE, stderr=PIPE)
        print('calling modeller start')
        stdout, stderr = p2.communicate()
        print('calling modeller over')

def call_foldx(infile_dir, pdbid_chainid, res_id, mutated_rescode, wild_rescode):
    pdb_id, mutated_chain = pdbid_chainid.split('_')
    outfilename = os.path.join(infile_dir, '{}.mut.{}_{}.pdb'.format(pdbid_chainid,res_id, mutated_rescode))
    if os.path.exists(outfilename):
        return
    # return
    individual_list = os.path.join(infile_dir, 'individual_list.txt')
    # individual_list = os.path.join(infile_dir, '_'.join([pdb_id, mutated_chain, res_id, mutated_rescode]))
    with open(individual_list, 'w') as pid:
        str_towrite = ''
        wild_rescode = wild_rescode.split('_')
        mutated_rescode = mutated_rescode.split('_')
        res_id = res_id.split('_')
        for index in range(len(wild_rescode)):
            str_towrite += wild_rescode[index]+mutated_chain+res_id[index]+mutated_rescode[index]+','
        str_towrite = str_towrite[:-1]+';'
        wild_rescode = '_'.join(wild_rescode)
        mutated_rescode = '_'.join(mutated_rescode)
        res_id = '_'.join(res_id)
        pid.write(str_towrite)
    args = ['./bin/foldx',
            '--command='+ 'BuildModel',
            '--pdb='+ pdbid_chainid+'.pdb',
            '--pdb-dir='+ infile_dir,

            '--mutant-file='+ individual_list,
            '--output-dir='+ infile_dir,
            '--numberOfRuns='+'3',
            '--rotabaseLocation='+ './bin/rotabase.txt'
            ]

    # if not os.path.exists(outfilename):
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    print('calling modeller start')
    stdout, stderr = p2.communicate()
    print('calling modeller over')
    shutil.move(os.path.join(infile_dir, pdbid_chainid+'_1_2.pdb'), outfilename)
    os.remove(os.path.join(infile_dir, 'WT_'+pdbid_chainid+'_1_2.pdb'))
    os.remove(os.path.join(infile_dir, 'WT_'+pdbid_chainid+'_1_1.pdb'))
    os.remove(os.path.join(infile_dir, 'WT_'+pdbid_chainid+'_1_0.pdb'))
    os.remove(os.path.join(infile_dir, pdbid_chainid+'_1_1.pdb'))
    os.remove(os.path.join(infile_dir, pdbid_chainid+'_1_0.pdb'))
    os.remove(os.path.join(infile_dir, 'Raw_'+pdbid_chainid+'.fxout'))
    os.remove(os.path.join(infile_dir, 'PdbList_'+pdbid_chainid+'.fxout'))
    os.remove(os.path.join(infile_dir, 'Dif_'+pdbid_chainid+'.fxout'))
    os.remove(os.path.join(infile_dir, 'Average_'+pdbid_chainid+'.fxout'))
