import sys, shutil
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
    # if not os.path.exists(outfilename):
    #     return
    build_model(infile_dir, '{}_{}'.format(res_id, mutated_rescode),
                pdbid_chainid, res_id, seq3(mutated_rescode).upper(), pdbid_chainid.split('_')[1])

def call_foldx(infile_dir, pdbid_chainid, res_id, mutated_rescode, wild_rescode):
    pdb_id, mutated_chain = pdbid_chainid.split('_')
    outfilename = os.path.join(infile_dir, '{}.mut.{}_{}.pdb'.format(pdbid_chainid,res_id, mutated_rescode))
    if os.path.exists(outfilename):
        return
    # return
    # individual_list = os.path.join(infile_dir, 'individual_list.txt')
    individual_list = os.path.join(infile_dir, 'individual_list_'+'_'.join([pdb_id, mutated_chain, res_id, mutated_rescode])+'.txt')
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
    print('calling foldx start')
    stdout, stderr = p2.communicate()
    print('calling foldx over')
    shutil.move(os.path.join(infile_dir, pdbid_chainid+'_1_2.pdb'), outfilename)
    os.remove(os.path.join(infile_dir, 'individual_list_'+pdb_id+'_'+mutated_chain+'_'+res_id+'_'+mutated_rescode+'.txt'))
    os.remove(os.path.join(infile_dir, 'WT_'+pdbid_chainid+'_1_2.pdb'))
    os.remove(os.path.join(infile_dir, 'WT_'+pdbid_chainid+'_1_1.pdb'))
    os.remove(os.path.join(infile_dir, 'WT_'+pdbid_chainid+'_1_0.pdb'))
    os.remove(os.path.join(infile_dir, pdbid_chainid+'_1_1.pdb'))
    os.remove(os.path.join(infile_dir, pdbid_chainid+'_1_0.pdb'))
    os.remove(os.path.join(infile_dir, 'Raw_'+pdbid_chainid+'.fxout'))
    os.remove(os.path.join(infile_dir, 'PdbList_'+pdbid_chainid+'.fxout'))
    os.remove(os.path.join(infile_dir, 'Dif_'+pdbid_chainid+'.fxout'))
    os.remove(os.path.join(infile_dir, 'Average_'+pdbid_chainid+'.fxout'))

# The following lines are used to call modeller to build mutate model
# Modified from https://github.com/NRGlab/ENCoM/blob/15cc2d61cecfa9b84eb2ed5f8e966771a0cc4b81/tutorial/mutate_model.py
#  Creates a single in silico point mutation to sidechain type and at residue position
#  input by the user, in the structure whose file is modelname.pdb
#  The conformation of the mutant sidechain is optimized by conjugate gradient and
#  refined using some MD.
#
#  Note: if the model has no chain identifier, specify "" for the chain argument.
import os
from modeller import *
from modeller.optimizers import molecular_dynamics, conjugate_gradients
from modeller.automodel import autosched

def optimize(atmsel, sched):
    #conjugate gradient
    for step in sched:
        step.optimize(atmsel, max_iterations=200, min_atom_shift=0.001)
    #md
    refine(atmsel)
    cg = conjugate_gradients()
    cg.optimize(atmsel, max_iterations=200, min_atom_shift=0.001)

#molecular dynamics
def refine(atmsel):
    # at T=1000, max_atom_shift for 4fs is cca 0.15 A.
    md = molecular_dynamics(cap_atom_shift=0.39, md_time_step=4.0,
                            md_return='FINAL')
    init_vel = True
    for (its, equil, temps) in ((200, 20, (150.0, 250.0, 400.0, 700.0, 1000.0)),
                                (200, 600,
                                 (1000.0, 800.0, 600.0, 500.0, 400.0, 300.0))):
        for temp in temps:
            md.optimize(atmsel, init_velocities=init_vel, temperature=temp,
                         max_iterations=its, equilibrate=equil)
            init_vel = False

#use homologs and dihedral library for dihedral angle restraints
def make_restraints(mdl1, aln):
   rsr = mdl1.restraints
   rsr.clear()
   s = selection(mdl1)
   for typ in ('stereo', 'phi-psi_binormal'):
       rsr.make(s, restraint_type=typ, aln=aln, spline_on_site=True)
   for typ in ('omega', 'chi1', 'chi2', 'chi3', 'chi4'):
       rsr.make(s, restraint_type=typ+'_dihedral', spline_range=4.0,
                spline_dx=0.3, spline_min_points = 5, aln=aln,
                spline_on_site=True)

#first argument

def build_model(file_dir, outname, modelname, respos, restyp, chain):
    __console__ = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    log.verbose()

    # Set a different value for rand_seed to get a different final model
    env = environ(rand_seed=-49837)

    env.io.hetatm = True
    #soft sphere potential
    env.edat.dynamic_sphere=False
    #lennard-jones potential (more accurate)
    env.edat.dynamic_lennard=True
    env.edat.contact_shell = 4.0
    env.edat.update_dynamic = 0.39

    # Read customized topology file with phosphoserines (or standard one)
    env.libs.topology.read(file='$(LIB)/top_heav.lib')

    # Read customized CHARMM parameter library with phosphoserines (or standard one)
    env.libs.parameters.read(file='$(LIB)/par.lib')


    # Read the original PDB file and copy its sequence to the alignment array:
    mdl1 = model(env, file=file_dir+modelname)
    ali = alignment(env)
    ali.append_model(mdl1, atom_files=modelname, align_codes=modelname)

    #set up the mutate residue selection segment
    s = selection(mdl1.chains[chain].residues[respos])

    #perform the mutate residue operation
    s.mutate(residue_type=restyp)
    #get two copies of the sequence.  A modeller trick to get things set up
    ali.append_model(mdl1, align_codes=modelname)

    # Generate molecular topology for mutant
    mdl1.clear_topology()
    mdl1.generate_topology(ali[-1])


    # Transfer all the coordinates you can from the template native structure
    # to the mutant (this works even if the order of atoms in the native PDB
    # file is not standard):
    #here we are generating the model by reading the template coordinates
    mdl1.transfer_xyz(ali)

    # Build the remaining unknown coordinates
    mdl1.build(initialize_xyz=False, build_method='INTERNAL_COORDINATES')

    #yes model2 is the same file as model1.  It's a modeller trick.
    mdl2 = model(env, file=file_dir+modelname)

    #required to do a transfer_res_numb
    #ali.append_model(mdl2, atom_files=modelname, align_codes=modelname)
    #transfers from "model 2" to "model 1"
    mdl1.res_num_from(mdl2,ali)

    #It is usually necessary to write the mutated sequence out and read it in
    #before proceeding, because not all sequence related information about MODEL
    #is changed by this command (e.g., internal coordinates, charges, and atom
    #types and radii are not updated).

    mdl1.write(file=file_dir+modelname+restyp+respos+'.tmp')
    mdl1.read(file=file_dir+modelname+restyp+respos+'.tmp')

    #set up restraints before computing energy
    #we do this a second time because the model has been written out and read in,
    #clearing the previously set restraints
    make_restraints(mdl1, ali)

    #a non-bonded pair has to have at least as many selected atoms
    mdl1.env.edat.nonbonded_sel_atoms=1

    sched = autosched.loop.make_for_model(mdl1)

    #only optimize the selected residue (in first pass, just atoms in selected
    #residue, in second pass, include nonbonded neighboring atoms)
    #set up the mutate residue selection segment
    s = selection(mdl1.chains[chain].residues[respos])

    mdl1.restraints.unpick_all()
    mdl1.restraints.pick(s)

    s.energy()

    s.randomize_xyz(deviation=4.0)

    mdl1.env.edat.nonbonded_sel_atoms=2
    optimize(s, sched)

    #feels environment (energy computed on pairs that have at least one member
    #in the selected)
    mdl1.env.edat.nonbonded_sel_atoms=1
    optimize(s, sched)

    s.energy()

    #give a proper name
    mdl1.write(file=file_dir+modelname+".mut."+outname+'.pdb')
    sys.stdout = __console__