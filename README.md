<img src="https://github.com/zpliulab/MuToN/blob/main/logo.png" width=512/>

## MuToN: An end-to-end geometric deep learning framework for predicting binding affinity changes upon protein mutations.
## Requirements
* [BioPython*](https://github.com/biopython/biopython) (v1.78). To deal with computings relating to structures and sequences of proteins.
* [Pytorch*](https://pytorch.org/) (v2.0.1). Pytorch with GPU version. Use to model, train, and evaluate the actual neural networks.
* [Scikit-learn*](https://scikit-learn.org/) (v0.24.1). For machine learning relating computation.
* [ESM*](https://github.com/facebookresearch/esm). For protein residue embedding.
* [Modeller*](https://salilab.org/modeller/). For mutant protein structure modeling.  
Installing refers to https://salilab.org/modeller/download_installation.html.  
You need a license key to use Modeller first.  
An easy way is to install Modeller using the 'conda' package manager, simply run from a command line:  
conda config --add channels salilab  
conda install modeller  
And then Edit the file config.py according to promption after installation by replacing XXXX with your Modeller license key.

## Download data.
We open sourced all the precomputed data using to reproduce the results of our paper https://zenodo.org/records/10445253.  
This repository includes the lists of mutation records, complexes, mutant protein structures and files of pre-computed LLM embeddings.
```
wget https://zenodo.org/records/10445253/files/data.zip 
unzip data.zip
```
## Training and evaluation.

The project does not contain a standalone script specifically for preprocessing protein structures. 
However, during the data loading stage, the protein preprocessing routine is executed. 

**Usage description**:  
--checkpoints_dir. #Default is Checkpoints/example. Specify a directory to save the model checkpoints.  
--dataset SKEMPI2 or S1131 or S4169 or M1101. #Default is S1131. Specify which dataset to use.  
--splitting mutation or complex. #Default is mutation. Specify the Train-Test partitioning mode of the dataset. Mutation-level or complex-level.  
--device cuda:0 or cpu, etc. #Default is cuda:0. Specifies the device to run the model on.  
**For example**, to train and evaluate the model on the SKEMPI2 dataset using the mutation-level splitting mode, run the following command:  
```
python train.py --checkpoints_dir SKEMPI2_mutation --dataset SKEMPI2 --splitting mutation --device cuda:0
```

## License
MuToN is released under an [MIT License](LICENSE).

MuToN quantifies binding affinity changes upon protein mutations by geometric deep learning.