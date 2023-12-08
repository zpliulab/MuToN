![MuToN](https://github.com/zpliulab/MuToN/blob/main/logo.png)
# MuToN: An end-to-end geometric deep learning framework for predicting binding affinity changes upon protein mutations.

## Requirements
### Python packages
* [BioPython*](https://github.com/biopython/biopython) (v1.78). To deal with computings relating to structures and sequences of proteins.
* [Pytorch*](https://pytorch.org/) (v2.0.1). Pytorch with GPU version. Use to model, train, and evaluate the actual neural networks.
* [scikit-learn*](https://scikit-learn.org/) (v0.24.1). For machine learning relating computation.
* [esm*](https://github.com/facebookresearch/esm). For protein residue embedding.
## Standalone software
* [MODELLER] (https://salilab.org/modeller/). Installing refers to https://salilab.org/modeller/download_installation.html.

## Download data.
To speed up training, download all the pre-computed data, including mutation list, complexes, mutant protein structures, pre-computed llm features.
```
wget https://zenodo.org/record...
unzip PDBs.zip

```
## Training and evaluation.

The project does not contain a standalone script specifically for preprocessing protein structures. However, during the data loading stage, the protein preprocessing routine is executed. To reproduce the experiments in Fig. 2. simply run:
```
python train.py --dataset SKEMPI2 --splitting mutation --checkpoints_dir SKEMPI2_mutation
```
## License
MuToN is released under an [MIT License](LICENSE).
