![MuToN](https://github.com/zpliulab/MuToN/blob/main/logo.png)
## MuToN: An end-to-end geometric deep learning framework for predicting binding affinity changes upon protein mutations.

## Python packages
* [BioPython*](https://github.com/biopython/biopython) (v1.78). To deal with computings relating to structures and sequences of proteins.
* [Pytorch*](https://pytorch.org/) (v2.0.1). Pytorch with GPU version. Use to model, train, and evaluate the actual neural networks.
* [scikit-learn*](https://scikit-learn.org/) (v0.24.1). For machine learning relating computation.
* [esm*] (https://github.com/facebookresearch/esm). For protein residue embedding.
## Standalone software
* [MODELLER] (https://salilab.org/modeller/). For modeling of mutant protein three-dimensional structures.

### 2. Training an evaluation.
This section is meant to introduce how to reproduce the results in Fig. 2, which involves conducting ten-fold cross-validation on the SKEMPI V2.0 dataset using MuToN.
The project does not contain a standalone script specifically for preprocessing protein structures. However, during the data loading stage before model training, the protein preprocessing routine is executed.
So just run:
```
python train.py
```
## License
MuToN is released under an [MIT License](LICENSE).
