# Transformer From Scratch

## Data
Protein sequences and their diheadral angles have gotten from Alphafold predictions, Pisces. \
Masking done for not learning the padding. Max length of sequence 128.

## Architecture structures
- Encoder only
- Embedding space (Prot Bert )
- Attention (single head, self, masked)
- Feed-forward neural network (2 hidden layers lowering dimensions to 2, layer normalization removed because of this)
- Only one layer normalization

- [ ] Code cleaning
- [ ] Code documentation

## Presentation

### Introduction/Background (sandra - mihriban)
- [ ] Problem explanation (alphafold, CASP)
- [ ] Protein structure explanation (folding procedure)
- [ ] Our data (pisces, alphafold predictions)

### Technical Approach 
- [ ] Angles (mihriban)
- [ ] Preparation of data (mihriban)
- [ ] Prot-bert (bert,T5) (mihriban)
- [ ] Transformer architecture/components (mandana)
- [ ] Custom loss (maryam)
- [ ] Masking (mandana)
  
### Results
- [ ] Predictions (sandra - mihriban)
- [ ] Attention weights (before-after training) (mihriban)
- [ ] NN weights (before-after training) (mihriban)
- [ ] Ramachandran
  
### Conclusions
- [ ] Comparison with other articles
- [ ] Alphafold (comparison)
- [ ]

### References


## Code structure

- data_processing: Extracting sequence and angles (clean the code make it more pretty)
- transformer: a file containing only the transformer (clean code maybe name the transformer ?)
- trainer: loading data, loading model and training
- evaluation: only for visualization and evaluation
- training files can be added to github maybe



University of Padova \
Laboratory of Computational Physics Project \
Group 2410 \
Supervisor Prof. Jeff Byers
