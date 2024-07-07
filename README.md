# Transformer From Scratch

## Data
Protein sequences and their diheadral angles have gotten from Alphafold predictions, Pisces. \
Masking done for not learning the padding. Max length of sequence 128.

## Architecture structures
- Encoder only
- Embedding space (Prot Bert )
- Attention (single head, self, unmasked)
- Feed-forward neural network (2 hidden layers lowering dimensions to 2, layer normalizzation removed because of this)
- Only one layer normalization

## Presentation

### Introduction/Background
- [ ] Problem explanation (alphafold, CASP)
- [ ] Protein structure explanation (folding procedure)
- [ ] Our data (pisces, alphafold predictions)

### Technical Approach 
- [ ] Angles
- [ ] Preparation of data
- [ ] Prot-bert (bert,T5)
- [ ] Transformer architecture/components
- [ ] Custom loss
- [ ] Masking
  
### Results
- [ ] Predictions
- [ ] Attention weights
- [ ] NN weights
  
### Conclusions
- [ ]
- [ ]
- [ ]

### References





University of Padova \
Laboratory of Computational Physics Project \
Group 2410 \
Supervisor Prof. Jeff Byers
