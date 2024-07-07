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

## Training






University of Padova \
Laboratory of Computational Physics Project \
Group 2410 \
Supervisor Prof. Jeff Byers
