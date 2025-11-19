# HyperDTI-Lite

**HyperDTI-Lite**:The model consists of three components: the feature representation module, the hyperbolic representation learning module, and the prediction module. First, the feature representation module extracts homologous heterogeneous features separately from drug SMILES strings and target sequences in Euclidean space. After concatenating these features, they are mapped into hyperbolic space. The hyperbolic representation learning module then performs representation learning on above hyperbolic features. Finally, the prediction module outputs drug–target interaction predictions based on the fused features.

## HyperDTI-Lite
<div align="center">
<p><img src="fig_model.png" width="800" /></p>
</div>

## Setup and dependencies 
Dependencies:
- python 3.9.16
- pytorch >=1.12
- pyg	2.2.0
- rdkit	2022.9.5
- numpy
- sklearn
- tqdm
- tensorboardX
- prefetch_generator
- matplotlib

## Resources:
+ Key Source Files
  + HyperbolicTriplet.py：
  + hyperparameter.py：
  + model.py：
  + pytorchtools.py：
  + dataset.py
+ Supporting Source Files
  + creat_drug_prefile.py：
  + creat_target_prefile.py：
  + calculate_parameters.py：
  + operate_drug.py：
  + operate_target.py：



# Run:

python SSGraphDTI.py
