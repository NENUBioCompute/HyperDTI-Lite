# HyperDTI-Lite

## Overview
<div align="center">
<p><img src="fig_model.png" width="800" /></p>
</div>

**HyperDTI-Lite**: The model consists of three components: the feature representation module, the hyperbolic representation learning module, and the prediction module. First, the feature representation module extracts homologous heterogeneous features separately from drug SMILES strings and target sequences in Euclidean space. After concatenating these features, they are mapped into hyperbolic space. The hyperbolic representation learning module then performs representation learning on above hyperbolic features. Finally, the prediction module outputs drug–target interaction predictions based on the fused features.

## Data
- get from the ./Dataset folder
   - DrugBank is available.
- get from Link: https://pan.baidu.com/s/1vGt320EY2EnnnIdD6qb4cA?pwd=85ej
   - Extraction code: 85ej
   - DrugBank is available for download.
  
## Code Approach
Since generating features with the pre-trained model takes a long time, all initial features are pre-generated and saved as .json files in the ./pre_files/ directory. They can then be directly loaded to produce the features shown in the figure above: F<sub>d</sub>, L<sub>d</sub>, P<sub>d</sub>, P<sub>t</sub>, L<sub>t</sub>, and K<sub>t</sub>. The pre-trained language model files can be downloaded from the following link. 
+ Pre-trained LLM
  + Link: https://pan.baidu.com/s/153kgwV5CmaHAyiODwa2D5Q?pwd=cnzw
  + Extraction code: cnzw
+ Download the local_model folder and put it in the root directory ./.

## Resources
+ Key Source Files
  + HyperbolicTriplet.py：Main File.
  + hyperparameter.py：Hyperparameter configuration file.
  + model.py：Model definition file.
  + pytorchtools.py：Early stopping and loss function definition.
  + dataset.py: Definition of the Dataset and Dataloader.
  + utils.py: Associated Functions.
+ Supporting Source Files
  + creat_drug_prefile.py：Pre-generate drug feature .json files for subsequent reading.
  + creat_target_prefile.py：Pre-generate target feature .json files for subsequent reading.
  + calculate_parameters.py：Count the learnable parameters of the model.
  + operate_drug.py：Preprocessing of Drug Inputs.
  + operate_target.py：Preprocessing of Target Inputs.
  + EuclideanDTI.py：Main Script for Manifold Ablation Experiment.
  + model_euc.py: The Euclidean variant of the model for manifold ablation studies.
  + hyperbolic_linear.py：Self-implemented hyperbolic linear layer.
  + Visualization.py: Visualization Script.

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


# Run

python HyperbolicTriplet.py
