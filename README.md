# Domain Generalization via Inference-time Label-Preserving Target Projections
Code for replicating experiments from the paper "Domain Generalization via Inference-time Label-Preserving Target Projections", set to appear at CVPR 2021.

Authors: Prashant Pandey, Mrigank Raman*, Sumanth Varambally*, Prathosh AP.

(* denotes equal contribution)

# Requirements

- Python 3.6.10 
- PyTorch version 1.6.0 
- CUDA version 10.1 
- 4 NVIDIA® Tesla® V100(16 GB Memory) GPUs. 

## Usage
Train {dataset}_{backbone}_FNet.py using source domains to get domain agnostic representations
```
python {dataset}_{backbone}_FNet.py
```

Learn generative model on features from FNet and perform Target projections
```
python {dataset}_{backbone}_Gphi_projection.py
```

To do Nearest Neighbor search on VLCS with FNet features
```
python vlcs_1NN_Sampler.py
```




For clarifications, contact [Prashant Pandey](mailto:getprashant57@gmail.com)
