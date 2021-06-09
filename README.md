# Domain Generalization via Inference-time Label-Preserving Target Projections
Code for the CVPR 2021 Oral paper "Domain Generalization via Inference-time Label-Preserving Target Projections".
The proposed method introduces a new way to handle Domain Generalization problem as compared to the traditional methods. It uses test-time
optimization to optimize the target features and projects them in the source manifold. 

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

## Citation

If you find our work useful, please consider citing our paper.

```
@article{pandey2021domain,
  title={Domain Generalization via Inference-time Label-Preserving Target Projections},
  author={Pandey, Prashant and Raman, Mrigank and Varambally, Sumanth and AP, Prathosh},
  journal={arXiv preprint arXiv:2103.01134},
  year={2021}
}
```


For clarifications, contact [Prashant Pandey](mailto:getprashant57@gmail.com)
