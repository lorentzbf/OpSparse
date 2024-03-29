The Source Code of OpSparse
========

This repository contain the source code of OpSparse, and part of the source code from [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html), [nsparse](https://github.com/EBD-CREST/nsparse.git), and [spECK](https://github.com/GPUPeople/spECK.git).
## Tested evironment
CUDA 11.2, NVIDIA Tesla V100 GPU, Ubuntu 18.04 LTS

## Get started
1 Execute ```$> bash download_matrix.sh``` in the current directory to download the matrix webbase-1M into matrix/suite_sparse directory

2 For detailed execution instruction, refer the readme.md in the opsparse, nsparse, and speck sub-directory

## Bibtex
```
@ARTICLE{9851653,
  author={Du, Zhaoyang and Guan, Yijin and Guan, Tianchan and Niu, Dimin and Huang, Linyong and Zheng, Hongzhong and Xie, Yuan},
  journal={IEEE Access}, 
  title={OpSparse: A Highly Optimized Framework for Sparse General Matrix Multiplication on GPUs}, 
  year={2022},
  volume={10},
  number={},
  pages={85960-85974},
  doi={10.1109/ACCESS.2022.3196940}}
  ```
