# "Probing Quantum Spin Systems with Kolmogorov-Arnold Neural Network Quantum States"
## by Mahmud Ashraf Shamim, Eric A F Reinhardt, Talal Ahmed Chowdhury, Sergei Gleyzer, Paulo T Araujo

This code was used to produce results in this published work https://arxiv.org/abs/2506.01891.

Please cite as:
```
@misc{shamim2025probingquantumspinsystems,
      title={Probing Quantum Spin Systems with Kolmogorov-Arnold Neural Network Quantum States}, 
      author={Mahmud Ashraf Shamim and Eric A F Reinhardt and Talal Ahmed Chowdhury and Sergei Gleyzer and Paulo T Araujo},
      year={2025},
      eprint={2506.01891},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2506.01891}, 
}
```


An example use of this code can be shown in J1J2_Training_Example.ipynb.

Example run script:
`!python3 vmc.py --ham j1j2_1d --boundary peri --sign mars --J2 0.0 --ham_dim 1 --L 100 --zero_mag --net sym_sinekan --layers_hidden 64,64,1 --grid_size 8 --seed 123 --optimizer custom --drop_step 30_000 --decay_time 1_000 --max_step 34_000 --show_progress --lr 1e-3`

This software is derived from the varbench package at https://github.com/varbench/varbench. Modifications to the code introduced in this work include introduction of demo ipython notebooks for training additional models and generating additional plots as well as introduction of new models.
