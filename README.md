# SHANG++

This repository contains the code for the numerical experiments presented in the paper:

**Yaxin Yu, Long Chen, and Minfu Feng.**
*SHANG++: Robust Stochastic Acceleration under Multiplicative Noise.*
**Computers & Mathematics with Applications**, 220:386–402, 2026.
https://doi.org/10.1016/j.camwa.2026.07.033

SHANG++ is an accelerated stochastic optimization method derived from a discretization of the Hessian-driven Nesterov accelerated gradient flow. The experiments in this repository evaluate its convergence, robustness to multiplicative noise, and performance on deep-learning tasks.

## Citation

If you find this repository useful in your research, please cite:

```bibtex
@article{YU2026386,
  title   = {SHANG++: Robust stochastic acceleration under multiplicative noise},
  journal = {Computers \& Mathematics with Applications},
  volume  = {220},
  pages   = {386--402},
  year    = {2026},
  issn    = {0898-1221},
  doi     = {10.1016/j.camwa.2026.07.033},
  url     = {https://www.sciencedirect.com/science/article/pii/S089812212600355X},
  author  = {Yaxin Yu and Long Chen and Minfu Feng},
  keywords = {Stochastic gradient descent, Momentum method, Acceleration, Multiplicative noise scaling, Convergence analysis}
}
```

## Credits and Third-Party Code

Parts of this repository are adapted from the following open-source projects:

* **AGNES** by Gupta et al. — [GitHub repository](https://github.com/kanangupta/AGNES), licensed under the MIT License.
* **cifar10-resnet** by Matthias Wright — [GitHub repository](https://github.com/matthias-wright/cifar10-resnet), licensed under the MIT License.

The corresponding original license and copyright notices are retained where required.

## License

The original code in this repository is released under the [MIT License](LICENSE). Third-party components remain subject to their respective original licenses.
