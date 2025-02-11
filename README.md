# Functional Connectivity Integrative Normative Modelling (FUNCOIN)

## Dependencies:
The funcoin module requires the following python modules:
* numpy
* scipy
* scikit-learn (some model evaluation metrics are only available in version 1.4 or higher, but funcoin can run with older versions)
* matplotlib (for the tutorial)
* notebook (for the tutorial)

## Installation
* Install the latest stable build by using the following command  
```
pip install fcin
```
* Install the latest development version by using the following command  
```
pip install git+https://github.com/kobbersmed/funcoin
```

## Links
* Source code repository: https://github.com/kobbersmed/funcoin
* Tutorial notebook: https://github.com/kobbersmed/funcoin/blob/main/notebooks/Tutorial_funcoin.ipynb

## References
* The python module is released with the following paper, which we ask you to cite: 
Kobbersmed, J.R.L., Gohil, C., Marquand, A.F. and Vidaurre, D., _One-shot normative modelling of whole-brain functional connectivity_, _bioRxiv_, 2024. [https://www.biorxiv.org/content/10.1101/2025.01.13.632752v1]
* Please also cite the original paper proposing the regression method: 
Zhao, Y. et al. (2021). "Covariate Assisted Principal regression for covariance matrix outcomes", _Biostatistics, 22_(3), pp. 629-45. [https://doi.org/10.1093/biostatistics/kxz057]

## Contact
* Author and maintainer of this Python package: Janus Rønn Lind Kobbersmed, mail: janus@cfin.au.dk or januslind@gmail.com
