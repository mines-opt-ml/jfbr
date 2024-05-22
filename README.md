
## Installation

Follow these steps to set up the project environment:

### Clone the Repository

```bash
git clone https://github.com/yourusername/your-repository.git
cd your-repository
```

### Set Up Virtual Environment

Create and activate a virtual environment:

```bash
python -m venv env
source env/bin/activate  # On Unix/macOS
venv\Scripts\activate  # On Windows
```

### Install the Package

Install the required packages:

```bash
pip install -r requirements.txt
```

## Overview

The `/models` folder contains the following models:

- `base_mon_net`
    - abstract base class for all monotone networks, which are implicit networks formed by repeatedly passing through same monotone layer
    - based on https://github.com/locuslab/monotone_op_net/blob/master/mon.py
- `mon_net_AD`
    - monotone network trained via automatic differentation (AD)
- `mon_net_JFB`
    - monotone network trained via Jacobian free backpropagation (JFB)
    - performs fixed number of iterations then backpropagates through last iteration only
- `mon_net_JFB_R`
    - monotone network trained via Jacobian free backpropagation (JFB), but with random selection of the number of iterations 
- `mon_net_JFB_CSBO`
    - monotone network trained via Jacobian free backpropagation (JFB), but with random selection of the number of iterations and use of the gradient estimator $\hat{v} = \hat{v}_1 + \frac{1}{p_k}(\hat{v}_{k+1}-\hat{v}_k)$
    - inspired by https://arxiv.org/abs/2310.18535