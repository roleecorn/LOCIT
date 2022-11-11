# LocIT

## This repository is a practice work clone from https://github.com/Vincent-Vercruyssen/LocIT

This repository contains the online supplement of the 2020 AAAI paper **Transfer Learning for Anomaly Detection through Localized and Unsupervised Instance Selection**. It contains: *the appendix*, *the code for the experiments*, and *the benchmark data*.

The paper authors are from the [DTAI group](https://dtai.cs.kuleuven.be/) of the [KU Leuven](https://kuleuven.be/):

- [Vincent Vercruyssen](https://people.cs.kuleuven.be/~vincent.vercruyssen/)
- [Wannes Meert](https://people.cs.kuleuven.be/~wannes.meert/)
- [Jesse Davis](https://people.cs.kuleuven.be/~jesse.davis/)


## Abstract

> *Anomaly detection attempts to identify instances that deviate from expected behavior. Constructing performant anomaly detectors on real-world problems often requires some labeled data, which can be difficult and costly to obtain. However, often one considers multiple, related anomaly detection tasks. Therefore, it may be possible to transfer labeled instances from a related anomaly detection task to the problem at hand. This paper proposes a novel transfer learning algorithm for anomaly detection that selects and transfers relevant labeled instances from a source anomaly detection task to a target one. Then, it classifies target instances using a novel semi-supervised nearest-neighbors technique that considers both unlabeled target and transferred, labeled source instances. The algorithm outperforms a multitude of state-of-the-art transfer learning methods and unsupervised anomaly detection methods on a large benchmark. Furthermore, it outperforms its rivals on a real-world task of detecting anomalous water usage in retail stores.*

In short, the paper tackles the following task:

```java
GIVEN:  a (partially) labeled source dataset Ds and
        an unlabeled target dataset Dt from the same feature space;

DO:     assign an anomaly score to each instance in Dt
        using both Dt and a subset of Ds.
```

The **appendix to the paper** (and the full conference paper) can either be accessed in `LocIT/paper/` or via the [webpage](https://people.cs.kuleuven.be/~vincent.vercruyssen/).


## Code and data

The Python code for the full **LocIT** algorithm and some of the baselines used in the paper, is in the folder: `LocIT/models/`

The zipped benchmark datasets and the script to construct them are in the folder: `LocIT/data/`


#### *DISCLAIMER*:

The core contributions of our paper are the **LocIT** *transfer learning* algorithm and the **SSkNNO** *semi-supervised anomaly detection* algorithm. Although the source code of these algorithms is also available in this repository for completeness, the recommended way to use these algorithms in your own work is to import them from the publicly available and *pip-installable* [*transfertools*](https://github.com/Vincent-Vercruyssen/transfertools) and [*anomatools*](https://github.com/Vincent-Vercruyssen/anomatools) packages.
```bash
pip install transfertools
pip install anomatools
```
Once installed, the models can be used as follows:
```python
from transfertools.models import LocIT
from anomatools.models import SSkNNO
```


## Interesting links

Implementations of the remaining baselines used in the paper can be found here:

algorithm | link
--- | ---
CBIT | [transfertools](https://github.com/Vincent-Vercruyssen/transfertools)
TCA | [transfertools](https://github.com/Vincent-Vercruyssen/transfertools)
GFK | [link](https://github.com/jindongwang/transferlearning/tree/master/code)
JDA | [link](https://github.com/jindongwang/transferlearning/tree/master/code)
TJM | [link](https://github.com/jindongwang/transferlearning/tree/master/code)
JGSA | [link](https://github.com/jindongwang/transferlearning/tree/master/code)
LOF | [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
HBOS | [HBOS-python](https://github.com/Kanatoko/HBOS-python/blob/master/hbos.py)


## Contact

Feel free to ask questions: [vincent.vercruyssen@kuleuven.be](mailto:vincent.vercruyssen@kuleuven.be)


## Citing the AAAI paper

```
@inproceedings{vercruyssen2020transfer,
    title       = {Transfer Learning for Anomaly Detection through Localized and Unsupervised Instance Selection},
    author      = {Vercruyssen, Vincent and Meert, Wannes and Davis Jesse},
    booktitle   = {Thirty-Fourth AAAI Conference on Artificial Intelligence},
    year        = {2020}
}
```