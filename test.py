import numpy as np
import pandas as pd
import os
import argparse
from typing import Tuple


def _load_and_preprocess_data(data_path) -> Tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], ]:
    """ Load and preprocess the data. """

    src_path = os.path.join(data_path, 'source')
    tgt_path = os.path.join(data_path, 'target')

    # source files
    source_files = [f for f in os.listdir(
        src_path) if os.path.isfile(os.path.join(src_path, f))]
    # print(source_files)
    source_files = [os.path.join(src_path, f)
                    for f in source_files if '.csv' in f]
    # print(source_files)
    # target files
    target_files = [f for f in os.listdir(
        tgt_path) if os.path.isfile(os.path.join(tgt_path, f))]
    target_files = [os.path.join(tgt_path, f)
                    for f in target_files if '.csv' in f]

    # load the data
    source_sets = dict()
    for sf in source_files:
        data = pd.read_csv(sf, sep=',', index_col=0).sample(
            frac=1).reset_index(drop=True)
        file_name = os.path.split(sf)[1].split('.csv')[0]
        source_sets[file_name] = data

    target_sets = dict()
    for sf in target_files:
        data = pd.read_csv(sf, sep=',', index_col=0).sample(
            frac=1).reset_index(drop=True)
        file_name = os.path.split(sf)[1].split('.csv')[0]
        target_sets[file_name] = data
    print(source_sets.keys())
    print(target_sets.keys())
    print(target_sets['shuttle_v9'])
    print(type(target_sets['shuttle_v9']))
    return source_sets, target_sets


parser = argparse.ArgumentParser(
    description='Run transfer learning - anomaly detection experiment')
parser.add_argument('-d', '--dataset', type=str, default='shuttle',
                    help='dataset = folder in data/ directory')
parser.add_argument('-m', '--method', type=str,
                    default='locit', help='method to use,default \'locit\'')
args, unknownargs = parser.parse_known_args()
main_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(main_path, 'data', args.dataset)
source_sets, target_sets = _load_and_preprocess_data(data_path)
for tgt_name, target_data in target_sets.items():

    Xt = target_data.iloc[:, :-1].values
    yt = np.zeros(Xt.shape[0])
