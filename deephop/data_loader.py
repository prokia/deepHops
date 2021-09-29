# --*-- coding: utf-8 --*--

import csv
from typing import List

import pandas as pd
import rdkit.Chem.AllChem as Chem


def load_sdf_data(sdf_file):
    mols = Chem.SDMolSupplier(sdf_file)
    mols = [m for m in mols if m is not None]
    if len(mols) == 0:
        return None
    colums = list(mols[0].GetPropsAsDict().keys())
    colums.sort()
    colums.append('smiles')
    data_list = []
    for m in mols:
        kv = m.GetPropsAsDict()
        if 'smiles' not in kv.keys():
            kv['smiles'] = Chem.MolToSmiles(m)
        data_list.append([kv[k] for k in colums])

    return pd.DataFrame(data_list, columns=colums)


def detect_delimiter(source_file):
    with open(source_file) as r:
        first_line = r.readline()
        if '\t' in first_line:
            return '\t'
        if ',' in first_line:
            return ','
    return ' '


def has_header(head_line: List[str]):
    for s in head_line:
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                return False
        except:
            continue
    return True


def get_csv_header(path: str) -> List[str]:
    """
    Returns the header of a data CSV file.

    :param path: Path to a CSV file.
    :return: A list of strings containing the strings in the comma-separated header.
    """
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def get_xls_header(path: str) -> List[str]:
    pass


def load_data_frame(source_file) -> object:
    if source_file.endswith('csv') or source_file.endswith('txt') or source_file.endswith('smi'):
        df = pd.read_csv(source_file, delimiter=detect_delimiter(source_file))
    elif source_file.endswith('xls') or source_file.endswith('xlsx'):
        df = pd.read_excel(source_file)
    elif source_file.endswith('sdf'):
        df = load_sdf_data(source_file)
    else:
        print("can not read %s" % source_file)
        df = None
    return df


if __name__ == '__main__':
    pass
