# -*- coding: utf-8 -*-
"""

"""
from __future__ import division, print_function
import pandas as pd
from glob import glob
from pprint import pprint



def summarize_table(path):
    print('-' * 80)
    print(path)

    # good_i = -1
    # good_line = None
    # try:
    #     with open(path, 'rt') as f:
    #         for i, line in enumerate(f):
    #             good_i, good_line = i, line
    # except:
    #     print('^' * 80)
    #     print('good_i=%d' % good_i)
    #     print('good_line="%s"' % good_line)
    #     raise

    df = pd.read_csv(path, sep='\t', encoding='cp1252')
    print(df.shape)
    print(df.columns)
    print(df.describe())
    col_desc = {col: [len(df[col]), len(set(df[col])),
                      int(round(100 * len(set(df[col])) / len(df[col]) ))]
                      for col in df.columns}
    return col_desc


tables = list(glob('sneak/*.csv'))

pprint(tables)

col_desc = {}
for path in tables:
    col_desc[path] = summarize_table(path)
pprint(col_desc)

