# -*- coding: utf-8 -*-
"""

"""
from __future__ import division, print_function
import re
import numpy as np
import pandas as pd
from pandas import DataFrame
from glob import glob
from pprint import pprint
from sklearn import cross_validation, utils
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


def null_func(*args):
    pass


CV = 2
N_SCORE_RUNS = 1
RANDOM_STATE = 112

# !@#$ TODO: Force callers to add random_state
def get_score(clf, X, y, cv=CV, n_runs=N_SCORE_RUNS, scoring='accuracy', random_state=RANDOM_STATE,
    verbose=False, n_jobs=-1):
    """Calculate CV scores for classfier clf N_SCORE_RUNS times and print the
        results
    """
    vprint = print if verbose else null_func

    X_in, y_in = X, y

    all_scores = np.empty(n_runs * cv)
    score_list = []
    for i in range(n_runs):
        # print((X_in.shape, y_in.shape, random_state))
        X, y = utils.shuffle(X_in, y_in, random_state=random_state + i)
        if 'random_state' in clf.get_params().keys():
            clf.set_params(random_state=2 * random_state + i)
        try:
            scores = cross_validation.cross_val_score(clf, X, y, cv=cv, scoring=scoring,
                                                      n_jobs=n_jobs)
        except:
            print(clf)
            print((X.shape, y.shape))
            raise
        score_list.append((scores.mean(), scores.std() * 2))
        all_scores[i * cv:(i + 1) * cv] = scores
        vprint('%2d: %6.3f +/-%6.3f' % (i + 1, scores.mean(), scores.std() * 2))

    if verbose:
        min_score = max(0.0, max(u - d for u, d in score_list))
        max_score = min(1.0, min(u + d for u, d in score_list))

        vprint('')
        vprint('%6.3f min' % min_score)
        for u, d in sorted(score_list):
            vprint('%6.3f' % u)
        vprint('%6.3f max\n' % max_score)
        vprint('%6.3f mean\n' % all_scores.mean())

    return all_scores.mean()


def getDummy(df_all, df, col):
    """For column `col` in DataFrame `df`
    """
    category_values = sorted(df_all[col].unique())
    data = np.zeros((len(df), len(category_values)), dtype=int)
    dic_category = {str(val): i for i, val in enumerate(category_values)}

    for i in range(len(df)):
        a = df[col].iloc[i]
        s = str(a)
        j = dic_category[s]
        data[i, j] = 1

    df = df.loc[:, [c for c in df.columns if c != col]]
    for i, val in enumerate(category_values):
        df.loc[:, '%s_%s' % (col, val)] = data[:, i]

    return df


def summarize_table(path):
    print('-' * 80)
    print(path)

    good_i = -1
    good_line = None
    try:
        with open(path, 'rb') as f:
            for i, line in enumerate(f):
                good_i, good_line = i, line
                decoded = good_line.decode('latin-1')
    except:
        print('^' * 80)
        print('good_i=%d' % good_i)
        print('good_line="%s"' % good_line)
        for i, c in enumerate(good_line[:140]):
            print('%3d: %02x "%s"' % (i, c, chr(c)))
        raise

    df = pd.read_csv(path, sep='\t', encoding='latin-1')
    print(df.shape)
    print(df.columns)
    print({col: df[col].dtype for col in df.columns})
    print(df.iloc[:2, :])
    print(df.describe())
    col_desc = {col: [len(df[col]), len(set(df[col])),
                      int(round(100 * len(set(df[col])) / len(df[col]) ))]
                      for col in df.columns}
    return col_desc


if False:
    tables = list(glob('sneak/*.csv'))
    tables = list(glob('small/*.csv'))
    tables = list(glob('all/jobs_all.csv'))

    pprint(tables)

    col_desc = {}
    for path in tables:
        col_desc[path] = summarize_table(path)
    pprint(col_desc)
    assert False


def get_data(path):

    df = pd.read_csv(path, sep='\t',
         # encoding='cp1250'
         encoding='latin-1'
        )
    print(df[['job_id', 'hat']].describe())
    df.set_index('job_id', inplace=True)
    df.sort_index(inplace=True)

    print('get_data: path=%s' % path)
    print(df.columns)
    print('df ', df.shape)

    print(df[['hat']].iloc[:20])
    print(df.index[:20])
    print(df[['hat']].iloc[-20:])
    print(df.index[-20:])

    # assert False

    print('df.index[:10]')
    print(df.index[:10])
    # print('df[job_id][:10]')
    # print(df['job_id'][:10])
    # assert False
    return df


def remove_nulls(df, columns):
    print('remove_nulls: columns=%s' % columns)
    for col in columns:
        x = df[col]
        n_null = sum(x.isnull())
        df = df[x.notnull()]
        print(col, df.shape, x.dtype, n_null)

    return df


def split_train_test(df):
    labelled_rows = df['hat'].values != -1
    df_train = df.iloc[labelled_rows, :]
    df_test = df.iloc[~labelled_rows, :]
    print('df      : %s ', list(df.shape))
    print('df_train: %s %.2f' % (list(df_train.shape), len(df_train) / len(df)))
    print('df_test: %s %.2f' % (list(df_test.shape), len(df_test) / len(df)))
    assert len(df_train) + len(df_test) == len(df)
    assert all(df_train['hat'] != -1)
    assert all(df_test['hat'] == -1)
    # print(df_train.index[:10])
    # print(df_test.index[:10])
    # assert False
    return df_train, df_test


def getXy(df, x_cols):
    X = df[x_cols]
    y = df['hat']
    return X, y


def exec_model(X, y, X_test, out_path, do_score=True):
    print(X.describe())
    print(y.describe())
    y = DataFrame(y, columns=['hat'])
    X.to_csv('%s.X_train.csv' % out_path)
    y.to_csv('%s.y_train.csv' % out_path, index_label='job_id')
    print(y.columns)

    if do_score:
        clf = ExtraTreesClassifier()
        score = get_score(clf, X, y.values.ravel(), cv=2, verbose=True)
        print('score=%f' % score)

    clf = ExtraTreesClassifier()
    clf.fit(X, y.values.ravel())
    y_test = clf.predict(X_test)
    y_test = DataFrame(y_test, columns=['hat'], index=X_test.index)
    print('X_test.index', X_test.index)
    print('y_test.index', y_test.index)
    # y_test.to_csv('blahty.csv', index_label='job_id')
    y_test.to_csv('%s.y_test.csv' % out_path, index_label='job_id')


def build_model001(df):

    x_cols = ['salary_min', 'subclasses']

    df = remove_nulls(df, x_cols + ['hat'])

    df_train, df_test = split_train_test(df)

    X, y = getXy(df_train, x_cols)
    X_test, _ = getXy(df_test, x_cols)

    X_all = pd.concat([X, X_test])

    X = getDummy(X_all, X, 'subclasses')
    X_test = getDummy(X_all, X_test, 'subclasses')

    return X, y, X_test


def build_model002(df):

    x_cols = ['salary_min', 'salary_max']

    df = remove_nulls(df, x_cols + ['hat'])

    df_train, df_test = split_train_test(df)

    X, y = getXy(df_train, x_cols)
    X_test, _ = getXy(df_test, x_cols)

    # X_all = pd.concat([X, X_test])

    # X = getDummy(X_all, X, 'subclasses')
    # X_test = getDummy(X_all, X_test, 'subclasses')

    return X, y, X_test


def build_model003(df):

    x_cols = ['salary_max', 'subclasses']

    df = remove_nulls(df, x_cols + ['hat'])

    df_train, df_test = split_train_test(df)

    X, y = getXy(df_train, x_cols)
    X_test, _ = getXy(df_test, x_cols)

    X_all = pd.concat([X, X_test])

    X = getDummy(X_all, X, 'subclasses')
    X_test = getDummy(X_all, X_test, 'subclasses')

    return X, y, X_test


RE_MODEL = re.compile(r'model(\d+)\.')


def model_num(path):
    m = RE_MODEL.search(path)
    assert m, path
    return int(m.group(1))


def combine_models():

    model_nums = [1, 2, 3]
    # model_nums = [2, 3]
    model_paths = ['model%03d.y_test.csv' % i for i in model_nums]
    # y1 = pd.read_csv('model001.y_test.csv').set_index('job_id')
    # y2 = pd.read_csv('model002.y_test.csv').set_index('job_id')
    # y3 = pd.read_csv('model003.y_test.csv').set_index('job_id')

    models = [pd.read_csv(path).set_index('job_id') for path in model_paths]

    path = 'all/jobs_all.csv'
    df = get_data(path)
    df_train, df_test = split_train_test(df)
    y_data = np.ones((len(df_test), len(models)), dtype=int) * -1
    y = DataFrame(y_data, columns=model_nums, index=df_test.index)

    for d in [y] + models:
        print(d.describe())
    for d in [y] + models:
        print(d.shape, len(y) - len(d), type(d))

    y_indexes = set(y.index)
    print('y_indexes: %s' % sorted(y_indexes)[:10])

    for c, d in zip(model_nums, models):
        d_indexes = set(d.index)
        print('c=%s, d_indexes: %s' % (c, sorted(d_indexes)[:10]))
        assert d_indexes.issubset(y_indexes), (len(d_indexes - y_indexes))
        y[c].loc[d.index] = d['hat']

    def func(row):
        return all(x == -1 for x in row)

    empties = y.apply(func, axis=1)
    print('empties: %d' % len(empties))
    print(y[empties])



path = 'sneak/jobs_sneak.csv'
# path = 'small/jobs_small.csv'
path = 'all/jobs_all.csv'

if False:
    df = get_data(path)
    X, y, X_test = build_model001(df)
    exec_model(X, y, X_test, 'model001', do_score=False)

if False:
    df = get_data(path)
    X, y, X_test = build_model002(df)
    exec_model(X, y, X_test, 'model002')

if False:
    df = get_data(path)
    X, y, X_test = build_model003(df)
    exec_model(X, y, X_test, 'model003')

if True:
    combine_models()

