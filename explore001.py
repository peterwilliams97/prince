# -*- coding: utf-8 -*-
"""

"""
from __future__ import division, print_function
import os
import re
import numpy as np
import pandas as pd
from pandas import DataFrame
from glob import glob
from pprint import pprint
from sklearn import cross_validation, utils
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from collections import defaultdict


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
    """Add dummy booleans to DataFrame `df` for each category in `col` of DataFrame `df_all`
        df_all is a superset of df
    """
    category_values = sorted(df_all[col].unique())
    data = np.zeros((len(df), len(category_values)), dtype=int)
    val_index = {str(val): i for i, val in enumerate(category_values)}
    assert len(val_index) == len(category_values)

    for i in range(len(df)):
        a = df[col].iloc[i]
        j = val_index[str(a)]
        data[i, j] = 1

    # df = df.loc[:, [c for c in df.columns if c != col]]
    for j, val in enumerate(category_values):
        df.loc[:, '%s_%s' % (col, val)] = data[:, j]

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
         encoding='latin-1',
         # dtype={'title': str, 'abstract': str}
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
    print('df      : %s ' % list(df.shape))
    print('df_train: %s %.2f' % (list(df_train.shape), len(df_train) / len(df)))
    print('df_test : %s %.2f' % (list(df_test.shape), len(df_test) / len(df)))

    # train_idx = set(df_train['subclasses'])
    # test_idx = set(df_test['subclasses'])
    # print('train:', sorted(train_idx))
    # print('test:', sorted(test_idx))
    # print('intersection:', sorted(train_idx & test_idx))

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


def S(df):
    # assert isinstance(df, DataFrame), type(df)
    return list(df.shape)


def exec_model(X, y, X_test, out_path, do_score=True):
    print('exec_model: X=%s,y=%s,X_test=%s,out_path="%s"' %
          (S(X), S(y), S(X_test), out_path))
    # print(X.describe())
    # print(y.describe())
    y = DataFrame(y, columns=['hat'], index=X.index)
    X.to_csv('%s.X_train.csv' % out_path, index_label='job_id')
    y.to_csv('%s.y_train.csv' % out_path, index_label='job_id')
    print(y.columns)

    if do_score:
        clf = ExtraTreesClassifier()
        score = get_score(clf, X, y.values.ravel(), cv=2, verbose=True, scoring='f1')
        print('score=%f' % score)

    clf = ExtraTreesClassifier()
    # clf = RandomForestClassifier()
    clf.fit(X, y.values.ravel())
    y_self = clf.predict(X)
    y_self = DataFrame(y_self, columns=['hat'], index=X.index)
    n = len(y)
    m = sum(y['hat'])
    s = sum(y_self['hat'])
    assert n == len(y_self)

    print('****', n, m, s, m / n, s / n)
    print(clf)
    for i in range(10):
        print('%4d: %d %d' % (i, y['hat'].iloc[i], y_self['hat'].iloc[i]))
    # assert False

    y_test = clf.predict(X_test)
    y_test = DataFrame(y_test, columns=['hat'], index=X_test.index)
    # print('X_test.index', X_test.index)
    # print('y_test.index', y_test.index)
    n = len(y)
    m = sum(y['hat'])
    print('y     : n=%d,m=%d=%.2f' % (n, m, m / n))
    n = len(y_self)
    m = sum(y_self['hat'])
    print('y_self: n=%d,m=%d=%.2f' % (n, m, m / n))
    n = len(y_test)
    m = sum(y_test['hat'])
    print('y_test: n=%d,m=%d=%.2f' % (n, m, m / n))
    n = len(y)
    m = sum(y_self['hat'] == y['hat'])
    print('accuracy: n=%d,m=%d=%.2f' % (n, m, m / n))
    y_test.to_csv('%s.y_test.csv' % out_path, index_label='job_id')
    # for i in range(100):
    #     print('%4d: %d %d' % (i, y['hat'].iloc[i], y_self['hat'].iloc[i]))

    pickle.dump('%s.pkl' % out_path, clf)
    print(clf.feature_importances_')


def build_model001(df):

    x_cols = ['salary_min', 'subclasses']

    df2 = remove_nulls(df, x_cols + ['hat'])

    df_train, df_test = split_train_test(df2)

    X, y = getXy(df_train, x_cols)
    X_test, _ = getXy(df_test, x_cols)

    X_all = pd.concat([X, X_test])

    X = getDummy(X_all, X, 'subclasses')
    X_test = getDummy(X_all, X_test, 'subclasses')

    print('df', df.shape)
    print('X_all', X_all.shape)
    assert frozenset(df.index) == frozenset(X_all.index)

    X_all = pd.concat([X, X_test])
    print('X_all', X_all.shape)
    assert frozenset(df.index) == frozenset(X_all.index)
    # assert False

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


def add_keywords(df, column, keywords):

    data = np.zeros((len(df), len(keywords)), dtype=np.int8)

    nan_count = 0

    for i, text in enumerate(df[column]):
        # assert isinstance(text, str), (column, type(text), text)
        if not isinstance(text, str):
            nan_count += 1
            continue
        words = set(RE_SPACE.split(text.lower()))
        words = {w.replace("'", '').replace("!", '') .replace("?", '')
                 for w in words}

        for j, kw in enumerate(keywords):
            if kw in words:
                data[i, j] = 1

    print('column=%s,df=%s,nan_count=%d=%.2f' % (column, list(df.shape), nan_count,
          nan_count / len(df)))
    # assert nan_count == 0

    df_out = df[[col for col in df.columns if col != column]]

    for j, kw in enumerate(keywords):
        df_out.loc[:, '%s_%s' % (column, kw)] = data[:, j]

    assert isinstance(df, DataFrame)

    return df_out


def build_model004(df):
    from keywords import get_keywords

    x_cols = ['salary_min', 'salary_max', 'title', 'abstract']
    x_cols = ['title', 'abstract']

    df_train, df_test = split_train_test(df)

    X, y = getXy(df_train, x_cols)
    X_test, _ = getXy(df_test, x_cols)

    keywords = get_keywords(50)

    X = add_keywords(X, 'title', keywords['title'])
    X = add_keywords(X, 'abstract', keywords['abstract'])
    X_test = add_keywords(X_test, 'title', keywords['title'])
    X_test = add_keywords(X_test, 'abstract', keywords['abstract'])

    # for col in X.columns:
    #     print(col, X[col].dtype)
    # for col in X_test.columns:
    #     print(col, X_test[col].dtype)
    # # assert False

    return X, y, X_test


STOP_WORDS = {
    '-',
    'and',
    'for',
    '/',
    '|',
    '&',
    'of',
    'in',
    'a',
    'as',
    'i',
    'on',
    'the',
    'this',
    'their',
    'an',
}

RE_SPACE = re.compile(r'[\s\.,;:\(\)\[\]/\+&\-\|]+')


def show_words_column(df, column, n_top):
    print('=' * 80)
    print('show_words:', df.shape, column)
    hat_counts = {}
    for hat in [-1, 0, 1]:
        counts = defaultdict(int)
        df2 = df[df['hat'] == hat]
        for title in df2[column]:
            title = title.lower()
            words = set(RE_SPACE.split(title))
            words = {w.replace("'", '').replace("!", '') .replace("?", '')
                     for w in words}
            for w in words:
                if w and (w not in STOP_WORDS) and ('$' not in w):
                    counts[w] += 1
        hat_counts[hat] = counts

    if False:
        for hat in [-1, 0, 1]:
            print('-' * 80)
            print('hat=%d' % hat)
            counts = hat_counts[hat]
            for i, w in enumerate(sorted(counts, key=lambda k: -counts[k])[:90]):
                print('%3d: %4d: %s' % (i, counts[w], w))

        top0 = sorted(hat_counts[0], key=lambda k: -hat_counts[0][k])
        top1 = sorted(hat_counts[1], key=lambda k: -hat_counts[0][k])

    key_words = set()
    for hat in [0, 1]:
        key_words |= set(hat_counts[hat].keys())
    contrasts = {w: [0, 0] for w in key_words}
    for i, hat in enumerate([0, 1]):
        for w, n in hat_counts[hat].items():
            contrasts[w][i] += n

    ratios = {}
    for w, (n0, n1) in contrasts.items():
        ratios[w] = (n1 + 10) / (n0 + 10)

    ratio_order = sorted(ratios.keys(), key=lambda k: ratios[k])


    print('-' * 80)
    for w in ratio_order[:n_top]:
        print('%8.3f %5d %5d "%s"' % (1.0 / ratios[w], contrasts[w][0], contrasts[w][1], w))
    print('-' * 80)
    for w in ratio_order[-n_top:]:
        print('%8.3f %5d %5d "%s"' % (ratios[w], contrasts[w][0], contrasts[w][1], w))

    # n_bottom = min(n_top, len(ratio_order) - n_top)
    # signicant_keys = ratio_order[:n_top] + ratio_order[-n_bottom:]

    pretty_dict = {}
    for w in ratio_order[:n_top] + ratio_order[-n_top:]:
        pretty_dict[w] = np.log10(ratios[w]), contrasts[w][0], contrasts[w][1]
    pretty_list = sorted(pretty_dict.items(), key=lambda x: (x[1], x[0]))
    print('*' * 80)
    pprint(pretty_list)

    return ratio_order, ratios, contrasts


def show_words(df, n_top):
    # for col in df.columns:
    #     print(col, df[col].dtype)
    # for s in df['abstract'].iloc[:5]:
    #     print(type(s), s)

    filled = df['abstract'].fillna('')
    df['abstract'] = filled
    # for s in df['abstract']:
    #     assert isinstance(s, str), (type(s), s)
    i_abstract = list(df.columns).index('abstract')
    for row in df.itertuples():
        s = row[i_abstract + 1]
        assert isinstance(s, str), (type(s), s, row)


    # assert False
    show_words_column(df, 'title', n_top)
    show_words_column(df, 'abstract', n_top)


RE_MODEL = re.compile(r'model(\d+)\.')


def model_num(path):
    m = RE_MODEL.search(path)
    assert m, path
    return int(m.group(1))


def combine_models():

    model_nums = [1, 2, 3]
    # model_nums = [2, 3]
    model_paths = ['model%03d.y_test.csv' % i for i in model_nums]
    assert all(os.path.exists(path) for path in model_paths)
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

    # empties = y.apply(func, axis=1)
    # print('empties: %d' % len(empties))
    # print(y[empties])

    def vote(row):
        print(row)
        for j in 1, 3, 2:
            if row[j] != -1:
                return row[j]
        # assert False, row
        return -3

    print(y.iloc[:20, :])
    y_series = y.iloc[:20, :].apply(vote, axis=1)
    assert False
    y_test = DataFrame(y_series, columns=['hat'], index=y.index)
    y_test.to_csv('%s.y_test.csv' % 'model004v', index_label='job_id')
    print(y_test.columns)
    print(y_test.describe())
    print(y_test.iloc[:10, :])


path = 'sneak/jobs_sneak.csv'
# path = 'small/jobs_small.csv'
# path = 'all/jobs_all.csv'

if False:
    df = get_data(path)
    X, y, X_test = build_model001(df)
    exec_model(X, y, X_test, 'model001', do_score=True)

if False:
    df = get_data(path)
    X, y, X_test = build_model002(df)
    exec_model(X, y, X_test, 'model002')

if False:
    df = get_data(path)
    X, y, X_test = build_model003(df)
    exec_model(X, y, X_test, 'model003')

if False:
    combine_models()

if False:
    df = get_data(path)
    show_words(df, 200)

if True:
    df = get_data(path)
    X, y, X_test = build_model004(df)
    # print('X', type(X))
    # print('y', type(y))
    # print('X_test', type(X_test))
    exec_model(X, y, X_test, 'model004')
