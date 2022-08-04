#!/usr/bin/env python

import pandas as pd
import numpy as np
from scipy import sparse
from scipy.stats.mstats import winsorize
import subprocess
from os import path
from pathlib import Path
from typing import Optional, Union
from tqdm import tqdm
import json

def load_track_data(track_filepath: str) -> pd.io.parsers.TextFileReader:
    return pd.read_csv(track_filepath, sep="\t", header=None, names=['user_id', 'song_id', 'play_count'],
                       iterator=True, chunksize=100_000)

def load_artist_mapping(artist_filepath: str) -> pd.DataFrame:
    return (pd.read_csv(artist_filepath, sep="<SEP>", on_bad_lines='warn',
                         header=None, names=['artist_id', 'song_id', 'artist_name', 'song_name']))

def add_artist(records: pd.DataFrame, song_artists: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(records, song_artists, on='song_id')

def clean_artist_name(artists: pd.Series) -> pd.DataFrame:
    """
    Clean artist naming so variations can be fully merged into a single artist
    """
    return (artists
            # remote featured artists
            .str.split("/ ", n=1).str[0]
            .str.split("ft. ", n=1).str[0]
            .str.split("feat. ", n=1).str[0]
            .str.split("Featuring ", n=1).str[0]
            # normalize unusual characters
            .str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
            # strip
            .str.strip()
            .str.title())


def agg_by_artist(records: pd.DataFrame) -> pd.DataFrame:
    """
    aggregate up to artist level
    """
    return (records
            .assign(artist_name = lambda df: clean_artist_name(df['artist_name']))
            .groupby(['user_id', 'artist_name']).agg(np.sum).reset_index()
           )

def modify_frequency_to_rating(frequencies: pd.Series, offset: int, log_offset: float) -> pd.Series:
    """
    ARGS:
        frequencies:    times item was listened/watched
        offset:         subtracted from final value to reflect a couple of listens => tried, didn't like
        log_offset:     small amount added to log to avoid large negative values for 0/1/2/3
    """
    return np.log2(frequencies + log_offset) - offset

def modify_count_data(records: pd.DataFrame, offset: int, log_offset: float) -> pd.DataFrame:
    """applies freq->rating to relevant entries"""
    return records.assign(play_count = lambda df: np.where(df['play_count']==0, np.NaN, 
                                   modify_frequency_to_rating(df['play_count'], offset, log_offset)))

def normalize_user_ratings(records: pd.DataFrame=None) -> pd.DataFrame:
    """

    ARGS:
        records:           data

    """
    user_means = (records.groupby(['user_id']).mean()
                  .rename(columns={"play_count": 'user_mean'}))

    return (pd.merge(left=records, right=user_means, on='user_id')
            .assign(play_count=lambda df: ((df['play_count'])/df['user_mean']))
            .drop(columns=['user_mean']))

def pivot_to_sparse(dense_data: pd.DataFrame=None) -> dict[dict, pd.DataFrame.sparse]:
    """
    change shape from long format to user x artist which is needed for factorization
    changes to sparse because dense matrix is very large and doesn't fit in memory
    """
    user_c   = pd.CategoricalDtype(sorted(dense_data['user_id'].unique()), ordered=True)
    artist_c = pd.CategoricalDtype(sorted(dense_data['artist_name'].unique()), ordered=True)

    row = dense_data['user_id'].astype(user_c)
    col = dense_data['artist_name'].astype(artist_c)
    sparse_matrix = (sparse.coo_matrix((dense_data["play_count"], 
                                            (row.cat.codes, col.cat.codes)),
                                        shape=(user_c.categories.size, artist_c.categories.size)))

    return {'data': (pd.DataFrame.sparse.from_spmatrix(
                            data=sparse_matrix,
                            index=user_c.categories,
                            columns=artist_c.categories)),
            'artist_dict': dict(zip(col.cat.codes, col.cat.categories.astype('str'))),
            'user_dict':   dict(enumerate(row.cat.categories.astype('str')))}

def drop_junk(df: pd.DataFrame=None, min_items: int=2, min_plays: int=0) -> pd.DataFrame:
    """min plays not implemented"""
    return df.loc[df.user_id.isin(df.groupby(['user_id']).count().query("""artist_name>@min_items""").index)]

def save_cleaned_data(records: pd.DataFrame=None, file_path:str=None, is_sparse: bool=False) -> None:
    Path(path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
    if is_sparse:
        sparse.save_npz(file_path, records.sparse.to_coo())
        col_json = json.dumps(records.columns.to_list())
        with open(file_path + '.columns.json', "w") as col_file:
            col_file.write(col_json)
        index_json = json.dumps(records.index.to_list())
        with open(file_path + '.index.json', "w") as index_file:
            index_file.write(index_json)
    else:
        records.to_parquet(path=file_path)


def process_raw_data(raw_records_filepath: str='./data/raw/train_triplets.txt', 
                     mapping_filepath: str='./data/raw/unique_tracks.txt',
                     long_df_savepath: str='./data/processed/long_df.parquet', 
                     sparse_pivot_df_savepath: str='./data/processed/sparse_pivot.parquet') -> None:
    """
    runs above code, from loading data to saving results
    """
    offset = 0
    log_offset = .8
    song_artists = load_artist_mapping(mapping_filepath)
    track_iterator = load_track_data(raw_records_filepath)
    long_data_df = pd.concat([modify_count_data(agg_by_artist(add_artist(chunk, song_artists)),
                                                offset=offset, log_offset=log_offset)
                             for chunk in tqdm(track_iterator)])
    normed_data_df = normalize_user_ratings(long_data_df)
    useful_data_df = drop_junk(normed_data_df, min_items=2, min_plays=5)
    save_cleaned_data(useful_data_df, file_path=long_df_savepath)
    sparse_data = pivot_to_sparse(normed_data_df)
    save_cleaned_data(sparse_data['data'], file_path=sparse_pivot_df_savepath, is_sparse=True)

if __name__ == "__main__":
    process_raw_data()
