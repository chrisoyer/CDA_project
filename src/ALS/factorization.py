#!/usr/bin/env python
#__package__ = None

import numpy as np
import pandas as pd
from scipy import sparse
# from src import preprocess #WTAF
from pathlib import Path
from os import path
from time import localtime, strftime


# data loading

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


def load_data(file_path: str, datatype: str) -> dict[str, pd.DataFrame.sparse, dict]:
    if datatype == 'sparse':
        raise NotImplementedError
    if datatype == 'dense':
        return pivot_to_sparse(pd.read_parquet(file_path))

def get_item_rarity(data: pd.DataFrame) -> pd.DataFrame:
    return 1. / data.sum()

# Init data

def initialize_UV(n_users: int, n_items: int, n_latents: int, n_personas: int) -> dict[str, np.array]:
    U_size = (n_users, n_latents, n_personas) if n_personas > 1 else (n_users, n_latents)
    return {'U':  np.random.normal(0, 1, U_size, ),
            'V': np.random.normal(0, 1, (n_items, n_latents))}

# Loss functions


def get_prediction_loss(X: np.array=None, X_hat: np.array=None, X_ignore_mask: np.array=None, 
                        non_null_only: bool=False) -> np.array:
    """
    loss≡λpredAcc∑i,j∈X(Xi,j−Wi,jFi,j)2
    Calculates the reconstruction loss.
    ARGS:
        X:
        U:
        V:
        non_null_only: if True, will only calculate squared difference of non-null elements
    """

    ignore = X_ignore_mask.astype(bool)

    if non_null_only:
        zero_mask = np.isnan(X)
        element_count = np.count_nonzero(np.where(ignore, 0,X))
    else: 
        zero_mask = 1
        element_count = (~ignore.astype(bool)).sum()

    return (np.linalg.norm(((X - X_hat) * (zero_mask + ignore)), ord=None)  # None => Frobenius norm
            /element_count)

def get_users_latents_loss(U: np.array) -> float:
    return np.linalg.norm(U, ord=2, keepdims=0).mean()

def add_up_personas(U: np.array) -> np.array:
    """not ideal way to regularize, but sparse personas caused non-convergence"""
    if len(U.shape) == 3:
        return np.sum(U, axis=2)
    else:
        return U

def get_items_latents_loss(V: np.array) -> float:
    return np.linalg.norm(V.T, ord=2, keepdims=1).mean()

def get_prediction_diversity_regularization_loss():
    """
    +λpredDiversity∥∑uWi,jFi,j22
    penalizes very frequent predictions to diversify predictions
    """
    pass

def get_persona_diversity_loss() -> float:
    """
    l2 penalty on count of time each person is used in user profile
    """

    #return np.linalg.norm(X_hat.)

def get_persona_distance_loss():
    """
    1/l2 penalty on total distance between persona vectors for each user
    """
    pass

def get_combined_loss(lambdas: dict, X: np.array, U: np.array, V: np.array, 
                      non_null_only: bool, n_personas: int, X_ignore_mask: np.array) -> dict[str, float]:
    
    # persona dim
    multi_persona = True if n_personas>1 else False
    if multi_persona:
        U = reshape_to_flatten_personas(U)
    
    X_hat = get_Xhat(U, V)

    # persona dim
    if multi_persona:
        X_hat = drop_irrelevant_preds(reshape_to_stack_personas(X_hat, n_personas))


    pred_loss = lambdas['reconstruction'] * get_prediction_loss(X=X, X_hat=X_hat, non_null_only=non_null_only,
                                                                X_ignore_mask=X_ignore_mask)
    user_loss = lambdas['users_reg'] * get_users_latents_loss(add_up_personas(U))
    item_loss = lambdas['items_reg'] * get_items_latents_loss(V)
    user_diversity_loss = get_persona_diversity_loss()
    return {'total_loss': 
                (pred_loss +
                user_loss +
                item_loss),
            'pred_loss': pred_loss,
            'user_loss': user_loss,
            'item_loss': item_loss,

    }

# optimize functions

def optimize_items(X: np.array, U: np.array, lambdas: dict[str, float]) -> np.array:
    return (X.T @ U) @ (np.linalg.inv((U.T@U) + lambdas['items_reg'] * np.identity(n=U.shape[1]))) 

def optimize_users(X: np.array, V: np.array, lambdas: dict[str, float]) -> np.array:
    return (X @ V) @ (np.linalg.inv(V.T @ V + lambdas['users_reg'] * np.identity(n=V.shape[1])))

def optimize_items_multip(Xstar: np.array, U: np.array, lambdas: dict[str, float]) -> np.array:
    Ustar = reshape_to_flatten_personas(U)
    Ustar_gram = Ustar.T @ Ustar
    return (Xstar.T @ Ustar) @ (np.linalg.inv(Ustar_gram + lambdas['users_reg'] * np.identity(n=Ustar_gram.shape[1])))

def reshape_to_flatten_personas(matrix: np.array) -> np.array:
    return np.squeeze(np.concatenate(np.split(matrix, indices_or_sections=matrix.shape[2], axis=2), axis=0))

def reshape_to_stack_personas(matrix: np.array, n_personas) -> np.array:
    return np.squeeze(np.stack(np.split(matrix, n_personas, axis=0), axis=2))

def drop_irrelevant_preds(X_hat: np.array) -> np.array:
    return np.amax(X_hat, axis=2)

def assign_items_to_personas(X: np.array, U: np.array, V: np.array) -> np.array:
    X_hat_full = np.stack([get_Xhat(U=U[:, :, persona], V=V).astype('float16') for persona in range(0, U.shape[2])], 2)
    best_persona = np.argmax(X_hat_full, axis=2)
    return (best_persona, np.concatenate([np.where(best_persona==persona, X, 0).astype('float16') 
                                          for persona in range(0, U.shape[2])], axis=0))

def optimize_users_multip(Xstar: np.array, n_personas: int, V: np.array, 
                          lambdas: dict[str, float]) -> np.array:
    U_hat_flat = (Xstar @ V) @ (np.linalg.inv(V.T @ V + lambdas['users_reg'] * np.identity(n=V.shape[1])))
    return reshape_to_stack_personas(U_hat_flat, n_personas)

def get_Xhat(U: np.array, V: np.array) -> np.array:
    return (U @ V.T)

def train(Xes: dict[str, np.array]=None, U: np.array=None, V: np.array=None,
            hyperparameters: dict=None, return_UV: bool=False) -> dict[str, np.array, list]:
    """
    ARGS:
        X
        U
        V
        hyperparameters:
            lambdas: dict[str, float] regularization strengths
            ε: float                  tolerance
            n_personas: int 
            non_null_only
            n_latents: int
            ε:       tolerance
            n_personas:
            non_null_only: ignore null values
            verbose: set to 0 to just run, 1 to do dev work
        return_UV:    whether to return matrices or just loss values
    RETVRNS:
    """
    X = Xes['X_train']
    X_test = Xes['X_test']
    X_ignore_mask = Xes['mask']
    loss = [{'total_loss': np.PINF}]
    i = 0

    
    # Get baseline
    loss.append({'dataset': 'baseline', 'iter': i, **get_combined_loss(lambdas=hyperparameters['lambdas'], X=X_test, U=U, V=V, 
                                          non_null_only=hyperparameters['non_null_only'], 
                                          n_personas=hyperparameters['n_personas'], 
                                          X_ignore_mask=(~X_ignore_mask))})
    while (loss[-1]['total_loss'] > hyperparameters['ε']) & (hyperparameters['max_iters'] > i):


        if hyperparameters['n_personas'] == 1:
            V = optimize_items(X=X, U=U, lambdas=hyperparameters['lambdas'])
            U = optimize_users(X=X, V=V, lambdas=hyperparameters['lambdas'])
        else:
            best_persona, Xstar = assign_items_to_personas(X, U, V)
            V = optimize_items_multip(Xstar=Xstar, U=U, lambdas=hyperparameters['lambdas'])
            U = optimize_users_multip(Xstar=Xstar, V=V,
                    n_personas=hyperparameters['n_personas'], lambdas=hyperparameters['lambdas'])

        loss_params = dict(lambdas=hyperparameters['lambdas'], X=X, U=U, V=V, non_null_only=hyperparameters['non_null_only'],
                           n_personas=hyperparameters['n_personas'], X_ignore_mask=X_ignore_mask)
        loss.append({'dataset': 'train', 'iter': i, **get_combined_loss(**loss_params)})
        if hyperparameters['verbose'] > 0:
            print(loss[-1]) 

        if i % hyperparameters['test_frequency']==0:
            loss.append({'dataset': 'test', 'iter': i, **get_combined_loss(lambdas=hyperparameters['lambdas'], X=X_test, U=U, V=V, 
                                          non_null_only=hyperparameters['non_null_only'], 
                                          n_personas=hyperparameters['n_personas'], 
                                          X_ignore_mask=(~X_ignore_mask))})
            if hyperparameters['verbose'] > 0:
                print(loss[-1])
        i += 1
    loss.append({'dataset': 'baseline', 'iter': i, **get_combined_loss(lambdas=hyperparameters['lambdas'], X=X_test, U=U, V=V, 
                                          non_null_only=hyperparameters['non_null_only'], 
                                          n_personas=hyperparameters['n_personas'], 
                                          X_ignore_mask=(~X_ignore_mask))})

    result = {'losses': loss}
    if return_UV:
        result = {**result, **{'U': U, 'V': V,}}
    return result

# runner code

def split(X: np.array=None, test_fraction: float=.15):
    # Nb: X_train referese to the X matrix, not a traditional design matrix
    test_mask = np.zeros(X.size, dtype=int).astype(bool)
    test_mask[: int(X.size*test_fraction)] = True
    rng = np.random.default_rng()
    rng.shuffle(test_mask)
    test_mask = test_mask.reshape((X.shape))
    X_train = np.where(test_mask, 0, X)
    X_test = np.where(~test_mask, 0, X)
    return {'X_train': X_train, 'X_test': X_test, 'mask': test_mask}

def multi_persona_factorization(data: dict[str, np.array, dict], hyperparameters: dict[str, int, dict[str, float]]):
    """
    ARGS:
        cv_schedule: iters i % this == 0 get cv instead of simple train """
    X = data['data'].to_numpy()
    initializations = initialize_UV(n_latents=hyperparameters['n_latents'], n_items=X.shape[1], n_users=X.shape[0],
                                    n_personas=hyperparameters['n_personas'])
    rarities = get_item_rarity(data['data'])
    Xes = split(X)
    result = train(Xes=Xes, **initializations, hyperparameters=hyperparameters)
    return result

def get_diversity_factor():
    pass

# posttrain
def predict_single_user(user_artist_ratings, U: np.array, V: np.array):
    pass

def save_results(results: dict, hypers: dict, file_path: str) -> None:
    Path(path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(results['losses']).to_hdf(path_or_buf=file_path, key='data', format='table')
    pd.DataFrame.from_records(hypers).to_hdf(file_path, key='hypers')


def main():
    hyperparameters = {
    'lambdas': {'reconstruction': 20,
                'items_reg':      .05,
                'users_reg':      .4,},
    'ε':              .001,
    'max_iters':      15,
    'n_personas':     3,
    'n_latents':      4,
    'non_null_only':  True,
    'verbose':        1,
    'cv_loops':       3,
    'test_frequency': 1,
    }
    load_path = "./data/processed/long_df.parquet"
    mod_descript = f"__personas-{hyperparameters['n_personas']}__latents-{hyperparameters['n_latents']}"
    save_path = f"./model_results/train_results{strftime('%Y-%m-%d__%H-%M', localtime())}_{mod_descript}.hdf5"
    data = load_data(file_path=load_path, datatype='dense')
    result = multi_persona_factorization(data=data, hyperparameters=hyperparameters)
    save_results(result, hyperparameters, save_path)

if __name__ == "__main__":
    main()
    
