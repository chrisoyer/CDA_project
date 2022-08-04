#!/usr/bin/env python

import re
import torch
from pandas import read_parquet, DataFrame

class NNMatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_latents, n_personas, 
                 reconstruction_λ, regularization_λ, persona_λ) -> None:
        super().init()
        self.user_latents = torch.nn.Embeddings(num_embeddings=n_users, embedding_dim=n_latents)
        self.item_latents = torch.nn.Embeddings(num_embeddings=n_items, embedding_dim=n_latents)
        self.reconstruction_λ = reconstruction_λ
        self.regularization_λ = regularization_λ
        self.persona_λ = persona_λ
        self.n_users = n_users
        self.n_latents = n_latents
        self.n_personas = n_personas

    def load_data(self, data_path: str) -> None:
        sparse_data = read_parquet(data_path)
        self.X = torch.tensor(sparse_data.to_numpy())
        self.users = sparse_data.index
        self.items = sparse_data.columns
        self.U = torch.nn.Parameter(torch.rand(self.U.repeat(1, 1, self.n_personas)), require_autograd=True)
    
    def get_Xhat_all(self):
        self.Xhat = torch.mm(self.U, self.V)

    def reconstruction_loss(self):
        return ((self.X - torch.mm(self.U, self.V))**2).mean()

    def get_best_personas(self):
        self.best_persona = torch.argmax(self.Xhat, dim=3)

    def forward(self, X, ):

        reconstruction_error = torch.sum()
    


    def weighted_predict(self, rare_factor):
        pass