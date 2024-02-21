import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import mahalanobis

def batt_distance(embedding_vectors, gamma, neighborhood_size, epsilon1):
    B, C, H, W = embedding_vectors.size()
    mu_1 = torch.zeros(C, H * W)
    sigma_1 = torch.zeros(C, C, H * W).numpy()
    mu_2 = torch.zeros(C, H * W)
    sigma_2 = torch.zeros(C, C, H * W).numpy()
    I = np.identity(C)
    ma = torch.zeros(H, W)
    normalized_ma = torch.zeros(H, W)
    for i in range(H * W):
        h = i // W
        w = i % W
        pixel_values = embedding_vectors[:, :, h, w]
        mu_1[:, i] = torch.mean(pixel_values, dim=0)
        sigma_1[:, :, i] = np.cov(pixel_values.numpy(), rowvar=False) + epsilon1 * I
        h_min = max(h - neighborhood_size // 2, 0)
        h_max = min(h + neighborhood_size // 2, H - 1)
        w_min = max(w - neighborhood_size // 2, 0)
        w_max = min(w + neighborhood_size // 2, W - 1)
        neighborhood = embedding_vectors[:, :, h_min:h_max + 1, w_min:w_max + 1]
        neighborhood_values = neighborhood.reshape(C, -1)
        mu_2[:, i] = torch.mean(neighborhood_values, dim=1)
        sigma_2[:, :, i] = np.cov((neighborhood_values.permute([1,0])).numpy(), rowvar=False) + epsilon1 * I 
        delta_mu = mu_1[:, i] - mu_2[:, i]
        sigma_prime_inv = torch.inverse((torch.tensor(sigma_1[:, :, i]) + torch.tensor(sigma_2[:, :, i])) / 2)
        bhatt_coefficient = 1 / 8 * delta_mu.T @ sigma_prime_inv @ delta_mu
        ma[h, w] = torch.exp(-gamma * bhatt_coefficient)
    for i in range(H * W):
        h = i // W
        w = i % W
        h_min = max(h - neighborhood_size // 2, 0)
        h_max = min(h + neighborhood_size // 2, H - 1)
        w_min = max(w - neighborhood_size // 2, 0)
        w_max = min(w + neighborhood_size // 2, W - 1)
        neighborhood_ma = ma[h_min:h_max + 1, w_min:w_max + 1]
        sum_ma = torch.sum(neighborhood_ma)
        normalized_ma[h, w] = ma[h, w] / (sum_ma)
    return normalized_ma

def mean_sigma(embedding_vectors, neighborhood_size, normalized_ma,epsilon2):
    B, C, H, W = embedding_vectors.size()
    I = np.identity(C)
    mean = torch.zeros(C, H * W).numpy()
    sigma = torch.zeros(C, C, H * W).numpy()
    new_embedding_vector = torch.zeros(B, C, H, W)
    for i in range(H * W):
        h = i // W
        w = i % W
        h_min = max(h - neighborhood_size // 2, 0)  # 1
        h_max = min(h + neighborhood_size // 2, H - 1)  # 1
        w_min = max(w - neighborhood_size // 2, 0)  # 1
        w_max = min(w + neighborhood_size // 2, W - 1)  # 1
        neighborhood_pixels = embedding_vectors[:, :, h_min:h_max + 1, w_min:w_max + 1]
        neighborhood_mas = normalized_ma[h_min:h_max + 1, w_min:w_max + 1]
        product = neighborhood_pixels * neighborhood_mas.unsqueeze(0).unsqueeze(0)
        new_embedding_vector[:, :, h, w] = torch.sum(product, dim=(2, 3))
        mean[:, i] = torch.mean(new_embedding_vector[:, :, h, w], dim=0)
        pix_val = new_embedding_vector[:, :, h, w]
        mean_val = mean[:, i][np.newaxis, :] * torch.sum(neighborhood_mas).numpy()
        X = (pix_val - mean_val)
        covariance_matrix = np.matmul(X.T, X) / B - torch.sum(neighborhood_mas ** 2)
        sigma[:, :, i] = covariance_matrix + epsilon2 * I
    return mean, sigma
    
def score_map(embedding_vectors, train_outputs):
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
    dist_list = []
    for i in range(H * W):
        mean = train_outputs[0][:, i]
        #conv_inv = np.linalg.inv(train_outputs[1][:, :, i] + epsilon3)
        conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
        dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
        dist_list.append(dist)
    dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)
    mean_dist = np.nanmean(dist_list)
    dist_list = np.nan_to_num(dist_list, nan=mean_dist)
    dist_list = torch.tensor(dist_list)
    return dist_list

