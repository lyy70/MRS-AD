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
        #每个像素点在所有样本与通道上的值
        pixel_values = embedding_vectors[:, :, h, w]#1[B, C]
        # 计算每个像素点在所有样本上的均值与方差
        mu_1[:, i] = torch.mean(pixel_values, dim=0)#1
        sigma_1[:, :, i] = np.cov(pixel_values.numpy(), rowvar=False) + epsilon1 * I#1
        #领域范围
        h_min = max(h - neighborhood_size // 2, 0)#1
        h_max = min(h + neighborhood_size // 2, H - 1)#1
        w_min = max(w - neighborhood_size // 2, 0)#1
        w_max = min(w + neighborhood_size // 2, W - 1)#1
        # 每个像素点在所有样本与通道上的领域值【B,C,neighbor_size,neighbor_size】
        neighborhood = embedding_vectors[:, :, h_min:h_max + 1, w_min:w_max + 1]#1
        neighborhood_values = neighborhood.reshape(C, -1)#1[77 6390]
        #  计算每个像素领域点在所有样本上的均值与方差
        mu_2[:, i] = torch.mean(neighborhood_values, dim=1)#1
        sigma_2[:, :, i] = np.cov((neighborhood_values.permute([1,0])).numpy(), rowvar=False) + epsilon1 * I #1
        #计算巴氏距离
        delta_mu = mu_1[:, i] - mu_2[:, i]
        sigma_prime_inv = torch.inverse((torch.tensor(sigma_1[:, :, i]) + torch.tensor(sigma_2[:, :, i])) / 2)
        bhatt_coefficient = 1 / 8 * delta_mu.T @ sigma_prime_inv @ delta_mu
        ma[h, w] = torch.exp(-gamma * bhatt_coefficient)
    # print("ma",ma)
    for i in range(H * W):
        h = i // W
        w = i % W
        h_min = max(h - neighborhood_size // 2, 0)
        h_max = min(h + neighborhood_size // 2, H - 1)
        w_min = max(w - neighborhood_size // 2, 0)
        w_max = min(w + neighborhood_size // 2, W - 1)
        neighborhood_ma = ma[h_min:h_max + 1, w_min:w_max + 1]  # 提取领域范围内的ma值
        sum_ma = torch.sum(neighborhood_ma)  # 计算领域范围内ma的和
        normalized_ma[h, w] = ma[h, w] / (sum_ma)
    # print("normalized_ma",normalized_ma)
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
        # 使用张量切片获取领域像素的值
        neighborhood_pixels = embedding_vectors[:, :, h_min:h_max + 1, w_min:w_max + 1]
        # 使用张量切片获取归一化的领域值
        neighborhood_mas = normalized_ma[h_min:h_max + 1, w_min:w_max + 1]
        # 计算领域像素值与归一化领域值的乘积
        product = neighborhood_pixels * neighborhood_mas.unsqueeze(0).unsqueeze(0)
        # 累加乘积到总和
        new_embedding_vector[:, :, h, w] = torch.sum(product, dim=(2, 3))
        # 计算累加后的总和的均值
        mean[:, i] = torch.mean(new_embedding_vector[:, :, h, w], dim=0)
        # 计算协方差
        # sigma[:, :, i] = np.cov(new_embedding_vector[:, :, h, w].numpy(), rowvar=False) + epsilon2 * I
        # 更新1
        pix_val = new_embedding_vector[:, :, h, w]
        # neighborhood_mas = normalized_ma[h_min:h_max + 1, w_min:w_max + 1]
        mean_val = mean[:, i][np.newaxis, :] * torch.sum(neighborhood_mas).numpy()
        X = (pix_val - mean_val)
        covariance_matrix = np.matmul(X.T, X) / B - torch.sum(neighborhood_mas ** 2)  # 0.958,0.919 #-1
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

def save_cm(embedding_vectors):
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    mean = torch.mean(embedding_vectors, dim=0).numpy()
    cov = torch.zeros(C, C, H * W).numpy()
    I = np.identity(C)
    for i in range(H * W):
        cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
    return mean, cov

def select_feature(embedding_vectors, topd):
    B, C, H, W = embedding_vectors.size()
    nonzero_counts = torch.count_nonzero(embedding_vectors, dim=(2, 3))
    sorted_channels = torch.argsort(nonzero_counts)
    top_channels = sorted_channels[:, :topd]
    selected_features = torch.zeros(B, topd, H, W, device=embedding_vectors.device)
    for b in range(B):
        for i in range(topd):
            selected_features[b, i, :, :] = embedding_vectors[b, top_channels[b, i], :, :]
    return selected_features
