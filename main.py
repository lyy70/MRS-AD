import pickle
import random
from random import sample
import argparse
import numpy as np
import os
import cm
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib
import torch
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
import datasets.RSAD as RSAD
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser('MSR-AD')
    parser.add_argument('--data_path', type=str, default='/media/LiuYuyao/Dataset/railway_anomaly_detection_standard5/')
    parser.add_argument('--save_path', type=str, default='./railway_result_test')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
    parser.add_argument('--neighborhood_size', type=int, default='3')
    parser.add_argument('--gamma', type=float, default='0.75')
    parser.add_argument('--epsilon1', type=float, default='0.5')
    parser.add_argument('--epsilon2', type=float, default='0.0015')
    parser.add_argument('--d2', type=int, default='250')
    parser.add_argument('--d3', type=int, default='450')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.arch == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif args.arch == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d2 = 512
        t_d3 = 1024
    model.to(device)
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx2 = torch.tensor(sample(range(0, t_d2), args.d2))
    idx3 = torch.tensor(sample(range(0, t_d3), args.d3))
    outputs = []
    def hook(module, input, output):
        outputs.append(output)
    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    os.makedirs(os.path.join(args.save_path, 'temp_%s' % args.arch), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]
    plt.close()
    total_roc_auc = []
    total_pixel_roc_auc = []
    total_pixel_pr = []

    for class_name in RSAD.CLASS_NAMES:
        train_dataset = RSAD.RSADDataset(args.data_path, class_name=class_name, is_train=True)
        test_dataset = RSAD.RSADDataset(args.data_path, class_name=class_name, is_train=False)
        train_dataloader = DataLoader(train_dataset, batch_size=16, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=16, pin_memory=False)
        train_outputs = OrderedDict([('layer1', []), ('layer2',[]), ('layer3', [])])
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        # extract train set features
        train_feature_filepath = os.path.join(args.save_path, 'temp_%s' % args.arch, 'train_%s.pkl' % class_name)
        if not os.path.exists(train_feature_filepath):
            for (x, _, _, _, _) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
                with torch.no_grad():
                    _ = model(x.to(device))
                for k, v in zip(train_outputs.keys(), outputs):
                    train_outputs[k].append(v.cpu().detach())
                outputs = []
            for k, v in train_outputs.items():
                train_outputs[k] = torch.cat(v, 0)
            embedding_vectors_train2 = train_outputs['layer2']
            embedding_vectors_train3 = train_outputs['layer3']

            embedding_vectors_train2 = torch.index_select(embedding_vectors_train2, 1, idx2)
            embedding_vectors_train3 = torch.index_select(embedding_vectors_train3, 1, idx3)

            Max_pool = torch.nn.MaxPool2d(3, 1, 1)
            embedding_vectors_train2 = Max_pool(embedding_vectors_train2)
            embedding_vectors_train3 = Max_pool(embedding_vectors_train3)

            normalized_ma2 = cm.batt_distance(embedding_vectors_train2, args.gamma, args.neighborhood_size, args.epsilon1)
            normalized_ma3 = cm.batt_distance(embedding_vectors_train3, args.gamma, args.neighborhood_size, args.epsilon1)

            mean2, cov2 = cm.mean_sigma(embedding_vectors_train2, args.neighborhood_size, normalized_ma2, args.epsilon2)
            mean3, cov3 = cm.mean_sigma(embedding_vectors_train3, args.neighborhood_size, normalized_ma3, args.epsilon2)
            train_outputs2 = [mean2, cov2]
            train_outputs3 = [mean3, cov3]
            with open(train_feature_filepath, 'wb') as f:
                pickle.dump(train_outputs2, f)
            with open(train_feature_filepath, 'wb') as f:
                pickle.dump(train_outputs3, f)
        else:
            print('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                train_outputs2 = pickle.load(f)
            with open(train_feature_filepath, 'rb') as f:
                train_outputs3 = pickle.load(f)

        gt_list = []
        gt_mask_list = []
        test_imgs = []
        img_name_list = []
        img_type_list = []
        total_pixel_f1 = []

        for (x, y, mask, img_name, img_type) in tqdm(test_dataloader,'| feature extraction | test | %s |' % class_name):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_np = mask.cpu().numpy()
            gt_np1 = np.where(gt_np > 0.5, 1, 0)
            gt_mask_list.extend(gt_np1.astype(int))
            img_name_list.extend(np.asarray(img_name))
            img_type_list.extend(np.asarray(img_type))
            with torch.no_grad():
                _ = model(x.to(device))
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
            outputs = []
        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)
        embedding_vectors_test2 = test_outputs['layer2']
        embedding_vectors_test3 = test_outputs['layer3']
        embedding_vectors_test2 = torch.index_select(embedding_vectors_test2, 1, idx2)
        embedding_vectors_test3 = torch.index_select(embedding_vectors_test3, 1, idx3)

        Max_pool = torch.nn.MaxPool2d(3, 1, 1)
        embedding_vectors_test2 = Max_pool(embedding_vectors_test2)
        embedding_vectors_test3 = Max_pool(embedding_vectors_test3)

        dist_list2 = cm.score_map(embedding_vectors_test2, train_outputs2)
        dist_list3 = cm.score_map(embedding_vectors_test3, train_outputs3)

        score_map2 = F.interpolate(dist_list2.unsqueeze(1), size=x.size(2), mode='bilinear',
                                   align_corners=False).squeeze().numpy()
        score_map3 = F.interpolate(dist_list3.unsqueeze(1), size=x.size(2), mode='bilinear',
                                   align_corners=False).squeeze().numpy()
        score_map = (score_map2 + score_map3) / 2

        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        total_roc_auc.append(img_roc_auc)
        print('image ROCAUC: %.3f' % (img_roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))

        gt_mask = np.asarray(gt_mask_list)
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        f1_scores = (2 * precision * recall) / (precision + recall)
        f1_px = np.max(f1_scores[np.isfinite(f1_scores)])
        print("F1max分数: ", f1_px)
        total_pixel_f1.append(f1_px)
        threshold = thresholds[np.argmax(f1)]

        ttt = gt_mask.flatten().astype("int64")
        uuu = scores.flatten()
        fpr, tpr, _ = roc_curve(ttt, uuu)
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print('pixel ROCAUC: %.4f' % (per_pixel_rocauc))

        pix_pr_auc = auc(recall, precision)
        total_pixel_pr.append(pix_pr_auc)

        print('pixel PR: %.4f' % (pix_pr_auc))
        fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
        save_dir = args.save_path + '/' + f'pictures_{args.arch}'
        os.makedirs(save_dir, exist_ok=True)
        plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name, img_name_list, img_type_list)

    print('Average ROCAUC: %.4f' % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    print('Average pixel ROCUAC: %.4f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")
    print('Average PR: %.4f' % np.mean(total_pixel_pr))
    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)
    print('Average f1: %.4f' % np.mean(total_pixel_f1))

def plot_fig(test_img, scores, gts, threshold, save_dir, class_name, img_name_list, img_type_list):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img_name = img_name_list[i]
        img_type = img_type_list[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        plt.imshow(img)
        plt.imshow(heat_map, cmap='jet', alpha=0.4,)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, class_name + '_heat_map_{}_{}.png'.format(i,img_type)), dpi=100,
                    bbox_inches='tight', pad_inches=0)
        plt.close()
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, class_name + '_mask_{}_{}.png'.format(i,img_type)), dpi=100,
                    bbox_inches='tight', pad_inches=0)
        plt.close()
        plt.imshow(vis_img)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, class_name + '{}_vis_img.png'.format(i)), dpi=100,
                    bbox_inches='tight', pad_inches=0)
        plt.close()

        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image_{}_{}'.format(img_name, img_type))
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth_{}_{}'.format(img_name, img_type))
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)
        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()
def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x

def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z
if __name__ == '__main__':
    main()

