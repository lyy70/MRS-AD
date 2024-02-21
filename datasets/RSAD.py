import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm

CLASS_NAMES = ['mild_disease', 'moderate_disease','severe_disease','mix_disease',]
#
class RSADDataset(Dataset):
    def __init__(self, dataset_path, class_name='mild_disease', is_train=True, resize=256, cropsize=224):
    #def __init__(self, dataset_path, class_name='mild_disease', is_train=True, resize=224):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        self.x, self.y, self.mask, self.img_name, self.img_type = self.load_dataset_folder()
        self.transform_x = T.Compose([T.Resize((resize, resize), Image.ANTIALIAS),
                                      T.ToTensor(),
                                      T.CenterCrop(cropsize),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize((resize, resize), Image.ANTIALIAS),
                                         T.ToTensor(),
                                         T.CenterCrop(cropsize)
                                         ])
    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask, img_fname, tot_types = [], [], [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            x.extend(img_fpath_list)
            # load gt labels
            if img_type == 'good':
                img_fname_list_good = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
                tot_types.extend(['good'] * len(img_fpath_list))
                img_fname.extend(img_fname_list_good)
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                img_fname_list_defect = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)
                img_fname.extend(img_fname_list_defect)
                tot_types.extend([img_type] * len(img_fpath_list))
        assert len(x) == len(y), 'number of x and y should be same'
        return list(x), list(y), list(mask), list(img_fname), list(tot_types)

    def __getitem__(self, idx):
        x, y, mask, img_name, img_type = self.x[idx], self.y[idx], self.mask[idx], self.img_name[idx], self.img_type[idx]
        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)
        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
            #mask = torch.zeros([1, self.resize, self.resize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
        return x, y, mask, img_name, img_type

# if __name__ == '__main__':
#     data = MVTecDataset(dataset_path='/data/LiuYuyao/Dataset/railway_anomaly_detection_standard2/', class_name='mild_disease', is_train=False)
#     test_dataloader = DataLoader(data, batch_size=1, pin_memory=True)
#     img_name_list = []
#     for (x, y, mask, img_name, img_type) in test_dataloader:
#         img_name_list.extend(img_name)
#     print("img_name_list",img_name_list)
