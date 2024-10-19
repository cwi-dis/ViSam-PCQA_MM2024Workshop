import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import random
from torchvision import transforms
from torch.utils import data
from PIL import Image
import cv2

from torchvision.transforms import InterpolationMode


class MMDataset(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(
        self,
        data_dir_texture,
        data_dir_depth,
        data_dir_normal,
        data_dir_vs,
        datainfo_path,
        transform,
        crop_size=224,
        img_length_read=6,
        is_train=True,
    ):
        super(MMDataset, self).__init__()
        dataInfo = pd.read_csv(
            datainfo_path, header=0, sep=",", index_col=False, encoding="utf-8-sig"
        )
        self.ply_name = dataInfo[["name"]]
        self.ply_mos = dataInfo["mos"]
        self.ply_disTypes = dataInfo["DT"]
        self.crop_size = crop_size
        self.data_dir_texture = data_dir_texture
        self.data_dir_depth = data_dir_depth
        self.data_dir_normal = data_dir_normal
        self.data_dir_vs = data_dir_vs
        self.transform = transform
        self.img_length_read = img_length_read
        self.length = len(self.ply_name)
        self.is_train = is_train

    def __len__(self):
        return self.length

    def random_crop(self, img, depth, normal, vs):
        # before random crop, make sure the img size is larger than crop size
        # img_width = img.size[1]
        # print("img width is:", img_width)
        # img_height = img.size[0]
        # print("img height is:", img_height)
        if img.size[1] < self.crop_size or img.size[0] < self.crop_size:
            # print("img size is smaller than crop size:", imge_name)
            temp_cropsize = min(img.size[0], img.size[1])
            # random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                img, output_size=(temp_cropsize, temp_cropsize)
            )
            img = transforms.functional.crop(img, i, j, h, w)
            depth = transforms.functional.crop(depth, i, j, h, w)
            normal = transforms.functional.crop(normal, i, j, h, w)
            vs = transforms.functional.crop(vs, i, j, h, w)
            # self.crop_size = 224
            img = transforms.functional.resize(
                img, (self.crop_size, self.crop_size)
            )  # xm: resize to crop_size
            depth = transforms.functional.resize(
                depth, (self.crop_size, self.crop_size)
            )
            normal = transforms.functional.resize(
                normal, (self.crop_size, self.crop_size)
            )
            vs = transforms.functional.resize(vs, (self.crop_size, self.crop_size))
            return img, depth, normal, vs
        else:
            # print("img size is larger than crop size:", imge_name)
            # random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                img, output_size=(self.crop_size, self.crop_size)
            )
            img = transforms.functional.crop(img, i, j, h, w)
            depth = transforms.functional.crop(depth, i, j, h, w)
            normal = transforms.functional.crop(normal, i, j, h, w)
            vs = transforms.functional.crop(vs, i, j, h, w)

            return img, depth, normal, vs

    def set_rand_seed(seed=1998):
        print("Random Seed: ", seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def __getitem__(self, idx):
        img_name = self.ply_name.iloc[idx, 0]
        texture_dir = os.path.join(self.data_dir_texture, img_name)
        depth_dir = os.path.join(self.data_dir_depth, img_name)
        normal_dir = os.path.join(self.data_dir_normal, img_name)
        vs_dir = os.path.join(self.data_dir_vs, img_name)
        # print("vs_dir:", vs_dir)

        img_channel = 3
        vs_channel = 1
        img_height_crop = self.crop_size
        img_width_crop = self.crop_size

        img_length_read = self.img_length_read
        transformed_img = torch.zeros(
            [img_length_read, img_channel, img_height_crop, img_width_crop]
        )
        transformed_depth = torch.zeros(
            [img_length_read, img_channel, img_height_crop, img_width_crop]
        )
        transformed_normal = torch.zeros(
            [img_length_read, img_channel, img_height_crop, img_width_crop]
        )
        transformed_vs = torch.zeros([img_length_read, img_height_crop, img_width_crop])
        # read images : only texture
        img_read_index = 0
        for i in range(img_length_read):
            # load images
            imge_name = os.path.join(texture_dir, str(i) + ".png")
            normal_name = os.path.join(normal_dir, str(i) + ".png")
            depth_name = os.path.join(depth_dir, str(i) + ".png")
            vs_name = os.path.join(vs_dir, str(i) + ".png")
            assert os.path.exists(imge_name), f"{imge_name}, Image do not exist!"
            assert os.path.exists(normal_name), f"{normal_name}, Normal do not exist!"
            assert os.path.exists(depth_name), f"{depth_name}, Depth do not exist!"
            assert os.path.exists(
                vs_name
            ), f"{vs_name}, vsiual saliency map do not exist!"

            read_frame = Image.open(imge_name)
            read_depth = Image.open(depth_name)
            read_normal = Image.open(normal_name)
            # read the vs_name which is a gray image
            read_vs = Image.open(
                vs_name,
            )
            # print("Image size:", read_vs.size)
            # print("Image mode:", read_vs.mode)

            read_frame = read_frame.convert("RGB")
            read_depth = read_depth.convert("RGB")
            read_normal = read_normal.convert("RGB")
            try:
                read_frame, read_depth, read_normal, read_vs = self.random_crop(
                    read_frame,
                    read_depth,
                    read_normal,
                    read_vs,
                )
            except:
                raise
            read_frame = self.transform(read_frame)
            read_depth = self.transform(read_depth)
            read_normal = self.transform(read_normal)
            read_vs = transforms.ToTensor()(read_vs)

            transformed_img[i] = read_frame
            transformed_depth[i] = read_depth
            transformed_normal[i] = read_normal
            transformed_vs[i] = read_vs

            img_read_index += 1

        if img_read_index < img_length_read:
            for j in range(img_read_index, img_length_read):
                transformed_img[j] = transformed_img[img_read_index - 1]
                transformed_depth[j] = transformed_depth[img_read_index - 1]
                transformed_normal[j] = transformed_normal[img_read_index - 1]
                transformed_vs[j] = transformed_vs[img_read_index - 1]

        # read gt
        y_mos = self.ply_mos.iloc[idx]
        y_label = torch.FloatTensor(np.array(y_mos))

        disType = self.ply_disTypes.iloc[idx]
        dis_label = torch.tensor(np.array(disType))

        return (
            transformed_img,
            transformed_depth,
            transformed_normal,
            transformed_vs,
            y_label,
            dis_label,
        )
