import cv2
from PIL import Image
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def preprocess_img(img, channels=3):
    shape_r = 288
    shape_c = 384
    img_padded = torch.zeros((channels, shape_r, shape_c), dtype=torch.uint8)
    original_shape = img.shape
    rows_rate = original_shape[1] / shape_r
    cols_rate = original_shape[2] / shape_c
    if rows_rate > cols_rate:
        print("rows_rate > cols_rate")
        new_cols = (original_shape[2] * shape_r) // original_shape[1]
        img = transforms.functional.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[
            :,
            :,
            ((img_padded.shape[2] - new_cols) // 2) : (
                (img_padded.shape[2] - new_cols) // 2 + new_cols
            ),
        ] = img
    else:
        print("cols_rate < rows_rate")
        new_rows = (original_shape[1] * shape_c) // original_shape[2]
        img = transforms.functional.resize(img, (shape_c, new_rows))

        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[
            :,
            :,
            ((img_padded.shape[1] - new_rows) // 2) : (
                (img_padded.shape[1] - new_rows) // 2 + new_rows
            ),
        ] = img

    # shape_r = 288
    # shape_c = 384
    # batch_img_padded = torch.ones(
    #     (batch_img.shape[0], batch_img.shape[1], batch_img.shape[2], shape_r, shape_c)
    # )
    # # img_padded should be a tensor
    # img_padded = torch.ones((shape_r, shape_c, channels), dtype=np.uint8)
    # original_shape = batch_img.shape
    # rows_rate = original_shape[3] / shape_r
    # cols_rate = original_shape[4] / shape_c
    # for i in range(batch_img.shape[0]):  # batch size
    #     for j in range(batch_img.shape[1]):  # img_numbers
    #         img = batch_img[i, j, :, :, :]  # 3, 224, 224
    #         if rows_rate > cols_rate:
    #             new_cols = (original_shape[4] * shape_r) // original_shape[3]
    #             img = transforms.functional.resize(img, (new_cols, shape_r))
    #             if new_cols > shape_c:
    #                 new_cols = shape_c
    #             img_padded[
    #                 :,
    #                 ((img_padded.shape[1] - new_cols) // 2) : (
    #                     (img_padded.shape[1] - new_cols) // 2 + new_cols
    #                 ),
    #             ] = img
    #         else:
    #             new_rows = (original_shape[3] * shape_c) // original_shape[4]
    #             img = transforms.functional.resize(img, (shape_c, new_rows))

    #             if new_rows > shape_r:
    #                 new_rows = shape_r
    #             img_padded[
    #                 ((img_padded.shape[0] - new_rows) // 2) : (
    #                     (img_padded.shape[0] - new_rows) // 2 + new_rows
    #                 ),
    #                 :,
    #             ] = img
    #         batch_img_padded[i, j, :, :, :] = img_padded

    return img_padded


def postprocess_img(pred, org):
    shape_r = org.shape[1]
    shape_c = org.shape[2]
    pred = np.array(pred)
    predictions_shape = pred.shape

    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[
            :,
            ((pred.shape[1] - shape_c) // 2) : (
                (pred.shape[1] - shape_c) // 2 + shape_c
            ),
        ]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[
            ((pred.shape[0] - shape_r) // 2) : (
                (pred.shape[0] - shape_r) // 2 + shape_r
            ),
            :,
        ]
    # change the PIL img to tensor
    img = torch.from_numpy(img)

    return img


class MyDataset(Dataset):
    """Load dataset."""

    def __init__(self, ids, stimuli_dir, saliency_dir, fixation_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ids = ids
        self.stimuli_dir = stimuli_dir
        self.saliency_dir = saliency_dir
        self.fixation_dir = fixation_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        im_path = self.stimuli_dir + self.ids.iloc[idx, 0]
        image = Image.open(im_path).convert("RGB")
        img = np.array(image) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)
        # if self.transform:
        #    img = self.transform(image)

        smap_path = self.saliency_dir + self.ids.iloc[idx, 1]
        saliency = Image.open(smap_path)

        smap = np.expand_dims(np.array(saliency) / 255.0, axis=0)
        smap = torch.from_numpy(smap)

        fmap_path = self.fixation_dir + self.ids.iloc[idx, 2]
        fixation = Image.open(fmap_path)

        fmap = np.expand_dims(np.array(fixation) / 255.0, axis=0)
        fmap = torch.from_numpy(fmap)

        sample = {"image": img, "saliency": smap, "fixation": fmap}

        return sample
