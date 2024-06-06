
import  os,random, shutil
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg",'.jpeg','.png','.bmp','.tif','.tiff'])
class DataLoaderTrain(Dataset):
    def __init__(self, input_dir, target_dir, resolution=256, use_Normalize=True, use_data_augmentation=False,
                 getname=False):
        super(DataLoaderTrain, self).__init__()
        self.use_data_augmentation = use_data_augmentation
        self.getname = getname
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.resolution = resolution
        self.target_files = sorted(os.listdir(self.target_dir))
        self.input_files = sorted(os.listdir(self.input_dir))
        self.targetfilenames = [os.path.join(self.target_dir, x) for x in self.target_files if is_image_file(x)]
        self.inputfilenames = [os.path.join(self.input_dir, x) for x in self.input_files if is_image_file(x)]
        self.nums = len(self.targetfilenames)
        if use_Normalize:
            transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        else:
            transform_list = [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        index = index % self.nums
        input_image = Image.open(self.inputfilenames[index]).convert('RGB')
        target_image = Image.open(self.targetfilenames[index]).convert('RGB')
        if self.use_data_augmentation:
            input_image = input_image.resize((self.resolution + 20, self.resolution + 20), Image.BICUBIC)
            target_image = target_image.resize((self.resolution + 20, self.resolution + 20), Image.BICUBIC)
            w_offset = np.random.randint(0, max(0, 20 - 1))
            h_offset = np.random.randint(0, max(0, 20 - 1))
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
            input_image = input_image[:, h_offset:h_offset + self.resolution, w_offset:w_offset + self.resolution]
            target_image = target_image[:, h_offset:h_offset + self.resolution, w_offset:w_offset + self.resolution]
        else:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        if self.getname:
            return input_image, target_image, self.input_files[index]
        else:
            return input_image, target_image

class DatasetFromFolder_in_test_mode(Dataset):
    '''
    The class deals with loading images from the dataset loader
    only return the test image itself for test mode.
    '''

    def __init__(self, image_dir,getname=False,use_Normalize=True):
        super(DatasetFromFolder_in_test_mode, self).__init__()
        self.img_path = image_dir
        self.getname=getname
        self.image_filenames = [x for x in sorted(os.listdir(self.img_path)) if is_image_file(x)]
        if use_Normalize:
            transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        else:
            transform_list=[transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)
        self.nums=len(self.image_filenames)

    def __len__(self):
        return self.nums
    def __getitem__(self, index):
        test_img = Image.open(os.path.join(self.img_path, self.image_filenames[index]))

        test_img = self.transform(test_img)
        if self.getname:
            return test_img,self.image_filenames[index]
        else:
            return test_img