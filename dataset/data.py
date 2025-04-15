import torch
from torch.utils.data import Dataset
from PIL import Image
import dataset.util as Utils
import logging

class Harvard_Dataset(Dataset):
    def __init__(self, dataroot, resolution=256, split='train', data_len=20):
        self.resolution = resolution
        self.data_len = data_len
        self.split = split

        self.vis_path = Utils.get_paths_from_images('{}/CT'.format(dataroot))
        self.ir_path = Utils.get_paths_from_images('{}/MRI'.format(dataroot))
        self.fusion_path = Utils.get_paths_from_images('{}/Fusion_K'.format(dataroot))

        self.dataset_len = len(self.vis_path)

        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_vis = Image.open(self.vis_path[index]).convert("YCbCr")
        img_ir = Image.open(self.ir_path[index]).convert("YCbCr")
        img_fusion = Image.open(self.fusion_path[index]).convert("L")

        img_full = Image.open(self.vis_path[index]).convert("RGB")
        img_vis = img_vis.split()[0]
        img_ir = img_ir.split()[0]

        if self.split == "val":
            [img_vis, img_ir, img_fusion] = Utils.transform_augment([img_vis, img_ir, img_fusion], split=self.split, min_max=(-1, 1))
            img_full = Utils.transform_full(img_full, min_max=(-1, 1))
            path = str(self.vis_path[index])
            path = path.replace("\\", "/")
            name = str(path.split("/")[-1].split(".png")[0])
            return {'vis': img_vis, 'ir': img_ir, 'fusion': img_fusion, 'img_full': img_full, 'Index': index}, name
        else:
            [img_vis, img_ir, img_fusion], *crop_params = Utils.transform_augment([img_vis, img_ir, img_fusion], split=self.split, min_max=(-1, 1))
            img_full = Utils.transform_full_augment(img_full, *crop_params, min_max=(-1, 1))
            return {'vis': img_vis, 'ir': img_ir, 'fusion': img_fusion, 'img_full': img_full, 'Index': index}

class Harvard_Test_Dataset(Dataset):
    def __init__(self, dataroot, resolution=256, split='train', data_len=20):
        self.resolution = resolution
        self.data_len = data_len
        self.split = split
        self.vis_path = Utils.get_paths_from_images('{}/CT'.format(dataroot))
        self.ir_path = Utils.get_paths_from_images('{}/MRI'.format(dataroot))

        self.dataset_len = len(self.vis_path)

        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_vis = Image.open(self.vis_path[index]).convert("YCbCr")
        img_ir = Image.open(self.ir_path[index]).convert("YCbCr")

        img_full = Image.open(self.vis_path[index]).convert("RGB")

        img_vis = self.resize_to_multiple_of_8(img_vis)
        img_ir = self.resize_to_multiple_of_8(img_ir)
        img_full = self.resize_to_multiple_of_8(img_full)

        img_vis = img_vis.split()[0]
        img_ir = img_ir.split()[0]

        if self.split == "val":
            [img_vis, img_ir] = Utils.transform_augment([img_vis, img_ir], split=self.split, min_max=(-1, 1))
            img_full = Utils.transform_full(img_full, min_max=(-1, 1))
            path = str(self.vis_path[index])
            path = path.replace("\\", "/")
            name = str(path.split("/")[-1].split(".png")[0])
            return {'vis': img_vis, 'ir': img_ir, 'img_full': img_full, 'Index': index}, name
        else:
            [img_vis, img_ir], *crop_params = Utils.transform_augment([img_vis, img_ir], split=self.split, min_max=(-1, 1))
            img_full = Utils.transform_full_augment(img_full, *crop_params, min_max=(-1, 1))
            return {'vis': img_vis, 'ir': img_ir, 'img_full': img_full, 'Index': index}

    def resize_to_multiple_of_8(self, img):
        width, height = img.size
        new_width = width - (width % 8)
        new_height = height - (height % 8)
        img_resized = img.resize((new_width, new_height))
        return img_resized

class Data:
    def __init__(self, train_path, eval_path):
        self.train_path = train_path
        self.eval_path = eval_path

    def create_dataloader(dataset, dataset_opt, phase):
        '''create dataloader '''
        if phase == 'train':
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=dataset_opt['batch_size'],
                shuffle=dataset_opt['use_shuffle'],
                num_workers=dataset_opt['num_workers'],
                pin_memory=True)
        elif phase == 'val':
            return torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        else:
            raise NotImplementedError(
                'Dataloader [{:s}] is not found.'.format(phase))

    def create_dataset(dataset_opt, phase):
        dataset = None
        if dataset_opt['dataset'] == 'Harvard':
            dataset = Harvard_Dataset(dataroot=dataset_opt['dataroot'],
                        resolution=dataset_opt['resolution'],
                        split=phase,
                        data_len=dataset_opt['data_len'])
            logger = logging.getLogger('base')
            logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                                       dataset_opt['name']))
        elif dataset_opt['dataset'] == 'Test_mif':
            dataset = Harvard_Test_Dataset(dataroot=dataset_opt['dataroot'],
                        resolution=dataset_opt['resolution'],
                        split=phase,
                        data_len=dataset_opt['data_len'])
            logger = logging.getLogger('base')
            logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                                       dataset_opt['name']))
        return dataset