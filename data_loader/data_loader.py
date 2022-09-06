from base.base_data_loader import BaseDataLoader
from .kadid10k_dataset import Kadid10kDataset

class Kadid10kDataLoader(BaseDataLoader):
    def __init__(self, width, height, img_path, flist, crop_size,
                 batch_size, validation_split=0.2, shuffle=True, num_workers=1, is_train=True):
        self.dataset = Kadid10kDataset(width=width,
                                       height=height,
                                       img_path=img_path,
                                       flist=flist,
                                       crop_size=crop_size,
                                       is_train=is_train)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)