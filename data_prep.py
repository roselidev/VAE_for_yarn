from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, Grayscale, ToPILImage
import os, sys
import cv2
import numpy as np

"""
Datasets
"""

class DongjinVideoDataset(Dataset):
    """
    Dataset class for dongjin factory raw videos
    """

    """
    Main Modules
    """

    def __init__(self, conf, transform=None):
        self.conf = conf['Dataset']
        self.video_dir = self.conf['video_dir'] # directory of raw videos
        self.image_dir = self.conf['image_dir'] # directory to save extracted images
        self.image_path_list = []               # image is loaded when __getitem__() is called
        self.transform = transform              # transformations to put on images when __getitem__() is called

        if not os.path.exists(self.image_dir):
            # self.videos_to_frames(save_transform=True) # [for debugging]
            self.videos_to_frames()
        self.image_path_list = self.get_image_path_list()
        assert len(self.image_path_list) > 0

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        sample = cv2.imread(os.path.join(self.image_dir, self.image_path_list[idx]))
        sample = sample[:,:,::-1]
        if self.transform:
            sample = self.transform(np.array(sample))

        return sample

    """
    Sub Modules
    """

    def get_image_path_list(self):
        """
        return : list<string> : all jpg filepaths in self.image_dir
        """
        return [img_path for img_path in os.listdir(self.image_dir) if img_path.endswith('jpg')]

    def videos_to_frames(self, save_transform=False):
        """
        return : meta : save videos into images (maybe with frame interval) and return video meta info
        """
        interval = int(self.conf['frame_interval']) if int(self.conf['frame_interval']) > 0 else 1

        for vpath in os.listdir(self.video_dir):
            cap = cv2.VideoCapture(os.path.join(self.video_dir, vpath))

            cnt = 0
            total = 0
            if not total:
                total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            while cap.isOpened():
                ret, frame = cap.read()
                if ret is False:
                    break
                if cnt % interval == 0:
                    frame = frame[:,:,::-1] # BGRtoRGB
                    if save_transform:
                        frame = self.transform_img(frame)
                    # save images
                    filename = f"{vpath.replace('.mp4','')}_{str(cnt).zfill(6)}.jpg"
                    filepath = os.path.join(self.image_dir, filename)
                    if not os.path.exists(self.image_dir):
                        os.mkdir(self.image_dir)
                    if save_transform:
                        cv2.imwrite(filepath, np.array(ToPILImage()(frame)).copy())
                    else:
                        cv2.imwrite(filepath, frame)
                cnt += 1
                if cnt % 30 == 0:
                    print('reading frame : ' , cnt, '/', total)
            
            width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
            cap.release()
            cv2.destroyAllWindows()

            meta = {'frame_len':total, 'width':width, 'height':height}

        return meta

"""
Transforms
    ref : https://pytorch.org/vision/stable/transforms.html
"""

def dongjin_transform():
    # return Compose([ToTensor(), Resize(256), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Grayscale()])
    return Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Grayscale()])

"""
DataLoader
"""

def get_dataloader(configuration):
    configuration = configuration['DataLoader']
    tsfm = dongjin_transform()
    dataset = DongjinVideoDataset(configuration, tsfm)
    dataloader = DataLoader(dataset, batch_size=int(configuration['batch_size']), shuffle=configuration['shuffle'], num_workers=configuration['num_workers'])

    return dataloader


"""
Test Codes
"""

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERROR] You have to put config path : `python data_loader.py [path_to_config_file]`")
        exit()
    import yaml
    import time

    s = time.time()
    conf = yaml.load(open(sys.argv[1], 'r'), Loader=yaml.SafeLoader)

    dl = get_dataloader(conf)
    print('----- time taken : ', time.time() - s)
    for d in iter(dl):
        print(d.size())
        break
