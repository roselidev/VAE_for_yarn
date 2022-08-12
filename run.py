import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import datetime
from pytz import timezone
import os
import matplotlib.pyplot as plt
import codecs
import sys, os, warnings
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
import numbers
from collections.abc import Sequence


class DONGJIN(VisionDataset):
    training_file = "training.pt"
    test_file = "test.pt"
    classes = [
        "0 - normal",
        "1 - anomaly",
    ]

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.img_paths

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.img_paths

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        self.train = train # train or test
        self.image_dir = os.path.join(self.raw_folder, f"{'train' if self.train else 'test'}-images")
        # self.image_dir = os.path.join(self.raw_folder, f"{'sample' if self.train else 'sample-test'}-images")
        self.label_file = os.path.join(self.raw_folder, f"{'train' if self.train else 'test'}-labels")
        # self.label_file = os.path.join(self.raw_folder, f"{'sample' if self.train else 'sample-test'}-labels")

        if self._check_legacy_exist():
            self.img_paths, self.targets = self._load_legacy_data()
            return

        if not self._check_exists():
            raise RuntimeError("Dataset not found at ", self.raw_folder)
        
        self.img_paths, self.targets = self._load_data()

    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False
        return True
    
    def _load_legacy_data(self):
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file))
    
    def _load_data(self):
        data = read_image_file(self.image_dir)
        targets = read_label_file(len(data))

        return data, targets

    def __getitem__(self, index:int) -> Tuple[Any, Any]:
        img_path, target = self.img_paths[index], int(self.targets[index])

        img = Image.open(os.path.join(self.image_dir, img_path))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.img_paths)
    
    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return os.path.isdir(os.path.join(self.raw_folder, f'{"train-images" if self.train else "test-images"}'))
        # return os.path.isdir(os.path.join(self.raw_folder, f'{"sample-images" if self.train else "sample-test-images"}'))

def read_label_file(path: int) -> int:
    # TODO : dummy function
    return np.zeros((path,))
    
def read_image_file(path: str) -> Image:
    return [p for p in os.listdir(path) if p.endswith('.jpg')]

def _setup_size(size, center, error_msg):
    if not isinstance(center, tuple):
        raise ValueError(error_msg)

    if isinstance(size, numbers.Number):
        return int(center[1]-(int(size)/2)), int(center[0]-(int(size)/2)), int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return int(center[1]-(size[0]/2)), int(center[0]-(size[0]/2)), size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return int(center[1]-(size[0]/2)), int(center[0]-(size[1]/2)), size[0], size[1]

class DongjinCrop(torch.nn.Module):
    """Crops the given image at given point
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
        center (tuple of int): the center point to crop. If provided coordinates exceeds image sizes, image with all 0 will be returned.
    """
    def __init__(self, size, center):
        super().__init__()
        self.size = _setup_size(size, center, error_msg="Please make sure that center is tuple of (x, y) and to provide only two dimensions (h, w) for size.")

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        return transforms.functional.crop(img, *self.size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"

def imshow_grid(imgs, tf_inv=None):
    figure, axes = plt.subplots(2, 3, figsize=(12,6))
    ax = axes.flatten()
    for i, im in enumerate(imgs):
        if tf_inv:
            ax[i].imshow(tf_inv(im))
        else:
            ax[i].imshow(im)
        ax[i].axis('off')
    figure.tight_layout()
    plt.show()

class VAE(nn.Module):
    def __init__(self, image_size, hidden_size_1, hidden_size_2, latent_size):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(image_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc31 = nn.Linear(hidden_size_2, latent_size)
        self.fc32 = nn.Linear(hidden_size_2, latent_size)

        self.fc4 = nn.Linear(latent_size, hidden_size_2)
        self.fc5 = nn.Linear(hidden_size_2, hidden_size_1)
        self.fc6 = nn.Linear(hidden_size_1, image_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h3 = F.relu(self.fc4(z))
        h4 = F.relu(self.fc5(h3))
        return torch.sigmoid(self.fc6(h4))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, IMG_RESIZE*IMG_RESIZE*COLOR_DEPTH))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, IMG_RESIZE*IMG_RESIZE*COLOR_DEPTH), reduction = 'sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE, KLD

def train(epoch, model, train_loader, optimizer, name=None):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(DEVICE)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        #recon_batch = torch.reshape(recon_batch, (-1, COLOR_DEPTH, IMG_RESIZE, IMG_RESIZE))
        BCE, KLD = loss_function(recon_batch, data, mu, logvar)

        loss = BCE + KLD

        writer.add_scalar("Train/Reconstruction Error", BCE.item() / len(data), batch_idx + epoch * (len(train_loader.dataset)/BATCH_SIZE) )
        writer.add_scalar("Train/KL-Divergence", KLD.item() / len(data), batch_idx + epoch * (len(train_loader.dataset)/BATCH_SIZE) )
        writer.add_scalar("Train/Total Loss" , loss.item() / len(data), batch_idx + epoch * (len(train_loader.dataset)/BATCH_SIZE) )

        loss.backward()

        train_loss += loss.item()

        optimizer.step()

        # if batch_idx % 100 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader),
        #         loss.item() / len(data)))
            
    print("{}======> Epoch: {} Average loss: {:.4f}".format(
        name if name else '=', epoch, train_loss / len(train_loader.dataset)
    ))        
    return train_loss / len(train_loader.dataset)

def test(epoch, model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(DEVICE)
            
            recon_batch, mu, logvar = model(data)
            #recon_batch = torch.reshape(recon_batch, (-1, COLOR_DEPTH, IMG_RESIZE, IMG_RESIZE))
            BCE, KLD = loss_function(recon_batch, data, mu, logvar)

            loss = BCE + KLD

            writer.add_scalar("Test/Reconstruction Error", BCE.item() / len(data), batch_idx + epoch * (len(test_loader.dataset)/BATCH_SIZE) )
            writer.add_scalar("Test/KL-Divergence", KLD.item() / len(data), batch_idx + epoch * (len(test_loader.dataset)/BATCH_SIZE) )
            writer.add_scalar("Test/Total Loss" , loss.item() / len(data), batch_idx + epoch * (len(test_loader.dataset)/BATCH_SIZE) )
            test_loss += loss.item()

            if batch_idx == 0:
                n = min(data.size(0), 6)
                comparison = torch.cat([data[:n], recon_batch.view(-1, COLOR_DEPTH, IMG_RESIZE, IMG_RESIZE)[:n]]) # (16, 1, 28, 28)
                grid = torchvision.utils.make_grid(comparison.cpu()) # (3, 62, 242)
                writer.add_image("Test image - Above: Real data, below: reconstruction data", grid, epoch)
    return test_loss / len(test_loader.dataset)

def latent_to_image(save_loc, model, latent_size):
    with torch.no_grad():
        sample = torch.randn(6, latent_size).to(DEVICE)
        recon_image = model.decode(sample).cpu()
        grid = torchvision.utils.make_grid(recon_image.view(-1, COLOR_DEPTH, IMG_RESIZE, IMG_RESIZE))

        plt.imshow(grid.permute(1,2,0))
        plt.savefig(os.path.join(save_loc, 'sample.png'))
        # writer.add_image("Latent To Image", grid, epoch)


if __name__ == "__main__":
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda:3" if USE_CUDA else "cpu")
    print("사용하는 Device : ", DEVICE)

    current_time = datetime.datetime.now(timezone('Asia/Seoul'))
    current_time = current_time.strftime('%Y-%m-%d-%H:%M')

    saved_loc = os.path.join('./outs', current_time)
    os.mkdir(saved_loc)

    print("저장 위치: ", saved_loc)

    writer = SummaryWriter(saved_loc)

    iterate = True
    EPOCHS = 3
    BATCH_SIZE = 32
    IMG_RESIZE = 500
    COLOR_DEPTH = 1 # GrayScale
    latent_size = 32

        
    tf = transforms.Compose([
        DongjinCrop(IMG_RESIZE, (1300, 600)),
        transforms.Grayscale(),
        # transforms.ColorJitter(brightness=.5, hue=.3, saturation=.3, contrast=.5),
        # transforms.RandomAffine(degrees=360, scale=(0.1, 1.1)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=(0,180)),
        transforms.ToTensor()
    ])
    tf_inv = transforms.Compose([
        transforms.ToPILImage()
    ])


    epochs = [3, 10, 100]
    bsize = [16, 64]
    latent = [2, 32, 128]
    exp_num = 1
    total_exp = len(epochs)*len(bsize)*len(latent)
    if iterate:
        for e in epochs:
            for b in bsize:
                for l in latent:

                    current_time = datetime.datetime.now(timezone('Asia/Seoul'))
                    current_time = current_time.strftime('%Y-%m-%d-%H:%M')

                    saved_loc = os.path.join('./outs', current_time)
                    if not os.path.exists(saved_loc):
                        os.mkdir(saved_loc)

                    print("저장 위치: ", saved_loc)
                        
                    # Loading trainset, testset and trainloader, testloader
                    trainset = DONGJIN(root = './data', train = True, transform = tf)
                    trainloader = torch.utils.data.DataLoader(trainset, batch_size = b, shuffle = True, num_workers = 2)

                    testset = DONGJIN(root = './data', train = False, transform = tf)
                    testloader = torch.utils.data.DataLoader(testset, batch_size = b, shuffle = True, num_workers = 2)

                    writer = SummaryWriter(saved_loc)

                    # Model and training settings
                    VAE_model = VAE(IMG_RESIZE*IMG_RESIZE*COLOR_DEPTH, 512, 256, l).to(DEVICE)
                    lr = 1e-4 if e < 10 else 1e-5
                    optimizer = optim.Adam(VAE_model.parameters(), lr = lr)

                    # train/test
                    for epoch in tqdm(range(0, e)):
                        train_loss = train(epoch, VAE_model, trainloader, optimizer, name=f'Experiment # {exp_num} / {total_exp} : saved at {saved_loc}')
                        test_loss = test(epoch, VAE_model, testloader)
                    
                    # save sample image and model
                    latent_to_image(saved_loc, VAE_model, l)

                    torch.save({
                        'epoch':e,
                        'model_state_dict':VAE_model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'train_loss': train_loss,
                        'test_loss' : test_loss,
                        'batch_size':b,
                        'leargning_rate':lr,
                        'latent_dim':l,
                        'img_resize':IMG_RESIZE
                        }, os.path.join(saved_loc, 'model.pt'))

                    writer.close()
                    exp_num += 1

    if not iterate:
        VAE_model = VAE(IMG_RESIZE*IMG_RESIZE*COLOR_DEPTH, 512, 256, latent_size).to(DEVICE)
        optimizer = optim.Adam(VAE_model.parameters(), lr = 1e-4)
        for epoch in tqdm(range(0, EPOCHS)):
            train(epoch, VAE_model, trainloader, optimizer)
            test(epoch, VAE_model, testloader)
            print("\n")
            latent_to_image(epoch, VAE_model)

        writer.close()

    def get_error_term(v1, v2, _rmse=True):
        if _rmse:
            return np.sqrt(np.mean((v1 - v2) ** 2, axis=1))
        #return MAE
        return np.mean(abs(v1 - v2), axis=1)



