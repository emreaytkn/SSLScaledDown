import torchvision.utils
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import meccano

import matplotlib.pyplot as plt
import config as cfg

if __name__ == '__main__':
    data_dir = cfg.data_dir
    modality = cfg.test_modality

    batch_size = cfg.test_batch_size
    num_frames = cfg.test_num_frames

    step_size = cfg.step_size

    img_resize = cfg.test_img_resize

    test_transform = transforms.Compose([
        transforms.Lambda(lambda frames: [F.resize(frame, img_resize) for frame in frames])
    ])

    # Create dataset
    test_dataset = meccano.Meccano(data_dir, 'test', modality, step_size, num_frames, test_transform)

    # Create dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    images, labels = next(iter(test_dataloader)) # Extract one batch of images and labels
    grid_img = torchvision.utils.make_grid(images, nrow=8)
    grid_img = grid_img.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(15, 15))
    plt.imshow(grid_img)
    plt.axis('off')
    plt.show()




