from torchvision import transforms
import torchvision.transforms.functional as F
import meccano

import config as cfg

if __name__ == '__main__':
    data_dir = cfg.data_dir
    modality = cfg.test_modality

    num_frames = cfg.test_num_frames

    step_size = cfg.step_size

    img_resize = cfg.test_img_resize

    test_transform = transforms.Compose([
        transforms.Lambda(lambda frames: [F.resize(frame, img_resize) for frame in frames])
    ])

    # Create dataset
    test_dataset = meccano.Meccano(data_dir, 'test', modality, step_size, num_frames, test_transform)
