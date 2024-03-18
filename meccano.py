import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os

"""
mode: {train} | {val} | {test} 
"""


def calculate_average_step_size(frame_start, frame_end):
    frame_diff = []
    for i in range(len(frame_start)):
        # print(f"Start: {int(frame_start[i][:-4])}, End: {int(frame_end[i][:-4])}")
        frame_diff.append(int(frame_end[i][:-4]) - int(frame_start[i][:-4]))

    print(f"Max: {np.max(frame_diff)}, Min: {np.min(frame_diff)}, Avg: {np.average(frame_diff)}, "
          f"StdDev: {np.std(frame_diff)}")
    print(f"p75: {np.percentile(frame_diff, 75)}")


def temporal_sampling(self, start_idx, end_idx, step_size, num_samples):
    # Calculate the total number of integers we can fit with the given step
    max_possible_elements = (end_idx - start_idx) // step_size + 1
    num_repeats = max(0, num_samples - max_possible_elements)

    if num_repeats == 0:
        start_idx = np.random.randint(start_idx, end_idx - ((num_samples - 1) * step_size) + 1)
        max_possible_elements = (end_idx - start_idx) // step_size + 1
        num_repeats = max(0, num_samples - max_possible_elements)

    frame_ids = np.arange(start_idx, start_idx + (num_samples - num_repeats) * step_size, step_size)

    # If num_repeats is greater than 0, append the last integer num_repeats tımes
    if num_repeats > 0:
        last_integer = frame_ids[-1]
        frame_ids = np.append(frame_ids, np.full(num_repeats, last_integer))

    index = frame_ids - start_idx
    return index


class Meccano(Dataset):
    def __init__(self, data_dir, mode, modality, step_size, num_frames, transform):
        self.data_dir = data_dir
        self.mode = mode
        self.modality = modality
        self.transform = transform
        self.step_size = step_size
        self.num_frames = num_frames

        self._construct_loader()

    def _construct_loader(self):
        path_to_file = os.path.join(
            self.data_dir, f"action_annotations/MECCANO_{self.mode}_actions.csv"
        )
        self._path_to_videos = []
        self._labels = []
        self._frame_start = []
        self._frame_end = []

        with open(path_to_file, 'r') as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()[1:]):
                video_path, action_label, action_noun, frame_start, frame_end = path_label.split(',')
                self._path_to_videos.append(video_path)
                self._labels.append(int(action_label))
                self._frame_start.append(frame_start)
                self._frame_end.append(frame_end)

        print(f"Total number of rows: {len(self._path_to_videos)}")
        print(f"Average step size: {calculate_average_step_size(self._frame_start, self._frame_end)}")

    def __getitem__(self, item):
        frames_desc = []
        frame_count = int(self._frame_start[item][:-4])

        while frame_count <= int(self._frame_start[item][:-4]):
            name_frame = str(frame_count)
            if len(name_frame) == 4:  # add a prefix 0
                name_frame = '0' + name_frame
            elif len(name_frame) == 3:  # add two prefix 0
                name_frame = '00' + name_frame
            elif len(name_frame) == 2:  # add three prefix 0
                name_frame = '000' + name_frame
            elif len(name_frame) == 1:  # add four prefix 0
                name_frame = '0000' + name_frame

            image_path = f"{self.data_dir}/{self.modality}/{self.mode}/{self._path_to_videos[item]}/{name_frame}.jpg"
            frames_desc.append(image_path)
            frame_count += 1

        frame_indexes = temporal_sampling(int(self._frame_start[item][:-4]), int(self._frame_end[item][:-4]),
                                          self.step_size, self.num_frames)
        frames_desc = np.take(frames_desc, frame_indexes, 0)

        frames = [Image.open(frame_path) for frame_path in frames_desc]
        aug_frames = self.transform(frames)

        out_frames = torch.stack(aug_frames)
        out_frames = out_frames.permute(1, 0, 2, 3)

        label = self._labels[item]
        path = self._path_to_videos[item]
        start = self._frame_start[item]
        end = self._frame_end[item]
        return out_frames, label, path, start, end

    def __len__(self):
        return len(self._path_to_videos)
