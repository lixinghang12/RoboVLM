# from data.calvin_dataset_video import DiskCalvinVideoDataset, RandomShiftsAug

# if __name__ == "__main__":
#     print('finish import test')

import enum
import random
import numpy as np
from lightning.pytorch.utilities.combined_loader import CombinedLoader, _SUPPORTED_MODES, _Sequential
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter
import logging
import tensorflow as tf
from typing import Callable, Dict, Union, Tuple
import dlimp as dl

# class WeightedCombinedLoader(CombinedLoader):
#     def __init__(self, loaders, mode="max_size_cycle", weights=None):
#         # 直接使用列表作为 loaders
#         super().__init__(loaders, mode)

#         if weights is None:
#             raise ValueError("You must provide weights for the DataLoaders.")
        
#         # Normalize the weights to sum to 1
#         self.weights = np.array(weights) / np.sum(weights)
#         self.loader_iters = None

#     def __iter__(self):
#         cls = _SUPPORTED_MODES[self._mode]["iterator"]
#         iterator = cls(self.flattened, self._limits)
#         iter(iterator)
#         self._iterator = iterator

#         # Initialize each DataLoader's iterator directly from the list
#         self.loader_iters = [iter(loader) for loader in self.flattened]
#         return self

#     def __next__(self):
#         if self.loader_iters is None or not self.loader_iters:
#             raise StopIteration

#         # Randomly choose a DataLoader index based on the given weights
#         selected_loader_idx = random.choices(range(len(self.loader_iters)), weights=self.weights, k=1)[0]
#         selected_loader_iter = self.loader_iters[selected_loader_idx]

#         try:
#             # Get the next batch from the selected DataLoader
#             batch = next(selected_loader_iter)
#         except:
#             # # If the selected DataLoader is exhausted, remove it and try again
#             # del self.loader_iters[selected_loader_idx]
#             # del self.weights[selected_loader_idx]
#             # if not self.loader_iters:
#             #     raise StopIteration
#             # self.weights = np.array(self.weights) / np.sum(self.weights)  # Normalize remaining weights
#             # return self.__next__()
#             self.loader_iters[selected_loader_idx] = iter(self.flattened[selected_loader_idx])
#             batch = next(self.loader_iters[selected_loader_idx])

#         # Format the output to match the expected structure
#         if isinstance(self._iterator, _Sequential):
#             return batch

#         batch_idx = 0  # Placeholder value, update as needed
#         return batch, batch_idx, selected_loader_idx


def decode_and_resize(
    obs: Dict,
    resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]],
    depth_resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]],
) -> Dict:
    """Decodes images and depth images, and then optionally resizes them."""
    image_names = {key[6:] for key in obs if key.startswith("image_")}
    depth_names = {key[6:] for key in obs if key.startswith("depth_")}

    if isinstance(resize_size, tuple):
        resize_size = {name: resize_size for name in image_names}
    if isinstance(depth_resize_size, tuple):
        depth_resize_size = {name: depth_resize_size for name in depth_names}

    for name in image_names:
        if name not in resize_size:
            logging.warning(
                f"No resize_size was provided for image_{name}. This will result in 1x1 "
                "padding images, which may cause errors if you mix padding and non-padding images."
            )
        import pdb; pdb.set_trace()
        image = obs[f"image_{name}"]
        if image.dtype == tf.string:
            if tf.strings.length(image) == 0:
                # this is a padding image
                image = tf.zeros((*resize_size.get(name, (1, 1)), 3), dtype=tf.uint8)
            else:
                image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)
        elif image.dtype != tf.uint8:
            raise ValueError(f"Unsupported image dtype: found image_{name} with dtype {image.dtype}")
        if name in resize_size:
            image = dl.transforms.resize_image(image, size=resize_size[name])
        obs[f"image_{name}"] = image

    for name in depth_names:
        if name not in depth_resize_size:
            logging.warning(
                f"No depth_resize_size was provided for depth_{name}. This will result in 1x1 "
                "padding depth images, which may cause errors if you mix padding and non-padding images."
            )
        depth = obs[f"depth_{name}"]

        if depth.dtype == tf.string:
            if tf.strings.length(depth) == 0:
                depth = tf.zeros((*depth_resize_size.get(name, (1, 1)), 1), dtype=tf.float32)
            else:
                depth = tf.io.decode_image(depth, expand_animations=False, dtype=tf.float32)[..., 0]
        elif depth.dtype != tf.float32:
            raise ValueError(f"Unsupported depth dtype: found depth_{name} with dtype {depth.dtype}")

        if name in depth_resize_size:
            depth = dl.transforms.resize_depth_image(depth, size=depth_resize_size[name])

        obs[f"depth_{name}"] = depth

    return obs



if __name__ == "__main__":
    import pickle as pkl
    from functools import partial

    with open('episode_traj.pkl', 'rb') as file:
        episode_traj = pkl.load(file)
    with open('action_traj.pkl', 'rb') as file:
        action_traj = pkl.load(file)

    def apply_obs_transform(fn: Callable[[Dict], Dict], frame: Dict) -> Dict:
        frame["task"] = fn(frame["task"])
        frame["observation"] = dl.vmap(fn)(frame["observation"])
        return frame
    fn = partial(
        apply_obs_transform,
        partial(decode_and_resize, resize_size=(224,224), depth_resize_size=(224, 224)),
    )
    from IPython import embed; embed()
    # import pdb; pdb.set_trace()
    x = fn(action_traj)
    # import pdb; pdb.set_trace()
    y = fn(episode_traj)



    # from torch.utils.data import DataLoader
    # iterables = {'a': DataLoader(range(10), batch_size=4), 'b': DataLoader(range(100, 150), batch_size=5)}
    # iterables = [DataLoader(range(10), batch_size=4, drop_last=True, shuffle=True), DataLoader(range(100, 150), batch_size=5, shuffle=True)]
    # weights = [0.8, 0.2]
    # combined_loader = WeightedCombinedLoader(iterables, mode="max_size_cycle", weights=weights)
    # for i, batch in enumerate(combined_loader):
    #     print(batch)