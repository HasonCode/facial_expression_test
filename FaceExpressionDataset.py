from pathlib import Path
from typing import List, Tuple, Union

import PIL.Image
import torch
from torch.utils.data import Dataset


class FaceExpressionDataset(Dataset):
    class_names = ["anger", "blink", "frown", "neutral", "smile"]

    def __init__(self, root_dir: str = "", transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.class_counts = [0] * len(self.class_names)

        for label, class_name in enumerate(self.class_names):
            class_dir = self.root_dir / class_name
            paths = sorted(class_dir.glob("*.png"))
            self.class_counts[label] = len(paths)
            for path in paths:
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, ind: Union[int, torch.Tensor]):
        if torch.is_tensor(ind):
            ind = int(ind.item())
        img_path, label = self.samples[ind]
        image = PIL.Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label