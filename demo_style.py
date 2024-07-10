import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

from styles import StyleAugmentor

style_augmentor = StyleAugmentor().eval()

image = Image.open("images/00007.png").convert("RGB")
image = image.resize((1280, 720))
image = np.array(image)[:512, :1024]
Image.fromarray(image).save("images/test.png")

image_tensor = torch.from_numpy(image)
# hwc -> nchw
image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).float()

for i in range(10):
    with torch.no_grad():
        image_translated = style_augmentor(image_tensor)
        image_translated = np.clip(image_translated.squeeze().permute(1, 2, 0).numpy() * 255., 0, 255).astype(np.uint8)

    # nchw -> hwc
    Image.fromarray(image_translated).save(f"images/test{i}.png")

