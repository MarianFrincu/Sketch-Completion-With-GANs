import torch

from models.discriminator import Discriminator
from models.generator import Generator
import matplotlib.pyplot as plt
import numpy as np
import cv2

gen = Generator(3, 3)
disc = Discriminator()

img_path = 'test_images/n02395406_1616-10.png'
small_img_path = 'test_images/n02395406_1616-10_small.png'

image = cv2.imread(img_path)
image1 = cv2.imread(small_img_path)

# Resize the image to 256x256 pixels (assuming the original image size is different)
image = cv2.resize(image, (256, 256))
image1 = cv2.resize(image1, (128, 128))

# Convert BGR image to RGB (OpenCV loads images as BGR by default)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

# Convert the image to PyTorch tensor and add batch dimension
tensor_image = torch.tensor(image, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
tensor_image = tensor_image.permute(2, 0, 1)  # Change from HWC to CHW
tensor_image = tensor_image.unsqueeze(0)  # Add batch dimension

tensor_image1 = torch.tensor(image1, dtype=torch.float32) / 255.0
tensor_image1 = tensor_image1.permute(2, 0, 1)
tensor_image1 = tensor_image1.unsqueeze(0)

with torch.no_grad():
    gen.eval()
    generated_img = gen(tensor_image)

with torch.no_grad():
    disc.eval()
    discriminated_pred = disc(tensor_image, tensor_image1)

np_img = generated_img.squeeze(0).numpy()

np_img = np.transpose(np_img, (1, 2, 0))
np_img = np.clip(np_img, 0, 1)

discriminated_pred = discriminated_pred.squeeze(0).cpu().numpy()

print(discriminated_pred)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

# Display generated image
plt.subplot(1, 2, 2)
plt.imshow(np_img)
plt.title('Generated Image')
plt.axis('off')

plt.tight_layout()
plt.show()
