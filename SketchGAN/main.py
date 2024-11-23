import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision.transforms import transforms

from models.gan.generator import Generator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gen = Generator(in_channels=3, out_channels=3)

gen.load_state_dict(torch.load('trained_models/GAN_alternate_discriminator/last_generator.pth'))

gen.to(device)

img = 'chair/n02738535_419-1.png'

corrupted_img_path = 'datasets/Sketchy/corrupted/all_images/' + img
original_img_path = 'datasets/Sketchy/original/all_images/' + img

corrupted = cv2.imread(corrupted_img_path)
corrupted = cv2.resize(corrupted, (256, 256))
corrupted = cv2.cvtColor(corrupted, cv2.COLOR_BGR2RGB)

original = cv2.imread(original_img_path)
original = cv2.resize(original, (256, 256))
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

# Convert the image to PyTorch tensor and add batch dimension
tensor_image = transforms.ToTensor()(corrupted)
tensor_image = tensor_image.unsqueeze(dim=0)
tensor_image = tensor_image.to(device)

with torch.no_grad():
    gen.eval()
    generated_img = gen(tensor_image)

np_img = generated_img.squeeze(0).cpu().numpy()

np_img = np.transpose(np_img, (1, 2, 0))
np_img = np.clip(np_img, 0, 1)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original)
plt.title('Original Image')
plt.axis('off')

# Display generated image
plt.subplot(1, 2, 2)
plt.imshow(np_img)
plt.title('Generated Image')
plt.axis('off')

plt.tight_layout()
plt.show()
