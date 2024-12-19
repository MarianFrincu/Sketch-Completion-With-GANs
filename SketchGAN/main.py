from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch

from models.gan.generator import Generator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gen = Generator(in_channels=1, out_channels=1)

gan_save = torch.load('trained_models/GrayscaleGAN/epoch_40_gan.pth', weights_only=True, map_location=device)

gen.load_state_dict(gan_save['generator_state_dict'])

gen.to(device)

img = 'armor/n03000247_33355-6.png'

corrupted_img_path = 'datasets/Sketchy/corrupted/all_images/' + img
original_img_path = 'datasets/Sketchy/original/all_images/' + img

original = Image.open(original_img_path).convert('RGB')
corrupted = Image.open(corrupted_img_path).convert('RGB')

gan_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

tensor_image = gan_transform(corrupted).unsqueeze(0).to(device)

gen.eval()
with torch.no_grad():
    generated_img = gen(tensor_image)


np_img = generated_img.squeeze(0).repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy()

plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(original)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(corrupted)
plt.title('Corrupted Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(np_img)
plt.title('Generated Image')
plt.axis('off')

plt.tight_layout()
plt.show()
