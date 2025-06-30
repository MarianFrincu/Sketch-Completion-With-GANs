import torch


def gradient_penalty(critic, real, fake, device):
    batch_size, c, h, w = real.shape
    epsilon = torch.rand((batch_size, 1, 1, 1), device=device, requires_grad=True)
    interpolated_images = epsilon * real + (1 - epsilon) * fake
    interpolated_images.requires_grad_(True)

    mixed_scores = critic(interpolated_images)
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty
