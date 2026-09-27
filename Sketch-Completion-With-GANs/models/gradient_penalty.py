import torch


def gradient_penalty(critic, real, fake):
    batch_size = real.size(0)
    epsilon = torch.rand((batch_size, 1, 1, 1), device=real.device)
    interpolated = (epsilon * real + (1 - epsilon) * fake.detach()).requires_grad_(True)

    mixed_scores = critic(interpolated)
    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient_norm = gradient.view(batch_size, -1).norm(2, dim=1)
    return torch.mean((gradient_norm - 1) ** 2)
