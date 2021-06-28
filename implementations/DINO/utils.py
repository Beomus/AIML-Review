import math
from PIL import Image
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import warnings


class DataAugmentation:
    """
    Create crops of an input iamge together with additiona augmentation.

    It generates 2 global crops and `n_local_crops` local crops.

    Parameters
    ----------
    global_crops_scale : tuple
        Range of sizes for the global crops.

    local_crops_scale : tuple
        Range of sizes for the local crops.

    n_local_crops : int
        Number of local crops to create.

    size : int
        The size of the final image.
    """

    def __init__(
        self,
        global_crops_scale=(0.4, 1),
        local_crops_scale=(0.05, 0.4),
        n_local_crops=8,
        size=224,
    ):
        self.n_local_crops = n_local_crops

        def RandomGaussianBlur(p):
            return transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2))],
                p=p,
            )

        flip_and_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.2,
                            hue=0.1,
                        )
                    ]
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.global_1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=size,
                    scale=global_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_jitter,
                RandomGaussianBlur(1.0),
                normalize,
            ]
        )

        self.global_2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=size,
                    scale=global_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_jitter,
                RandomGaussianBlur(0.1),
                transforms.RandomSolarize(170, p=0.2),
                normalize,
            ]
        )

        self.local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=size,
                    scale=local_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_jitter,
                RandomGaussianBlur(0.5),
                normalize,
            ]
        )

    def __call__(self, img: Image) -> List[torch.Tensor]:
        """
        Apply transformations.

        Parameters
        ----------
        img: PIL.Image
            Input image.

        Returns
        -------
        all_crops: list
            List of `torch.Tensor` representing different views of the input.
        """
        all_crops = []
        all_crops.append(self.global_1(img))
        all_crops.append(self.global_2(img))

        all_crops.extend([self.local(img) for _ in range(self.n_local_crops)])

        return all_crops


class MultiCropWrapper(nn.Module):
    """
    Convenience class for forward pass of multiple crops.

    Parameters
    ----------
    backbone: ViT
        Vision Transformer model replacing the head with `nn.Identity`

    head: MlpHead
        New head that is going to be on top of the `backbone`.
    """

    def __init__(self, backbone, head):
        super().__init__()
        backbone.head = nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Run the forward pass.

        The different crops are concatenated along the batch dimension and then
        a single forward pass is run. The resulting tensor is then chunked back
        to per crop tensors.

        Parameters
        ----------
        x: list
            List of `torch.Tensor` with each of shape `(n_samples, 3, size, size)

        Returns
        -------
        tuple
            Tuple of `torch.Tensor` each of shape `(n_samples, out_dim)`
        """

        n_crops = len(x)
        concatenated = torch.cat(x, dim=0)  # (n_samples * n_crops, 3, size, size)
        cls_embeddings = self.backbone(concatenated)  # (n_samples * n_crops, in_dim)
        logits = self.head(cls_embeddings)  # (n_samples * n_crops, out_dim)
        chunks = logits.chunk(n_crops)

        return chunks


class Loss(nn.Module):
    """
    The loss function.

    Subclassing `nn.Module` to create a buffer for the logits center of the teacher.

    Parameters
    ----------
    out_dim: int
        The dimensionality of the final layer to compute softmax over.

    teacher_temp, student_temp: float
        Softmax temperature of the teacher resp. student.

    center_momentum: float
        Hyperparameter for the exponential moving average that determines the
        center logits. The higher the more the more the running avg matters.
    """

    def __init__(
        self,
        out_dim: int,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        """
        Evaluate loss.

        Parameters
        ----------
        student_output, teacher_output: tuple
            Tuple of tensors of shape `(n_samples, out_dim)` representing logits.
            The length is equal to the number of crops.
            Note that student processed all crops and that the two initial crops
            are the global ones.

        Returns
        -------
        loss: torch.Tensor
            Scalar representing the average loss.
        """

        student_temp = [s / self.student_temp for s in student_output]
        teacher_temp = [(t - self.center) / self.teacher_temp for t in teacher_output]

        student_sm = [F.softmax(s, dim=-1) for s in student_temp]
        teacher_sm = [F.softmax(t, dim=-1).detach() for t in teacher_temp]

        total_loss = 0
        n_loss_terms = 0

        for t_ix, t in enumerate(teacher_sm):
            for s_ix, s in enumerate(student_sm):
                if t_ix == s_ix:
                    continue

                loss = torch.sum(-t * s, dim=-1)  # (n_samples, )
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.

        Compute the exponential moving average.

        Parameters
        ----------
        teacher_output : tuple
            Tuple of tensors of shape `(n_samples, out_dim)` where each tensor
            represent a different crop.
        """
        batch_center = torch.cat(teacher_output).mean(dim=0, keepdim=True)
        # (1, out_dim)

        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


def clip_gradients(model, clip=0.2):
    """
    Rescale norm of computed gradients.

    Parameters
    ----------
    model: nn.Module

    clip: float
        Maxium norm.
    """
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # -> Copied and pasted from https://github.com/facebookresearch/dino/
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
