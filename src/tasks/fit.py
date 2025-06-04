import os
import torch
import flytekit as fl

image_spec = fl.ImageSpec(
    # The name of the image. This image will be used by the `say_hello`` task.
    name="preprocessing-image",

    # Lock file with dependencies to be installed in the image.
    requirements="uv.lock",

    # Image registry to to which this image will be pushed.
    # Set the Environment variable FLYTE_IMAGE_REGISTRY to the URL of your registry.
    # The image will be built on your local machine, so enure that your Docker is running.
    # Ensure that pushed image is accessible to your Flyte cluster, so that it can pull the image
    # when it spins up the task container.
    registry=os.environ.get("FLYTE_IMAGE_REGISTRY", "ghcr.io/charlesunderhill72"),
    source_root="src"
    )

@fl.task(container_image=image_spec)
def final_loss(bce_loss: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    This function will add the reconstruction loss (BCELoss) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
