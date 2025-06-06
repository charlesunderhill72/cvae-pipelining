import os
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

