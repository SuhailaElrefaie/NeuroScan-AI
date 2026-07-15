import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM: # Generate a Grad-CAM heatmap for a segmentation model.
    """
    The class stores:
    - activations from a chosen convolutional layer
    - gradients flowing back through that layer

    These are combined to produce a heatmap of the regions that influenced the model prediction.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # Hooks save the layer output and its gradients during prediction
        self.forward_hook = target_layer.register_forward_hook(
            self.save_activation
        )
        self.backward_hook = target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output): # Save feature maps produced during the forward pass.
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output): # Save gradients produced during the backward pass.
        self.gradients = grad_output[0]

    def generate(self, input_tensor): # Generate a normalized Grad-CAM heatmap for one input image.
        """
        For segmentation, the heatmap target is based on the model's soft tumor
        prediction scores across the image. This keeps the explanation focused
        on likely tumor regions without forcing it into a very small hard mask.

        Returns:
            A 2D NumPy array containing heatmap values between 0 and 1.
        """

        self.model.zero_grad()

        # Run a forward pass with gradients enabled
        output = self.model(input_tensor)
        probability_map = torch.sigmoid(output)

        # Use the soft tumor prediction as the explanation target
        target = (output * probability_map).sum()
        target.backward()

        gradients = self.gradients
        activations = self.activations

        # Average gradients over the spatial dimensions to obtain channel weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Combine activation maps using the gradient-based weights
        cam = (weights * activations).sum(dim=1, keepdim=True)

        # Keep only positive contributions
        cam = F.relu(cam)

        # Resize the heatmap to match the input image size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().detach().cpu().numpy()

        # Normalize values to the range 0 to 1
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self): # Remove the stored forward and backward hooks.
        """
        This prevents hooks from staying attached after the heatmap is created.
        """
        self.forward_hook.remove()
        self.backward_hook.remove()


def create_gradcam_overlay(image_pil, cam, heatmap_alpha=0.40): # Create a colored Grad-CAM heatmap and overlay it on an MRI image.
    """
    Returns:
        A tuple containing:
        - the MRI image with the heatmap overlaid
        - the heatmap image by itself
    """

    image_rgb = np.array(image_pil.convert("RGB"))

    # Convert the normalized heatmap into a colored OpenCV heatmap
    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend the original MRI image with the heatmap
    overlay = cv2.addWeighted(
        image_rgb,
        1 - heatmap_alpha,
        heatmap,
        heatmap_alpha,
        0
    )

    return Image.fromarray(overlay), Image.fromarray(heatmap)