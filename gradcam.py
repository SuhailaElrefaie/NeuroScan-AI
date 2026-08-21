import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, inputs, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        self.model.zero_grad(set_to_none=True)

        output = self.model(input_tensor)
        probability_map = torch.sigmoid(output)
        target = (output * probability_map).sum()
        target.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM did not capture activations/gradients.")

        activations = self.activations
        gradients = self.gradients

        if activations.ndim == 4:
            spatial_dims = (2, 3)
            interpolation_mode = "bilinear"
        elif activations.ndim == 5:
            spatial_dims = (2, 3, 4)
            interpolation_mode = "trilinear"
        else:
            raise ValueError(
                f"Unsupported activation shape: {tuple(activations.shape)}"
            )

        weights = gradients.mean(dim=spatial_dims, keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode=interpolation_mode,
            align_corners=False,
        )

        cam = (
            cam.squeeze(0)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        cam_min = float(cam.min())
        cam_max = float(cam.max())

        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

        return cam

    def remove_hooks(self):
        if self.forward_hook is not None:
            self.forward_hook.remove()
            self.forward_hook = None

        if self.backward_hook is not None:
            self.backward_hook.remove()
            self.backward_hook = None


def create_gradcam_overlay(image_pil, cam, heatmap_alpha=0.40):
    if not isinstance(image_pil, Image.Image):
        image_pil = Image.fromarray(np.asarray(image_pil))

    image_rgb = np.array(image_pil.convert("RGB"))
    cam = np.asarray(cam, dtype=np.float32)

    if cam.ndim != 2:
        raise ValueError(
            "create_gradcam_overlay expects a 2D CAM slice. "
            f"Received shape: {cam.shape}"
        )

    if cam.shape != image_rgb.shape[:2]:
        cam = cv2.resize(
            cam,
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    cam = np.clip(cam, 0.0, 1.0)
    heatmap_uint8 = (cam * 255).astype(np.uint8)

    heatmap_bgr = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )
    heatmap_rgb = cv2.cvtColor(
        heatmap_bgr,
        cv2.COLOR_BGR2RGB,
    )

    alpha = float(np.clip(heatmap_alpha, 0.0, 1.0))

    overlay = cv2.addWeighted(
        image_rgb,
        1.0 - alpha,
        heatmap_rgb,
        alpha,
        0,
    )

    return Image.fromarray(overlay), Image.fromarray(heatmap_rgb)
