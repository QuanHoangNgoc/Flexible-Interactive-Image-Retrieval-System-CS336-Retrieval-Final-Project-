import numpy as np
import cv2
from PIL import Image
from transformers import AutoProcessor, CLIPSegForImageSegmentation
import torch
# from sklearn.preprocessing import RobustScaler, StandardScaler


class ClipSeg:
    def __init__(self):
        # Initialize processor, model, and translator
        self.Processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.Clip_model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        ).to(
            "cpu"
        )  #!!!

    def get_segment_image(self, image, text):
        image = image.convert("RGB").resize(
            (224, 224), Image.BILINEAR
        )  #!!! Ensure image form

        # Preprocess image and text
        inputs = self.Processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        # Model inference
        with torch.inference_mode():
            outputs = self.Clip_model(**inputs)
            logits = outputs.logits  # shape: [1, height, width]

        # Normalize logits
        logits = logits[0].cpu().numpy()  # Remove batch dimension #!!! to numpy

        # logits_normalized = logits
        logits_normalized = (logits - logits.mean()) / logits.std()
        # logits_normalized = (logits - logits.min()) / (
        #     logits.max() - logits.min()
        # )  # Normalize to [0, 1]

        # Split logits by value relative to 0
        right_side = logits_normalized[logits_normalized >= 0.0]
        left_side = logits_normalized[logits_normalized < 0.0]

        # Compute thresholds
        threshold_high = np.percentile(right_side, 85)  # Top 90% of right side
        threshold_low = np.percentile(right_side, 10)  # Bottom 10% of left side

        # Create binary mask
        binary_mask = (
            (logits_normalized <= threshold_high) & (logits_normalized >= threshold_low)
        ).astype(np.uint8) * 255

        # Find contours in the binary mask
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Convert PIL image to OpenCV format (RGB -> BGR)
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Draw contours on the original image
        cv2.drawContours(image_bgr, contours, -1, (0, 255, 0), 2)

        # Convert the result back to PIL image for visualization
        result_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

        return result_image


segmenter = ClipSeg()
