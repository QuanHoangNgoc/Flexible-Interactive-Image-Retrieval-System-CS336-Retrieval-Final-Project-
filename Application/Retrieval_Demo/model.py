from transformers import AutoProcessor, AutoModel, AutoTokenizer
import torch


class SigLipEncoder:
    def __init__(self):
        self.Model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        self.Processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.Tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")

    def form_model(self, dtype=None, device=None, grad_mode=False, eval_model=True):
        """Moves and converts the model to the specified dtype and device."""
        self.Model.to(dtype=dtype, device=device)
        for param in self.Model.parameters():
            param.requires_grad = grad_mode
        if eval_model:
            self.Model.eval()
        return self

    def infor_model(self):
        cnt = 0
        params = [(name, param) for name, param in self.Model.named_parameters()]
        for name, param in params:
            cnt += 1
            if cnt < 3 or cnt > len(params) - 3:
                print(f"Parameter: {name}")
                print(f"  Data type: {param.dtype}")
                print(f"  Requires gradient: {param.requires_grad}")
                print(f"  Device: {param.device}")
                print("-" * 20)  # Separator for readability

    def form_ts(self, inputs_dict):
        dtype = self.Model.dtype
        device = self.Model.device
        for key, value in inputs_dict.items():
            if isinstance(value, torch.Tensor):
                inputs_dict[key] = value.to(device=device)
                if (
                    dtype is not None and value.dtype.is_floating_point
                ):  # only convert float tensor
                    inputs_dict[key] = inputs_dict[key].type(dtype)
        return inputs_dict

    def get_np_text(self, str_chunk):
        inputs_ts = self.Tokenizer(str_chunk, padding="max_length", return_tensors="pt")
        inputs_ts = self.form_ts(inputs_ts)
        with torch.inference_mode():
            text_features = (
                self.Model.get_text_features(**inputs_ts).cpu().numpy()
            )  # Move to CPU before converting to NumPy
        return text_features

    def get_np_image(self, pil_chunk):
        inputs_ts = self.Processor(images=pil_chunk, return_tensors="pt")
        inputs_ts = self.form_ts(inputs_ts)
        with torch.inference_mode():
            image_features = (
                self.Model.get_image_features(**inputs_ts).cpu().numpy()
            )  # Move to CPU before converting to NumPy
        return image_features


siglip = SigLipEncoder()
siglip = siglip.form_model(dtype=torch.float32, device="cpu")
siglip.infor_model()
