from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer


class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, tokenizer_path: str = None, **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai") or (tokenizer_path and "tokenizer" in tokenizer_path and "tokenizer_2" not in tokenizer_path)
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        # 使用tokenizer_path如果提供，否则使用version
        tokenizer_version = tokenizer_path if tokenizer_path else version
        model_version = version

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(tokenizer_version, max_length=max_length)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(model_version, **hf_kwargs)
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(tokenizer_version, max_length=max_length)
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(model_version, **hf_kwargs)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
