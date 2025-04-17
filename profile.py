import sys

sys.path.append('./MobileVLM')

import torch

from MobileVLM.mobilevlm.model.mobilevlm import load_pretrained_model


def get_model_size(model):
    total_params = sum(param.count_nonzero() for param in model.parameters())
    return total_params


def get_trainable_model_size(model):
    trainable_params = sum(p.count_nonzero() for p in model.parameters() if p.requires_grad)
    return trainable_params

def get_model_size_bytes(model):
    total_size = 0
    for param in model.parameters():
        if param.is_floating_point():
          total_size += param.numel() * torch.finfo(param.dtype).bits
        else:
           total_size += param.numel() * torch.iinfo(param.dtype).bits
    return total_size / 8 #bits to bytes


if __name__ == "__main__":
    model_path = "mtgv/MobileVLM-1.7B"
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, load_8bit=False,
                                                                           load_4bit=False, device_map="auto",
                                                                           device="cuda")

    print(f"Model name: {model_path}")
    print(f"Total parameters: {get_model_size(model)}")
    print(f"Trainable parameters: {get_trainable_model_size(model)}")
    print(f"Model size in bytes: {get_model_size_bytes(model)}")
