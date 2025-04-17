import sys
sys.path.append('./MobileVLM')

import torch
import torch.nn.utils.prune as prune

from MobileVLM.mobilevlm.model.mobilevlm import load_pretrained_model
from profile import get_model_size, get_model_size_bytes
from evaluate import Evaluate

class Pruner:
    def __init__(self, model):
        self.model = model

    def prune_granular(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Prune 30% of weights by L1 norm
                prune.l1_unstructured(module, name='weight', amount=0)
                prune.remove(module, 'weight')

if __name__ == "__main__":
    tokenizer, model, image_processor, context_len = load_pretrained_model("mtgv/MobileVLM-1.7B", load_8bit=False, load_4bit=False, device_map="auto", device="cuda")
    pruner = Pruner(model)
    pruner.prune_granular()
    model = pruner.model
    print(f"Pruned Model")
    print(f"Total parameters: {get_model_size(model)}")
    print(f"Model size in bytes: {get_model_size_bytes(model)}")

    evaluator = Evaluate(model=model)
    accuracy = evaluator.evaluate()
    print(f"Accuracy: {accuracy}")
