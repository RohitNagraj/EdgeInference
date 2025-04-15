from tqdm import tqdm

from data.dataloader import Dataloader
from MobileVLM.scripts.inference import inference_once


class Evaluate:
    def __init__(self):
        self.dataloader = Dataloader("data")
        self.model_path = "mtgv/MobileVLM-1.7B"

    def evaluate(self):
        data_len = len(self.dataloader)
        correct = 0
        total = 0
        for idx in tqdm(range(data_len)):
            item = self.dataloader.get_item()

            args = type('Args', (), {
                "model_path": self.model_path,
                "image_file": item['image_path'],
                "prompt": item['prompt'],
                "conv_mode": "v1",
                "temperature": 0,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 64,
                "load_8bit": False,
                "load_4bit": False,
            })()
            
            output = inference_once(args)
            if item['label'] in output:
                correct += 1
            total += 1
        return correct / total
            
            
if __name__ == "__main__":
    evaluator = Evaluate()
    accuracy = evaluator.evaluate()
    print(f"Accuracy: {accuracy}")
    