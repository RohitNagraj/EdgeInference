from tqdm import tqdm

from data.dataloader import Dataloader
from MobileVLM.scripts.inference import inference_once


class Evaluate:
    def __init__(self, model=None):
        self.dataloader = Dataloader("data")
        self.model_path = "mtgv/MobileVLM-1.7B"
        self.eval_size = 100
        self.model = model

    def evaluate(self):
        data_len = len(self.dataloader)
        correct = 0
        total = 0
        for idx in tqdm(range(self.eval_size)):
            item = self.dataloader.get_item()

            args = type('Args', (), {
                "model_path": self.model_path,
                "image_file": item['image_path'],
                "prompt": item['prompt'],
                "conv_mode": "v1",
                "temperature": 0,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 16,
                "load_8bit": False,
                "load_4bit": False,
            })()

            if self.model is not None:
                output = inference_once(args, model_custom=self.model)
            else:
                output = inference_once(args)
            if item['label'] in output:
                correct += 1
            total += 1
            print(item['prompt'])
            print("Assistant: ", output)
        return correct / total
            
            
if __name__ == "__main__":
    evaluator = Evaluate()
    accuracy = evaluator.evaluate()
    print(f"Accuracy: {accuracy}")
    