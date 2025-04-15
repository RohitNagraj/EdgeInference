from MobileVLM.scripts.inference import inference_once

model_path = "mtgv/MobileVLM-1.7B"
image_file = "MobileVLM/assets/samples/demo.jpg"
prompt_str = "Who is the author of this book?\nAnswer the question using a single word or phrase."

args = type('Args', (), {
    "model_path": model_path,
    "image_file": image_file,
    "prompt": prompt_str,
    "conv_mode": "v1",
    "temperature": 0, 
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512,
    "load_8bit": False,
    "load_4bit": False,
})()

output = inference_once(args)
print(f"LLM: {output}")