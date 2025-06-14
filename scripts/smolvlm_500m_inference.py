import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load images
image = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")

# Initialize processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
).to(DEVICE)

# Create input messages
messages = [
    # {
    #     "role": "system",
    #     "content": "You are a hipster assistant."
    # },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Can you describe this image?"}
        ]
    }
]


# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = inputs.to(DEVICE)

# Generate outputs
generated_ids = model.generate(**inputs, max_new_tokens=100)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(generated_texts[0])
"""
Assistant: The image depicts a cityscape featuring a prominent landmark, the Statue of Liberty, prominently positioned on Liberty Island. The statue is a green, humanoid figure with a crown atop its head and is situated on a small island surrounded by water. The statue is characterized by its large, detailed structure, with a statue of a woman holding a torch above her head and a tablet in her left hand. The statue is surrounded by a small, rocky island, which is partially visible in the foreground.
In the background, the cityscape is dominated by numerous high-rise buildings, which are densely packed and vary in height. The buildings are primarily made of glass and steel, reflecting the sunlight and creating a bright, urban skyline. The skyline is filled with various architectural styles, including modern skyscrapers and older, more traditional buildings.
The water surrounding the island is calm, with a few small boats visible, indicating that the area is likely a popular tourist destination. The water is a deep blue, suggesting that it is a large body of water, possibly a river or a large lake.
In the foreground, there is a small strip of land with trees and grass, which adds a touch of natural beauty to the urban landscape. The trees are green, indicating that it is likely spring or summer.
The image captures a moment of tranquility and reflection, as the statue and the cityscape come together to create a harmonious and picturesque scene. The statue's presence in the foreground draws attention to the city's grandeur, while the calm water and natural elements in the background provide a sense of peace and serenity.
In summary, the image showcases the Statue of Liberty, a symbol of freedom and democracy, set against a backdrop of a bustling cityscape. The statue is a prominent and iconic representation of human achievement, while the cityscape is a testament to human ingenuity and progress. The image captures the beauty and complexity of urban life, with the statue serving as a symbol of hope and freedom, while the cityscape provides a glimpse into the modern world.
"""
