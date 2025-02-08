# !pip install -q -U google-generativeai

import google.generativeai as genai
from PIL import Image
import os
import time


# Get API KEY from colab.
GOOGLE_API_KEY = "AIzaSyAF2vRk-cw5_6Gn-abNAED6D-HqQVxlOXc"
genai.configure(api_key=GOOGLE_API_KEY)


for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)


model_name = "gemini-1.5-flash"


# Define the ClientFactory class to manage API clients
class ClientFactory:
    def __init__(self):
        self.clients = {}

    def register_client(self, name, client_class):
        self.clients[name] = client_class

    def create_client(self, name, **kwargs):
        client_class = self.clients.get(name)
        if client_class:
            return client_class(**kwargs)
        raise ValueError(f"Client '{name}' is not registered.")


# Register and create the Google generative AI client
client_factory = ClientFactory()
client_factory.register_client("google", genai.GenerativeModel)

client_kwargs = {
    "model_name": model_name,
    "generation_config": {"temperature": 0.4},
    "system_instruction": None,
}

client = client_factory.create_client("google", **client_kwargs)


prompt_captioning = """
You are an experienced expert in data labeling and image description. Your task is to write an accurate and natural caption describing the content of the image. The caption should be concise, clear, and informative based on the following key aspects:

1. **Main subjects**: Identify the primary subjects in the image (people, animals, objects, landscapes) and their notable characteristics.
2. **Actions or situations**: Describe the actions, events, or situations taking place in the image.
3. **Context**: Include relevant contextual details, such as location, colors, states, emotions, or other important related elements.

### Requirements:
- **Respond in English only.**
- Ensure the caption is of the highest quality, accurate, coherent, and natural.
- Avoid adding inferred information that cannot be directly observed from the image.

### Examples:
1. Image: A dog lying on the grass.
   - Caption: "A golden retriever relaxing on a green lawn."
2. Image: A girl holding a balloon.
   - Caption: "A young girl in a white dress holding a red balloon under sunlight."

Carefully examine the image and provide a corresponding caption."""


def generate_caption(prompt_captioning, retrieved_image):
    caption_list = []

    for image in retrieved_image:
        try:
            contents = [prompt_captioning] + [image]
            response = client.generate_content(contents=contents, stream=True)
            response.resolve()

            try:
                # Check if 'candidates' list is not empty
                if response.candidates:
                    # Access the first candidate's content if available
                    if response.candidates[0].content.parts:
                        generated_text = response.candidates[0].content.parts[0].text
                        caption_list.append(generated_text)
                        time.sleep(3)
                    else:
                        print("No generated text found in the candidate.")
                        caption_list.append("Không có chú thích")
                else:
                    print("No candidates found in the response.")
                    caption_list.append("Không nhận được phản hổi")
            except (AttributeError, IndexError) as e:
                print("Error:", e)
                continue

        except genai.types.BlockedPromptException as e:
            print(f"Prompt blocked for image: {image}, reason: {e}")
            caption_list.append("Phản hồi tạo ra chứa nội dung cấm")
            continue  # Skip to the next image

    return caption_list


prompt_fuse = """
You are an expert in information synthesis, data analysis, and image retrieval optimization. You are provided with a list of descriptive captions and an initial **Retrieval Query**. Your task is to:

1. **Thoroughly analyze** each caption, identifying key points and main ideas in each description.
2. **Incorporate the Retrieval Query** into the synthesis process to ensure that the new query is aligned with the intent and context of the query.
3. **Synthesize the information** from all captions and the Retrieval Query to create a new, complete query that ensures:
   - All important information is fully integrated.
   - The new caption is accurate, coherent, and natural.
   - The caption aligns with the intent of the Retrieval Query to improve image retrieval performance.
   - Redundancies are avoided, and any conflicting or irrelevant details are removed.

**Input Format:**
[Retrieval Query]

[Caption 1,
Caption 2,
...
Caption n]

**Response Format:**
New Query

**Requirements:**
- **Respond in English only.**
- The new query must not exceed **50 words**.
- Just response the content of new query
- Ensure the query is concise yet comprehensive, natural, and easy to understand.
- Focus on combining information from captions and the query to produce a general, inclusive, and retrieval-enhancing caption.

## Example:

**Input:**
["A cat on a sofa"]

["A golden cat is sleeping on a sofa.",
"A cat is relaxing on a gray sofa."]

**Response:**
"A golden cat is relaxing on a gray sofa."

Carefully synthesize the captions and the query, and provide a new query.
"""


def fuse_caption(prompt_fuse, caption_list):
    fused_caption_list = []

    try:
        contents = [prompt_fuse] + caption_list
        response = client.generate_content(contents=contents, stream=True)
        response.resolve()

        try:
            # Check if 'candidates' list is not empty
            if response.candidates:
                # Access the first candidate's content if available
                if response.candidates[0].content.parts:
                    generated_text = response.candidates[0].content.parts[0].text
                    fused_caption_list.append(generated_text)
                    time.sleep(3)
                else:
                    print("No generated text found in the candidate.")
                    fused_caption_list.append("Không có chú thích")
            else:
                print("No candidates found in the response.")
                fused_caption_list.append("Không nhận được phản hổi")
        except (AttributeError, IndexError) as e:
            print("Error:", e)

    except genai.types.BlockedPromptException as e:
        print(f"Prompt blocked for image: , reason: {e}")
        fused_caption_list.append("Phản hồi tạo ra chứa nội dung cấm")

    return fused_caption_list


def edit_query(clicked_pils, text):
    caption_list = generate_caption(text, clicked_pils)
    fused_caption_list = fuse_caption(prompt_fuse, caption_list)
    new_cap = fused_caption_list[0]
    return new_cap
