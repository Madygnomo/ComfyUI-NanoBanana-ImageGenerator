import os
import io
import json
import torch
import numpy as np
from PIL import Image
import requests
import traceback

class NanoBananaImageGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        # Aquí definiremos las entradas del nodo más adelante
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "a cute cat, high quality, 4k"}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (["sdxl-turbo", "stable-diffusion-2.1"], {"default": "sdxl-turbo"}), # Asumiendo estos modelos, ajusta si Nanobanana tiene otros
                "width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "cfg_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "ugly, bad art, deformed"}),
                # "images": ("IMAGE",), # Por ahora, omitiremos la entrada de imagen para simplificar
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "API Response Log")
    FUNCTION = "generate_image"
    CATEGORY = "NanoBanana"

    def __init__(self):
        self.log_messages = []
        self.node_dir = os.path.dirname(os.path.abspath(__file__))
        self.key_file = os.path.join(self.node_dir, "nanobanana_api_key.txt")

    def log(self, message):
        """Global logging function: records messages to a list."""
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        return message

    def get_api_key(self, user_input_key):
        """Fetches API key, prioritizing user input."""
        if user_input_key and len(user_input_key) > 10:
            self.log("Using user-provided API key.")
            try:
                with open(self.key_file, "w") as f:
                    f.write(user_input_key)
                self.log("API key saved to node directory.")
            except Exception as e:
                self.log(f"Failed to save API key: {e}")
            return user_input_key

        if os.path.exists(self.key_file):
            try:
                with open(self.key_file, "r") as f:
                    saved_key = f.read().strip()
                if saved_key and len(saved_key) > 10:
                    self.log("Using saved API key.")
                    return saved_key
            except Exception as e:
                self.log(f"Failed to read saved API key: {e}")

        self.log("Warning: No valid API key provided.")
        return ""

    def generate_empty_image(self, width=512, height=512):
        """Generates a standard format empty RGB image tensor."""
        empty_image = np.ones((height, width, 3), dtype=np.float32) * 0.2
        tensor = torch.from_numpy(empty_image).unsqueeze(0)
        self.log(f"Creating ComfyUI-compatible empty image: Shape={tensor.shape}, Type={tensor.dtype}")
        return tensor

    def generate_image(self, prompt, api_key, model, width, height, steps, cfg_scale, seed=0, negative_prompt=""):
        self.log_messages = [] # Reset logs for each run
        response_text = ""
        generated_image_tensor = self.generate_empty_image(width, height) # Default to empty image

        try:
            actual_api_key = self.get_api_key(api_key)
            if not actual_api_key:
                error_message = "Error: No valid API key provided. Please enter your NanoBanana API key or ensure it's saved."
                self.log(error_message)
                full_log = "## Error\n" + error_message + "\n\n## Usage Instructions\n1. Enter your NanoBanana API Key in the node.\n2. The key will be automatically saved to the node directory for future use."
                return (generated_image_tensor, full_log)

            # Handle seed
            if seed == 0:
                import random
                seed = random.randint(1, 2**31 - 1)
                self.log(f"Generated random seed: {seed}")
            else:
                self.log(f"Using specified seed: {seed}")

            # Define Nanobanana API endpoint and headers
            # THIS IS A PLACEHOLDER. YOU NEED TO REPLACE WITH ACTUAL NANOBANANA API INFO.
            # Consult Nanobanana documentation for correct endpoint and request body.
            api_endpoint = "YOUR_NANOBANANA_API_ENDPOINT_HERE"
            headers = {
                "Authorization": f"Bearer {actual_api_key}",
                "Content-Type": "application/json"
            }

            # Construct the request payload based on Nanobanana's API
            # This is a generic example, you MUST adapt it to Nanobanana's specific requirements.
            payload = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "model": model,
                "width": width,
                "height": height,
                "num_inference_steps": steps, # Common parameter name
                "guidance_scale": cfg_scale, # Common parameter name
                "seed": seed,
                "output_format": "jpeg", # Requesting JPEG for smaller size, adjust if needed
            }
            self.log(f"Sending request to Nanobanana API with payload: {json.dumps(payload, indent=2)}")

            # Make the API call
            api_response = requests.post(api_endpoint, headers=headers, json=payload)
            api_response.raise_for_status() # Raise an exception for HTTP errors
            self.log(f"API Response Status: {api_response.status_code}")

            # Parse the response
            response_data = api_response.json()
            # print(json.dumps(response_data, indent=2)) # For debugging

            # --- Assuming Nanobanana returns a base64 encoded image ---
            # You will need to check Nanobanana's documentation for the actual structure
            # of the response and where the image data is located.
            image_b64 = None
            if "image_base64" in response_data: # Common pattern
                image_b64 = response_data["image_base64"]
                response_text += "Image data received successfully.\n"
            elif "output" in response_data and isinstance(response_data["output"], list) and len(response_data["output"]) > 0:
                # Some APIs return an array of images or a structure where "output" contains data
                if "base64" in response_data["output"][0]: # Another common pattern
                     image_b64 = response_data["output"][0]["base64"]
                     response_text += "Image data received from 'output' field.\n"
                elif "url" in response_data["output"][0]:
                    image_url = response_data["output"][0]["url"]
                    self.log(f"Image URL received: {image_url}. Downloading image...")
                    image_response = requests.get(image_url)
                    image_response.raise_for_status()
                    image_data = image_response.content
                    self.log(f"Downloaded image: {len(image_data)} bytes.")
                    pil_image = Image.open(io.BytesIO(image_data))
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    img_array = np.array(pil_image).astype(np.float32) / 255.0
                    generated_image_tensor = torch.from_numpy(img_array).unsqueeze(0)
                    response_text += f"Image downloaded from URL: {image_url}\n"
                    self.log(f"Image converted to tensor, shape: {generated_image_tensor.shape}")

            if image_b64:
                image_data = base64.b64decode(image_b64)
                pil_image = Image.open(io.BytesIO(image_data))
                self.log(f"Successfully opened image: {pil_image.width}x{pil_image.height}, format: {pil_image.format}")

                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')

                img_array = np.array(pil_image).astype(np.float32) / 255.0
                generated_image_tensor = torch.from_numpy(img_array).unsqueeze(0)
                self.log(f"Image converted to tensor successfully, shape: {generated_image_tensor.shape}")
            else:
                self.log("No base64 image data or URL found in the API response. Check API documentation.")
                response_text += "No image data found in response.\n"

        except requests.exceptions.RequestException as req_err:
            error_message = f"API Request Error: {req_err}"
            self.log(error_message)
            if hasattr(req_err, 'response') and req_err.response is not None:
                self.log(f"API Response Content: {req_err.response.text}")
            response_text += f"Request failed: {req_err}\n"
        except json.JSONDecodeError as json_err:
            error_message = f"JSON Decode Error: {json_err}. Response: {api_response.text}"
            self.log(error_message)
            response_text += f"JSON decoding failed: {json_err}\n"
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            self.log(f"NanoBanana Image Generation Error: {error_message}")
            traceback.print_exc()
            response_text += f"Error: {e}\n"

        full_log = "## Processing Log\n" + "\n".join(self.log_messages) + "\n\n## API Response Text\n" + response_text
        return (generated_image_tensor, full_log)

# Register the node
NODE_CLASS_MAPPINGS = {
    "NanoBanana_ImageGenerator": NanoBananaImageGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBanana_ImageGenerator": "NanoBanana Image Generator"
}