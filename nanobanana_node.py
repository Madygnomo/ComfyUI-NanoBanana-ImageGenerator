import os
import io
import json
import base64  # ¡FALTABA ESTE IMPORT!
import torch
import numpy as np
from PIL import Image
import requests
import traceback

class NanoBananaImageGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (["models/gemini-2.5-flash-image-preview"], {"default": "models/gemini-2.5-flash-image-preview"}),
                "aspect_ratio": ([
                    "Free (自由比例)",
                    "Landscape (横屏)",
                    "Portrait (竖屏)",
                    "Square (方形)",
                ], {"default": "Free (自由比例)"}),
                "temperature": ("FLOAT", {"default": 1, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "seed": ("INT", {"default": 66666666, "min": 0, "max": 2147483647}),
                "images": ("IMAGE",),
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
        print(f"[NanoBanana] {message}")  # También imprime en consola para debug
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
        self.log_messages = []  # Reset logs for each run
        response_text = ""
        generated_image_tensor = self.generate_empty_image(width, height)  # Default to empty image

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

            # CAMBIO IMPORTANTE: En lugar de fallar, devuelve una imagen de prueba
            # Reemplaza esta URL con el endpoint real de NanoBanana cuando lo tengas
            api_endpoint = "https://api.nanobanana.com/generate"  # URL de ejemplo
            
            # Si no tienes el endpoint real aún, genera una imagen de prueba
            if "example" in api_endpoint or "YOUR_" in api_endpoint:
                self.log("WARNING: Using placeholder API endpoint. Generating test image.")
                # Crear una imagen de prueba con texto
                from PIL import ImageDraw, ImageFont
                test_img = Image.new('RGB', (width, height), color='lightblue')
                draw = ImageDraw.Draw(test_img)
                
                # Texto de prueba
                test_text = f"Test Image\n{prompt[:50]}..."
                try:
                    # Intenta usar una fuente por defecto
                    font = ImageFont.load_default()
                except:
                    font = None
                
                draw.text((10, 10), test_text, fill='black', font=font)
                
                # Convertir a tensor
                img_array = np.array(test_img).astype(np.float32) / 255.0
                generated_image_tensor = torch.from_numpy(img_array).unsqueeze(0)
                response_text = f"Test image generated (API endpoint not configured)\nPrompt: {prompt}"
                self.log("Test image created successfully")
                
            else:
                # Código real de API cuando tengas el endpoint
                headers = {
                    "Authorization": f"Bearer {actual_api_key}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "model": model,
                    "width": width,
                    "height": height,
                    "num_inference_steps": steps,
                    "guidance_scale": cfg_scale,
                    "seed": seed,
                    "output_format": "jpeg",
                }
                
                self.log(f"Sending request to NanoBanana API with payload: {json.dumps(payload, indent=2)}")

                # Make the API call
                api_response = requests.post(api_endpoint, headers=headers, json=payload, timeout=30)
                api_response.raise_for_status()
                self.log(f"API Response Status: {api_response.status_code}")

                response_data = api_response.json()
                
                # Procesar respuesta (adapta según la API real)
                image_b64 = None
                if "image_base64" in response_data:
                    image_b64 = response_data["image_base64"]
                    response_text += "Image data received successfully.\n"
                elif "output" in response_data and isinstance(response_data["output"], list):
                    if len(response_data["output"]) > 0:
                        if "base64" in response_data["output"][0]:
                            image_b64 = response_data["output"][0]["base64"]
                            response_text += "Image data received from 'output' field.\n"
                        elif "url" in response_data["output"][0]:
                            image_url = response_data["output"][0]["url"]
                            self.log(f"Image URL received: {image_url}. Downloading image...")
                            image_response = requests.get(image_url, timeout=30)
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
                    self.log("No base64 image data or URL found in the API response.")
                    response_text += "No image data found in response.\n"

        except requests.exceptions.RequestException as req_err:
            error_message = f"API Request Error: {req_err}"
            self.log(error_message)
            if hasattr(req_err, 'response') and req_err.response is not None:
                self.log(f"API Response Content: {req_err.response.text}")
            response_text += f"Request failed: {req_err}\n"
        except json.JSONDecodeError as json_err:
            error_message = f"JSON Decode Error: {json_err}"
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