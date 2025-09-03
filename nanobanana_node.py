import os
import io
import json
import base64
import torch
import numpy as np
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
import requests
import traceback

class NanoBananaImageGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ([
                    "models/gemini-2.5-flash-image-preview",
                    "models/gemini-2.0-flash-preview-image-generation",
                    "models/gemini-2.0-flash-exp"
                ], {"default": "models/gemini-2.5-flash-image-preview"}),
                "aspect_ratio": ([
                    "Free",
                    "Landscape",
                    "Portrait",
                    "Square"
                ], {"default": "Free"})
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295}),
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

    def determine_dimensions_from_aspect_ratio(self, aspect_ratio):
        """Determina las dimensiones basadas en el aspect ratio seleccionado."""
        if "Free" in aspect_ratio:
            # Dimensiones libres, usa valores por defecto
            return 1024, 1024
        elif "Landscape" in aspect_ratio:
            # Horizontal - más ancho que alto
            return 1344, 768
        elif "Portrait" in aspect_ratio:
            # Vertical - más alto que ancho  
            return 768, 1344
        elif "Square" in aspect_ratio:
            # Cuadrado
            return 1024, 1024
        else:
            # Por defecto
            return 1024, 1024

    def generate_image(self, prompt, api_key, model, aspect_ratio, seed=0, images=None):
        self.log_messages = []
        response_text = ""
        
        width, height = self.determine_dimensions_from_aspect_ratio(aspect_ratio)
        generated_image_tensor = self.generate_empty_image(width, height)

        try:
            actual_api_key = self.get_api_key(api_key)
            if not actual_api_key:
                error_message = "Error: No API Key provided."
                self.log(error_message)
                return (generated_image_tensor, error_message)

            # 1. Configurar la API con la clave (¡Mucho más simple!)
            self.log("Configuring Google GenAI with API Key...")
            genai.configure(api_key=actual_api_key)

            # 2. Crear una instancia del modelo generativo
            # Nota: El modelo de imagen puede no usar todos los parámetros como seed o aspect_ratio
            self.log(f"Creating instance of model: {model}")
            model_instance = genai.GenerativeModel(model)

            # 3. Llamar a la API para generar la imagen
            self.log(f"Sending prompt to generate image...")
            # Usamos una configuración para pedir explícitamente una imagen
            generation_config = genai.types.GenerationConfig(
                response_mime_type="image/png"
            )
            response = model_instance.generate_content(prompt, generation_config=generation_config)
            
            self.log("Response received from API.")
            
            # 4. Procesar la respuesta para extraer los datos de la imagen
            image_data = None
            if response.parts and response.parts[0].inline_data:
                image_part = response.parts[0]
                if image_part.inline_data.mime_type.startswith("image/"):
                    image_data = image_part.inline_data.data
                    response_text += "Image data received successfully from GenAI API.\n"
                    self.log("Image data extracted from response.")

            if image_data:
                pil_image = Image.open(io.BytesIO(image_data))
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                img_array = np.array(pil_image).astype(np.float32) / 255.0
                generated_image_tensor = torch.from_numpy(img_array).unsqueeze(0)
            else:
                error_message = "Error: The API response did not contain valid image data."
                # A veces, si hay un error de seguridad, la respuesta viene en `response.text`
                if hasattr(response, 'text'):
                    error_message += f" API Text Response: {response.text}"
                self.log(error_message)
                response_text += error_message

        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            self.log(error_message)
            traceback.print_exc()
            response_text += f"Error: {e}\n"

        full_log = "## Log de Procesamiento\n" + "\n".join(self.log_messages) + "\n\n## Respuesta de la API\n" + response_text
        return (generated_image_tensor, full_log)

# Register the node
NODE_CLASS_MAPPINGS = {
    "NanoBanana_ImageGenerator": NanoBananaImageGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBanana_ImageGenerator": "NanoBanana Image Generator"
}