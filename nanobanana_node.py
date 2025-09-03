import os
import io
import json
import base64
import torch
import numpy as np
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

    def generate_image(self, prompt, api_key, model, aspect_ratio, seed=66666666, images=None):
        self.log_messages = []  # Reiniciar logs en cada ejecución
        response_text = ""
        
        # Determinar dimensiones para la imagen de placeholder en caso de error
        width, height = self.determine_dimensions_from_aspect_ratio(aspect_ratio)
        generated_image_tensor = self.generate_empty_image(width, height)

        try:
            actual_api_key = self.get_api_key(api_key)
            if not actual_api_key:
                error_message = "Error: No se proporcionó una clave de API válida."
                self.log(error_message)
                return (generated_image_tensor, error_message)

            # 1. ID de tu proyecto de Google Cloud
            project_id = "gen-lang-client-0587771574"

            # Endpoint de la API de Google para generación de imágenes
            api_endpoint = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/publishers/google/models/imagegeneration@006:predict"
            
            headers = {
                "Authorization": f"Bearer {actual_api_key}",
                "Content-Type": "application/json"
            }

            # 2. Mapeo para convertir la selección del menú al formato que la API necesita
            aspect_ratio_api_map = {
                "Landscape": "16:9",
                "Portrait": "9:16",
                "Square": "1:1",
                "Free": "1:1" # API de Google no tiene libre, se usa 1:1
            }
            google_aspect_ratio = aspect_ratio_api_map.get(aspect_ratio, "1:1")
            
            # 3. Payload CORRECTO para la API de Google
            payload = {
                "instances": [
                    {"prompt": prompt}
                ],
                "parameters": {
                    "sampleCount": 1,
                    "aspectRatio": google_aspect_ratio,
                    "seed": seed
                }
            }

            self.log(f"Enviando solicitud a la API de Google AI: {api_endpoint}")
            self.log(f"Payload: {json.dumps(payload, indent=2)}")

            # Realizar la llamada a la API
            api_response = requests.post(api_endpoint, headers=headers, json=payload, timeout=90)
            api_response.raise_for_status()
            self.log(f"Respuesta de la API: {api_response.status_code}")

            response_data = api_response.json()
            
            # 4. LÓGICA CORREGIDA para procesar la respuesta de Google AI
            image_data = None
            if "predictions" in response_data and len(response_data["predictions"]) > 0:
                image_b64_data = response_data["predictions"][0].get("bytesBase64Encoded")
                if image_b64_data:
                    image_data = base64.b64decode(image_b64_data)
                    response_text += "Imagen recibida exitosamente desde Google AI.\n"
                    self.log("Imagen decodificada desde base64.")
            
            if image_data:
                pil_image = Image.open(io.BytesIO(image_data))
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                img_array = np.array(pil_image).astype(np.float32) / 255.0
                generated_image_tensor = torch.from_numpy(img_array).unsqueeze(0)
                self.log(f"Imagen convertida a tensor: {generated_image_tensor.shape}")
            else:
                error_message = "Error: La respuesta de la API no contenía datos de imagen válidos."
                self.log(error_message)
                response_text += f"{error_message}\nRespuesta completa: {json.dumps(response_data)}"

        except requests.exceptions.RequestException as req_err:
            error_message = f"Error en la solicitud a la API: {req_err}"
            self.log(error_message)
            if hasattr(req_err, 'response') and req_err.response is not None:
                self.log(f"Detalle del error de la API: {req_err.response.text}")
                response_text += f"Detalle del error: {req_err.response.text}\n"
        except Exception as e:
            error_message = f"Ocurrió un error inesperado: {e}"
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