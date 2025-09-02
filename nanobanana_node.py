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
                "model": (["models/gemini-2.0-flash-preview-image-generation", "models/gemini-2.0-flash-exp", "models/gemini-2.5-flash-image-preview"], {"default": "models/gemini-2.5-flash-image-preview"}),
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

    def generate_image(self, prompt, api_key, model, aspect_ratio, temperature, seed=66666666, images=None):
        self.log_messages = []  # Reset logs for each run
        response_text = ""
        
        # Determinar dimensiones basadas en aspect_ratio
        width, height = self.determine_dimensions_from_aspect_ratio(aspect_ratio)
        generated_image_tensor = self.generate_empty_image(width, height)

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

            self.log(f"Using dimensions: {width}x{height} for aspect ratio: {aspect_ratio}")
            self.log(f"Temperature: {temperature}, Model: {model}")

            # ENDPOINT PLACEHOLDER - Reemplaza con la URL real de NanoBanana
            api_endpoint = "https://api.nanobanana.com/v1/generate"  # URL de ejemplo
            
            # Si no tienes el endpoint real aún, genera una imagen de prueba
            if "example" in api_endpoint or not actual_api_key.startswith("nb-"):  # Asumiendo que las keys de NanoBanana empiezan con "nb-"
                self.log("WARNING: Using placeholder API endpoint or test key. Generating test image.")
                
                # Crear una imagen de prueba con información del prompt
                test_img = Image.new('RGB', (width, height), color='lightblue')
                draw = ImageDraw.Draw(test_img)
                
                # Texto de prueba con información de parámetros
                test_text = f"NanoBanana Test Image\n\nPrompt: {prompt[:100]}...\nModel: {model}\nAspect: {aspect_ratio}\nTemp: {temperature}\nSeed: {seed}"
                
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
                
                # Dividir texto en líneas para mejor presentación
                lines = test_text.split('\n')
                y_pos = 20
                for line in lines:
                    draw.text((20, y_pos), line, fill='black', font=font)
                    y_pos += 30
                
                # Convertir a tensor
                img_array = np.array(test_img).astype(np.float32) / 255.0
                generated_image_tensor = torch.from_numpy(img_array).unsqueeze(0)
                response_text = f"Test image generated (API endpoint not configured)\nDimensions: {width}x{height}\nPrompt: {prompt}"
                self.log("Test image created successfully")
                
            else:
                # Código real de API cuando tengas el endpoint configurado
                headers = {
                    "Authorization": f"Bearer {actual_api_key}",
                    "Content-Type": "application/json"
                }

                # Adapta este payload según la documentación real de NanoBanana
                payload = {
                    "prompt": prompt,
                    "model": model,
                    "aspect_ratio": aspect_ratio,
                    "temperature": temperature,
                    "seed": seed,
                    "width": width,
                    "height": height,
                }

                # Si hay imágenes de referencia, procesarlas
                if images is not None:
                    try:
                        # Procesar imágenes de referencia (similar al ejemplo de Gemini)
                        reference_images = []
                        batch_size = images.shape[0]
                        self.log(f"Processing {batch_size} reference images")
                        
                        for i in range(batch_size):
                            input_image = images[i].cpu().numpy()
                            input_image = (input_image * 255).astype(np.uint8)
                            pil_image = Image.fromarray(input_image)
                            
                            # Convertir a base64 para enviar en la API
                            img_byte_arr = io.BytesIO()
                            pil_image.save(img_byte_arr, format='PNG')
                            img_byte_arr.seek(0)
                            img_b64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')
                            reference_images.append(img_b64)
                            
                        payload["reference_images"] = reference_images
                        self.log(f"Added {len(reference_images)} reference images to request")
                        
                    except Exception as img_error:
                        self.log(f"Reference image processing error: {str(img_error)}")
                
                self.log(f"Sending request to NanoBanana API")

                # Realizar la llamada a la API
                api_response = requests.post(api_endpoint, headers=headers, json=payload, timeout=60)
                api_response.raise_for_status()
                self.log(f"API Response Status: {api_response.status_code}")

                response_data = api_response.json()
                
                # Procesar respuesta (adapta según la respuesta real de NanoBanana)
                image_data = None
                if "image" in response_data:
                    if "base64" in response_data["image"]:
                        image_data = base64.b64decode(response_data["image"]["base64"])
                        response_text += "Image data received successfully.\n"
                    elif "url" in response_data["image"]:
                        image_url = response_data["image"]["url"]
                        self.log(f"Image URL received: {image_url}. Downloading image...")
                        image_response = requests.get(image_url, timeout=30)
                        image_response.raise_for_status()
                        image_data = image_response.content
                        response_text += f"Image downloaded from URL: {image_url}\n"

                if image_data:
                    pil_image = Image.open(io.BytesIO(image_data))
                    self.log(f"Successfully opened image: {pil_image.width}x{pil_image.height}")

                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')

                    img_array = np.array(pil_image).astype(np.float32) / 255.0
                    generated_image_tensor = torch.from_numpy(img_array).unsqueeze(0)
                    self.log(f"Image converted to tensor successfully, shape: {generated_image_tensor.shape}")
                else:
                    self.log("No image data found in the API response.")
                    response_text += "No image data found in response.\n"

        except requests.exceptions.RequestException as req_err:
            error_message = f"API Request Error: {req_err}"
            self.log(error_message)
            if hasattr(req_err, 'response') and req_err.response is not None:
                try:
                    error_detail = req_err.response.text
                    self.log(f"API Error Response: {error_detail}")
                except:
                    pass
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