import os
import base64
import io
import json
import mimetypes
import traceback
from io import BytesIO

import numpy as np
import torch
from PIL import Image

from google import genai
from google.genai import types


# --- Persistencia de API key (estilo ComfyUI) ---
NODE_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.join(NODE_DIR, "gemini_api_key.txt")


def _save_api_key_to_file(api_key: str, logs: list):
    try:
        with open(KEY_FILE, "w", encoding="utf-8") as f:
            f.write(api_key.strip())
        logs.append(f"[INFO] API key guardada en: {KEY_FILE}")
    except Exception as e:
        logs.append(f"[WARN] No se pudo guardar la API key: {e}")


def _load_api_key_from_file(logs: list) -> str:
    try:
        if os.path.exists(KEY_FILE):
            with open(KEY_FILE, "r", encoding="utf-8") as f:
                key = f.read().strip()
                if key:
                    logs.append("[INFO] API key cargada desde archivo persistente")
                    return key
    except Exception as e:
        logs.append(f"[WARN] No se pudo leer {KEY_FILE}: {e}")
    return ""


def _resolve_api_key(user_input_key: str, logs: list) -> str:
    """
    Prioridad:
      1) api_key desde UI (y se guarda)
      2) KEY_FILE
      3) var de entorno GEMINI_API_KEY
    """
    if user_input_key and len(user_input_key.strip()) > 10:
        logs.append("[INFO] Usando API key provista por UI")
        _save_api_key_to_file(user_input_key.strip(), logs)
        return user_input_key.strip()

    file_key = _load_api_key_from_file(logs)
    if file_key and len(file_key) > 10:
        return file_key

    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key and len(env_key) > 10:
        logs.append("[INFO] Usando API key desde GEMINI_API_KEY")
        return env_key

    return ""


def _tensor_from_pil(pil_img: Image.Image, logs: list) -> torch.Tensor:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img).astype(np.float32) / 255.0  # HWC [0..1]
    tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W, 3]
    logs.append(f"[INFO] Imagen convertida a tensor: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    return tensor


def _bytes_from_inline_data(inline_data, logs: list) -> bytes:
    """
    Convierte inline_data.data a bytes:
      - Si es str: se asume base64 y se decodifica.
      - Si ya es bytes: se devuelve tal cual.
    """
    data = getattr(inline_data, "data", None)
    if data is None:
        raise ValueError("inline_data.data está vacío")

    if isinstance(data, bytes):
        logs.append(f"[INFO] inline_data.data recibido como bytes (len={len(data)})")
        return data

    if isinstance(data, str):
        try:
            decoded = base64.b64decode(data)
            logs.append(f"[INFO] inline_data.data decodificado desde base64 (len={len(decoded)})")
            return decoded
        except Exception as e:
            raise ValueError(f"No se pudo decodificar base64: {e}")

    raise TypeError(f"Tipo inesperado en inline_data.data: {type(data)}")


def _hint_from_aspect_ratio(ar: str) -> str:
    if "Landscape" in ar:
        return "Generate as a wide rectangular image (width > height)."
    if "Portrait" in ar:
        return "Generate as a tall rectangular image (height > width)."
    if "Square" in ar:
        return "Generate as a square image (width = height)."
    return ""  # Free


class GeminiImageGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (
                    [
                        "models/gemini-2.0-flash-preview-image-generation",
                        "models/gemini-2.0-flash-exp",
                        "models/gemini-2.5-flash-image-preview",
                    ],
                    {"default": "models/gemini-2.5-flash-image-preview"},
                ),
                "aspect_ratio": (
                    [
                        "Free (自由比例)",
                        "Landscape (横屏)",
                        "Portrait (竖屏)",
                        "Square (方形)",
                    ],
                    {"default": "Free (自由比例)"},
                ),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "seed": ("INT", {"default": 66666666, "min": 0, "max": 2147483647}),
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "API Respond")
    FUNCTION = "generate_image"
    CATEGORY = "Google-Gemini"

    def __init__(self):
        self.logs = []

    def _log(self, msg: str):
        self.logs.append(msg)
        return msg

    def _reset_logs(self):
        self.logs = []

    def generate_image(self, prompt, api_key, model, aspect_ratio, temperature, seed=66666666, images=None):
        self._reset_logs()
        resp_text = ""

        # 1) API key
        resolved_key = _resolve_api_key(api_key, self.logs)
        if not resolved_key:
            err = (
                "错误: 未提供有效的API密钥。\n"
                "请在节点中输入API密钥，或将其保存到 gemini_api_key.txt，"
                "或设置变量 de entorno GEMINI_API_KEY。"
            )
            self._log(err)
            # Retorna una imagen vacía estándar (gris) para no romper grafos
            empty = np.ones((512, 512, 3), dtype=np.float32) * 0.2
            return (torch.from_numpy(empty).unsqueeze(0), "## 日志\n" + "\n".join(self.logs))

        client = genai.Client(api_key=resolved_key)

        # 2) Prompt + hints
        ar_hint = _hint_from_aspect_ratio(aspect_ratio)
        full_prompt = prompt if not ar_hint else f"{ar_hint} Create a detailed image of: {prompt}"

        # 3) Config
        gen_config = types.GenerateContentConfig(
            temperature=float(temperature),
            seed=int(seed) if isinstance(seed, (int, np.integer)) else 66666666,
            response_modalities=["Image", "Text"],
        )
        self._log(f"[INFO] 使用模型: {model}")
        self._log(f"[INFO] 温度: {temperature} | 种子: {gen_config.seed}")

        # 4) Contenido (texto + imágenes de referencia opcionales)
        contents = [{"text": full_prompt}]
        ref_count = 0
        if images is not None:
            try:
                batch = images.shape[0]
                self._log(f"[INFO] 检测到参考图像数量: {batch}")
                for i in range(batch):
                    # ComfyUI: [B, H, W, 3] float32 [0..1]
                    arr = images[i].cpu().numpy()
                    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                    pil = Image.fromarray(arr)
                    buf = BytesIO()
                    pil.save(buf, format="PNG")
                    img_bytes = buf.getvalue()
                    contents.append({"inline_data": {"mime_type": "image/png", "data": img_bytes}})
                    ref_count += 1
            except Exception as e:
                self._log(f"[WARN] 参考图像处理失败: {e}")

        if ref_count > 0:
            contents[0]["text"] += f" Use {ref_count} reference image(s) as guidance."

        # 5) Llamada a la API (no stream: respuestas más simples de manejar en nodos)
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_config,
            )
        except Exception as api_err:
            self._log(f"[ERROR] API 调用失败: {api_err}")
            empty = np.ones((512, 512, 3), dtype=np.float32) * 0.2
            return (torch.from_numpy(empty).unsqueeze(0), "## 日志\n" + "\n".join(self.logs))

        # 6) Parsear respuesta (texto + primera imagen disponible)
        out_tensor = None
        try:
            if not getattr(response, "candidates", None):
                self._log("[WARN] API响应中没有candidates")
            else:
                parts = getattr(response.candidates[0].content, "parts", []) or []
                for p in parts:
                    if getattr(p, "text", None):
                        resp_text += p.text
                    elif getattr(p, "inline_data", None):
                        try:
                            img_bytes = _bytes_from_inline_data(p.inline_data, self.logs)
                            pil_img = Image.open(BytesIO(img_bytes))
                            out_tensor = _tensor_from_pil(pil_img, self.logs)
                            # En caso de múltiples imágenes, nos quedamos con la primera válida
                            break
                        except Exception as img_err:
                            self._log(f"[WARN] 解析图像失败: {img_err}")

            if out_tensor is None:
                # No hubo imagen, devolvemos un lienzo vacío para no romper el flujo
                self._log("[INFO] API未返回图像，返回空图像")
                empty = np.ones((512, 512, 3), dtype=np.float32) * 0.2
                out_tensor = torch.from_numpy(empty).unsqueeze(0)

        except Exception as parse_err:
            self._log(f"[ERROR] 响应解析错误: {parse_err}")
            traceback.print_exc()
            empty = np.ones((512, 512, 3), dtype=np.float32) * 0.2
            out_tensor = torch.from_numpy(empty).unsqueeze(0)

        # 7) Construir texto de salida (logs + texto del modelo)
        full_text = "## 处理日志\n" + "\n".join(self.logs)
        if resp_text.strip():
            full_text += "\n\n## API返回\n" + resp_text

        return (out_tensor, full_text)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "Google-Gemini": GeminiImageGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Google-Gemini": "Gemini 2.0 image"
}
