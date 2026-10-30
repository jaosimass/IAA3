from pathlib import Path
import colorsys
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def extract_image_features(image_path: Path) -> dict:
    """Extrai características simples de cor de uma imagem RGB."""
    with Image.open(image_path) as image:
        rgb = image.convert("RGB").resize((128, 128))
        pixels = np.asarray(rgb, dtype=np.float32) / 255.0

    features = {}
    channel_names = ("red", "green", "blue")
    for index, channel_name in enumerate(channel_names):
        channel = pixels[:, :, index]
        features[f"{channel_name}_mean"] = float(channel.mean())
        features[f"{channel_name}_std"] = float(channel.std())

    # Conversão para HSV para resumir brilho e saturação.
    hsv = np.asarray(Image.fromarray((pixels * 255).astype(np.uint8)).convert("HSV"),
                     dtype=np.float32) / 255.0
    features["hue_mean"] = float(hsv[:, :, 0].mean())
    features["saturation_mean"] = float(hsv[:, :, 1].mean())
    features["value_mean"] = float(hsv[:, :, 2].mean())
    features["value_std"] = float(hsv[:, :, 2].std())

    # Heurística simples, não um detector definitivo de fogo.
    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    fire_like = (red > 0.60) & (green > 0.20) & (red > green * 1.15) & (green > blue * 0.90)
    features["fire_like_color_ratio"] = float(fire_like.mean())
    return features

def extract_dataset(dataset_dir: str | Path) -> pd.DataFrame:
    """
    Espera subpastas por classe, por exemplo:
    data/train/fire/imagem.jpg e data/train/nofire/imagem.jpg.
    O rótulo é guardado para avaliação; não entra no ajuste do K-Means.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Não encontrei a pasta '{dataset_dir}'. Confira se o dataset foi extraído nela."
        )

    rows = []
    image_paths = sorted(
        path for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(
            f"Nenhuma imagem encontrada em '{dataset_dir}'. "
            "Organize as imagens em subpastas como fire/ e nofire/."
        )

    for image_path in image_paths:
        # O nome da primeira pasta abaixo de dataset_dir representa a classe.
        relative = image_path.relative_to(dataset_dir)
        label = relative.parts[0] if len(relative.parts) > 1 else "unknown"
        try:
            row = extract_image_features(image_path)
            row["image_path"] = str(image_path)
            row["label"] = label.lower()
            rows.append(row)
        except (UnidentifiedImageError, OSError) as error:
            print(f"[AVISO] Imagem ignorada: {image_path} ({error})")

    if not rows:
        raise ValueError(f"Não foi possível ler nenhuma imagem válida em '{dataset_dir}'.")
    return pd.DataFrame(rows)
