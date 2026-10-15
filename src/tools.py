import os
import logging
import mimetypes
from io import BytesIO
from pathlib import Path

import requests
from dotenv import load_dotenv
from google import genai
from langchain.tools import tool
from PIL import Image

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY not found. Add it to the project's root .env file."
    )

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-3.8-flash"

class ExtractIngredientsTool:

    @tool("Extract ingredients")
    def extract_ingredient(image_input: str) -> str:
        """
        Identify the visible food ingredients in an image.

        Args:
            image_input: Local image path or publicly accessible image URL.

        Returns:
            A text description of the identified ingredients.
        """
        logger.info("Extracting ingredients using Gemini")

        # Read the image from a URL or local file.
        if image_input.startswith(("http://", "https://")):
            response = requests.get(image_input, timeout=30)
            response.raise_for_status()
            image_bytes = response.content
        else:
            image_path = Path(image_input).expanduser()

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"No image file found at: {image_input}"
                )

            image_bytes = image_path.read_bytes()

        # Validate the image and determine its MIME type.
        try:
            image = Image.open(BytesIO(image_bytes))
            image.load()
        except Exception as exc:
            raise ValueError(
                "The supplied input is not a valid supported image."
            ) from exc

        mime_type = Image.MIME.get(image.format)

        if not mime_type:
            mime_type = mimetypes.guess_type(image_input)[0]

        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError("Could not determine the image format.")

        # Send the image and prompt to Gemini.
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                (
                    "Identify the visible food ingredients in this image. "
                    "Return a concise, readable list of ingredients. "
                    "Do not invent ingredients that cannot reasonably be "
                    "identified from the image. If uncertain, say so."
                ),
                genai.types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                ),
            ],
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        return response.text.strip()