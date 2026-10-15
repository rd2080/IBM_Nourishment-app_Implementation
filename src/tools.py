# Enabling Gemini Integration instead of Watson
from dotenv import load_dotenv
from google import genai
from google.genai import types
# Removal of existing Watson integration
'''from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import ModelInference '''

import json
import os
import base64
import requests
from langchain.tools import tool
from PIL import Image
from io import BytesIO
from typing import List
import logging

logging.basicConfig(level=logging.INFO)

logging.info("Extracting ingredients from image...")

''' Removed Watson Integration
credentials = Credentials(
                   url = "https://us-south.ml.cloud.ibm.com",
                   # api_key = "<YOUR_API_KEY>" # Normally you'd put an API key here, but we've got you covered here
                  )
client = APIClient(credentials)
project_id = "skills-network"
'''
## TODO: Define the tools for the NourishBot crew
class ExtractIngredientsTool:
    @tool("Extract ingredients")
    def extract_ingredient(image_input: str):
        """
        Extract ingredients from a food item image.

        Args:
            image_input: Local image file path or remote image URL.

        Returns:
            Ingredients identified in the image.
        """
        # Step 1: Read the image
        if image_input.startswith("http"):
            response = requests.get(image_input)
            response.raise_for_status()
            image_bytes = BytesIO(response.content)
        else:
            if not os.path.isfile(image_input):
                raise FileNotFoundError(
                    f"No file found at path: {image_input}"
                )

            with open(image_input, "rb") as file:
                image_bytes = BytesIO(file.read())

        # Step 2: Encode the image in Base64
        encoded_image = base64.b64encode(
            image_bytes.read()
        ).decode("utf-8")

        # Step 3: Initialize the vision model
        model = ModelInference(
            model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
            credentials=credentials,
            project_id=project_id,
            params={"max_tokens": 300},
        )

        # Step 4: Send the image and prompt to the model
        response = model.chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract ingredients from the food item image",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64," + encoded_image
                            },
                        },
                    ],
                }
            ]
        )

        # Step 5: Return the model's response
        return response["choices"][0]["message"]["content"]