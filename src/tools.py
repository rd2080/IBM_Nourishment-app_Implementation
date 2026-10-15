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

class FilterIngredientsTool:

    @tool("Filter ingredients")
    def filter_ingredients(ingredients: str) -> str:
        """
        Clean and normalize a list of ingredients extracted from an image.

        Args:
            ingredients: Raw ingredient extraction output.

        Returns:
            A clean, deduplicated list of ingredients.
        """
        logger.info("Cleaning extracted ingredients")

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                f"""
                Clean the following food ingredient list.

                Requirements:
                1. Extract only the identified food ingredients.
                2. Remove duplicate ingredients.
                3. Remove descriptions of shelves, containers, packaging,
                   and other irrelevant visual details.
                4. Use simple, standardized ingredient names.
                5. Do not invent ingredients or assume uncertain items
                   are present.
                6. If an item is uncertain, label it as uncertain.

                Return only the cleaned ingredient list.

                Raw ingredients:
                {ingredients}
                """
            ],
        )

        if not response.text:
            raise RuntimeError(
                "Gemini returned an empty cleaned ingredient list."
            )

        return response.text.strip()


class DietaryFilterTool:

    @tool("Filter ingredients by dietary restrictions")
    def filter_by_dietary_restrictions(
        ingredients: str,
        dietary_restrictions: str,
    ) -> str:
        """
        Filter ingredients according to the user's dietary restrictions.

        Args:
            ingredients: Cleaned list of ingredients.
            dietary_restrictions: User's dietary requirements.

        Returns:
            Ingredients compatible with the specified restrictions.
        """
        logger.info("Filtering ingredients by dietary restrictions")

        if not dietary_restrictions or not dietary_restrictions.strip():
            return ingredients

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                f"""
                You are a dietary ingredient filtering assistant.

                Filter the following ingredients according to the
                user's dietary restrictions.

                Dietary restrictions:
                {dietary_restrictions}

                Ingredients:
                {ingredients}

                Rules:
                1. Return only ingredients that are compatible with
                   the specified dietary restrictions.
                2. Exclude ingredients that clearly violate them.
                3. Do not invent ingredients or replace excluded items.
                4. If compatibility cannot be established from the
                   available information, mark the item as uncertain
                   rather than claiming it is safe.
                5. Do not assume that an ingredient is vegan,
                   gluten-free, or allergen-free when its composition
                   is unclear.
                6. If no ingredients remain, state that clearly.

                Return the filtered ingredient list and briefly explain
                any exclusions or uncertainties.
                """
            ],
        )

        if not response.text:
            raise RuntimeError(
                "Gemini returned an empty dietary filtering result."
            )

        return response.text.strip()

class NutrientAnalysisTool:

    @tool("Analyze nutritional content")
    def analyze_nutrients(
        ingredients: str,
        portion_details: str = "Not provided",
    ) -> str:
        """
        Estimate calories and nutrients from a list of food ingredients.

        Args:
            ingredients: Ingredients present in the meal.
            portion_details: Available information about quantities,
                serving sizes, or number of servings.

        Returns:
            A nutritional report with estimates and limitations.
        """
        logger.info("Analyzing nutritional content using Gemini")

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                f"""
                You are a nutrition analysis assistant.

                Estimate the nutritional content of the following meal.

                INGREDIENTS:
                {ingredients}

                PORTION INFORMATION:
                {portion_details}

                Analyze the meal and provide the following sections:

                1. CALORIE ESTIMATE
                   - Estimated total calories (kcal).
                   - Estimated calories per serving, if the number of
                     servings is known.

                2. MACRONUTRIENTS
                   - Protein in grams.
                   - Carbohydrates in grams.
                   - Total fat in grams.
                   - Fiber in grams, if estimable.

                3. MICRONUTRIENTS
                   - Relevant vitamins and minerals likely present.
                   - Explain that their quantities are not established
                     unless sufficient information is available.

                4. NUTRITIONAL ASSESSMENT
                   - Briefly assess the meal's nutritional composition.
                   - Mention notable sources of protein, carbohydrates,
                     fats, fiber, or other relevant nutrients.

                5. ASSUMPTIONS AND UNCERTAINTY
                   - State any assumptions about portion sizes.
                   - Explain how missing quantities, preparation methods,
                     sauces, and cooking oils affect the estimates.
                   - If quantities are unknown, do not present estimates
                     as precise measurements.
                   - Do not invent ingredient quantities.
                   - Clearly distinguish estimates from verified values.

                Important rules:
                - Use kcal for energy and grams for macronutrients.
                - Do not claim laboratory-level accuracy.
                - Do not infer exact nutrient quantities from ingredient
                  names alone.
                - If the available information is insufficient for a
                  meaningful numerical estimate, explain what information
                  is needed.

                Return a readable report with headings.
                """
            ],
        )

        if not response.text:
            raise RuntimeError(
                "Gemini returned an empty nutritional analysis."
            )

        return response.text.strip()