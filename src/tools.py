import os
import logging
import mimetypes
import base64
from io import BytesIO
from pathlib import Path

import requests
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference
from langchain.tools import tool
from PIL import Image


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


WATSONX_URL = os.getenv("WATSONX_URL")
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

if not all([WATSONX_URL, WATSONX_APIKEY, WATSONX_PROJECT_ID]):
    raise EnvironmentError(
        "Watsonx credentials are missing. Add WATSONX_URL, "
        "WATSONX_APIKEY, and WATSONX_PROJECT_ID to the root .env file."
    )


TEXT_MODEL_ID = "meta-llama/llama-3-3-70b-instruct"
VISION_MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"


def _create_model(model_id: str, max_new_tokens: int = 1200):
    """Create a Watsonx model inference client."""

    return ModelInference(
        model_id=model_id,
        credentials={
            "url": WATSONX_URL,
            "apikey": WATSONX_APIKEY,
        },
        project_id=WATSONX_PROJECT_ID,
        params={
            "max_new_tokens": max_new_tokens,
            "temperature": 0.2,
        },
    )


def _generate_text(prompt: str) -> str:
    """Generate a text response using Watsonx."""

    model = _create_model(TEXT_MODEL_ID)

    response = model.generate_text(prompt=prompt)

    if not response or not str(response).strip():
        raise RuntimeError("Watsonx returned an empty text response.")

    return str(response).strip()


def _generate_vision(prompt: str, image_bytes: bytes) -> str:
    """Generate a response from an image and text prompt using Watsonx."""

    image = Image.open(BytesIO(image_bytes))
    image.load()

    image_buffer = BytesIO()
    image.convert("RGB").save(image_buffer, format="JPEG")
    encoded_image = base64.b64encode(image_buffer.getvalue()).decode("utf-8")

    image_data_url = f"data:image/jpeg;base64,{encoded_image}"

    model = _create_model(VISION_MODEL_ID, max_new_tokens=500)

    response = model.chat(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                        },
                    },
                ],
            }
        ]
    )

    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "Watsonx returned an unexpected image-analysis response."
        ) from exc

    if isinstance(content, list):
        content = "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        )

    if not content or not str(content).strip():
        raise RuntimeError("Watsonx returned an empty image-analysis response.")

    return str(content).strip()


def _load_image(image_input: str) -> bytes:
    """Load an image from a local path or public URL and validate it."""

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

    return image_bytes


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

        logger.info("Extracting ingredients using Watsonx")

        image_bytes = _load_image(image_input)

        prompt = (
            "Identify the visible food ingredients in this image. "
            "Return a concise, readable list of ingredients. "
            "Do not invent ingredients that cannot reasonably be "
            "identified from the image. If uncertain, say so."
        )

        return _generate_vision(prompt, image_bytes)


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

        prompt = f"""
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

        return _generate_text(prompt)


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

        prompt = f"""
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

        return _generate_text(prompt)


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

        logger.info("Analyzing nutritional content using Watsonx")

        prompt = f"""
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

        return _generate_text(prompt)


class RecipeSuggestionTool:

    @tool("Suggest recipes")
    def suggest_recipes(
        ingredients: str,
        dietary_restrictions: str = "None",
        calorie_goal: str = "Not specified",
        nutrient_analysis: str = "Not available",
        servings: str = "1",
    ) -> str:
        """
        Suggest recipes using available ingredients and nutritional goals.

        Args:
            ingredients: Ingredients available and compatible with
                the user's dietary restrictions.
            dietary_restrictions: User's dietary requirements.
            calorie_goal: Target calories per serving or meal.
            nutrient_analysis: Available nutritional analysis.
            servings: Number of servings requested.

        Returns:
            Recipe suggestions with instructions and estimated nutrition.
        """

        logger.info("Generating recipe suggestions using Watsonx")

        prompt = f"""
        You are a recipe recommendation assistant.

        Suggest practical recipes using the information below.

        AVAILABLE INGREDIENTS:
        {ingredients}

        DIETARY RESTRICTIONS:
        {dietary_restrictions}

        CALORIE GOAL:
        {calorie_goal}

        AVAILABLE NUTRITIONAL ANALYSIS:
        {nutrient_analysis}

        NUMBER OF SERVINGS:
        {servings}

        REQUIREMENTS:

        1. RECIPE SELECTION
           - Suggest 2 or 3 distinct recipes.
           - Prioritize the available ingredients.
           - Do not use ingredients that clearly violate the
             user's dietary restrictions.
           - Do not treat uncertain ingredients as confirmed.
           - Clearly identify any additional ingredients needed.
           - Separate optional pantry ingredients from essential
             ingredients that are not available.

        2. CALORIE GOAL
           - Aim to meet the calorie goal when one is provided.
           - Clarify whether the target refers to one serving
             or the entire meal if that information is available.
           - Do not claim a recipe meets the goal when its
             estimated calories exceed the target.
           - If the goal is unspecified, do not invent one.

        3. FOR EACH RECIPE, INCLUDE
           - Recipe name.
           - Brief description.
           - Ingredients and quantities.
           - Preparation steps in the correct order.
           - Estimated preparation and cooking time.
           - Number of servings.
           - Estimated calories per serving.
           - Estimated protein, carbohydrates, and fat per serving.
           - Any relevant nutritional benefits.

        4. NUTRITIONAL ACCURACY
           - Nutritional values must be estimates, not verified facts.
           - Use reasonable ingredient quantities for the stated
             number of servings.
           - Account for cooking oil and additional ingredients
             when estimating nutrition.
           - If the available information is insufficient,
             state the assumptions.
           - Do not invent precise nutritional values.

        5. DIETARY SAFETY
           - Respect the stated dietary restrictions.
           - Do not assume an ingredient is allergen-free when
             its composition is uncertain.
           - Flag uncertainty where ingredient labels or
             preparation methods would need to be checked.

        6. FINAL RECOMMENDATION
           - Identify which suggested recipe best fits the
             available ingredients and stated calorie goal.
           - Briefly explain why.

        Return a clear, structured response with headings
        for each recipe.
        """

        return _generate_text(prompt)