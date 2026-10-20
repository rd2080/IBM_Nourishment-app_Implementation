import gradio_client.utils as gradio_client_utils

_original_get_type = gradio_client_utils.get_type

def _patched_get_type(schema):
    if isinstance(schema, bool):
        return "boolean"
    return _original_get_type(schema)

gradio_client_utils.get_type = _patched_get_type

import gradio as gr
import time
from src.crew import NourishBotRecipeCrew, NourishBotAnalysisCrew

def analyze_food(image_path, dietary_restrictions, portion_details):
    """Identify ingredients, apply dietary filters, and analyze nutrition."""

    if not image_path:
        return "Please upload an image of your food or ingredients."

    dietary_restrictions = (
        dietary_restrictions.strip()
        if dietary_restrictions and dietary_restrictions.strip()
        else "None"
    )

    portion_details = (
        portion_details.strip()
        if portion_details and portion_details.strip()
        else "Not provided"
    )

    try:
        crew = NourishBotAnalysisCrew(
            image_data=image_path,
            dietary_restrictions=dietary_restrictions,
            portion_details=portion_details,
        )

        result = crew.run()
        return str(result)

    except Exception as exc:
        return (
            "### Analysis failed\n\n"
            f"**Error:** `{type(exc).__name__}: {exc}`\n\n"
            "Check your Watsonx credentials, model access, and terminal logs."
        )


def suggest_recipes(
    image_path,
    dietary_restrictions,
    calorie_goal,
    portion_details,
    servings,
):
    """Analyze the uploaded food and suggest suitable recipes."""

    if not image_path:
        return "Please upload an image of your food or ingredients."

    dietary_restrictions = (
        dietary_restrictions.strip()
        if dietary_restrictions and dietary_restrictions.strip()
        else "None"
    )

    calorie_goal = (
        calorie_goal.strip()
        if calorie_goal and calorie_goal.strip()
        else "Not specified"
    )

    portion_details = (
        portion_details.strip()
        if portion_details and portion_details.strip()
        else "Not provided"
    )

    servings = str(servings)

    try:
        crew = NourishBotRecipeCrew(
            image_data=image_path,
            dietary_restrictions=dietary_restrictions,
            calorie_goal=calorie_goal,
            portion_details=portion_details,
            servings=servings,
        )

        result = crew.run()
        return str(result)

    except Exception as exc:
        return (
            "### Recipe generation failed\n\n"
            f"**Error:** `{type(exc).__name__}: {exc}`\n\n"
            "Check your Watsonx credentials, model access, and terminal logs."
        )


with gr.Blocks(title="NourishBot — AI Food Assistant") as demo:
    gr.Markdown(
        """
        # NourishBot
        ### AI-powered ingredient detection, nutrition analysis, and recipes

        Upload a food or ingredient image. NourishBot will identify
        ingredients, apply your dietary restrictions, and estimate
        nutritional information using IBM Watsonx.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            food_image = gr.Image(
                type="filepath",
                label="Upload food or ingredient image",
            )

            dietary_restrictions = gr.Textbox(
                label="Dietary restrictions",
                placeholder="e.g. vegetarian, vegan, gluten-free",
                value="None",
            )

            portion_details = gr.Textbox(
                label="Portion details",
                placeholder="e.g. 200 g rice, 2 tomatoes, 1 tbsp oil",
                lines=2,
            )

            with gr.Accordion("Recipe preferences", open=True):
                calorie_goal = gr.Textbox(
                    label="Calorie goal",
                    placeholder="e.g. 500 kcal per serving",
                    value="Not specified",
                )

                servings = gr.Number(
                    label="Number of servings",
                    value=1,
                    minimum=1,
                    maximum=50,
                    precision=0,
                )

            with gr.Row():
                analyze_button = gr.Button(
                    "Analyze Nutrition",
                    variant="secondary",
                )

                recipe_button = gr.Button(
                    "Suggest Recipes",
                    variant="primary",
                )

        with gr.Column(scale=1):
            analysis_output = gr.Markdown(
                label="Nutrition Analysis",
            )

            recipe_output = gr.Markdown(
                label="Recipe Suggestions",
            )

    analyze_button.click(
        fn=analyze_food,
        inputs=[
            food_image,
            dietary_restrictions,
            portion_details,
        ],
        outputs=analysis_output,
    )

    recipe_button.click(
        fn=suggest_recipes,
        inputs=[
            food_image,
            dietary_restrictions,
            calorie_goal,
            portion_details,
            servings,
        ],
        outputs=recipe_output,
    )


if __name__ == "__main__":
    demo.queue().launch()
