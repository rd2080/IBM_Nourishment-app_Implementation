import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from crewai import Agent, Crew, LLM, Process, Task

from src.models import NutrientAnalysisOutput, RecipeSuggestionOutput
from src.tools import (
    DietaryFilterTool,
    ExtractIngredientsTool,
    FilterIngredientsTool,
    NutrientAnalysisTool,
    RecipeSuggestionTool,
)

load_dotenv()

WATSONX_URL = os.getenv("WATSONX_URL")
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

if not all([WATSONX_URL, WATSONX_APIKEY, WATSONX_PROJECT_ID]):
    raise EnvironmentError(
        "Watsonx credentials are missing. Check WATSONX_URL, "
        "WATSONX_APIKEY, and WATSONX_PROJECT_ID in the root .env file."
    )

watsonx_llm = LLM(
    model="watsonx/meta-llama/llama-3-3-70b-instruct",
    project_id=WATSONX_PROJECT_ID,
)

CONFIG_DIR = Path(__file__).parent / "config"


class BaseNourishBotCrew:
    """Coordinate the complete NourishBot workflow."""

    def __init__(
        self,
        image_data: str,
        dietary_restrictions: str = "None",
        calorie_goal: str = "Not specified",
        portion_details: str = "Not provided",
        servings: str = "1",
    ):
        self.image_data = image_data
        self.dietary_restrictions = dietary_restrictions
        self.calorie_goal = calorie_goal
        self.portion_details = portion_details
        self.servings = servings

        with open(CONFIG_DIR / "agents.yaml", encoding="utf-8") as file:
            self.agents_config = yaml.safe_load(file)

        with open(CONFIG_DIR / "tasks.yaml", encoding="utf-8") as file:
            self.tasks_config = yaml.safe_load(file)

    def _create_agents(self):
        """Create specialized agents and assign their tools."""
        ingredient_agent = Agent(
            config=self.agents_config["ingredient_detection_agent"],
            llm=watsonx_llm,
            tools=[ExtractIngredientsTool.extract_ingredient],
            verbose=True,
            allow_delegation=False,
        )

        cleaning_agent = Agent(
            role="Ingredient Cleaning Specialist",
            goal="Clean and standardize the ingredients identified in the image.",
            backstory=(
                "Normalize ingredient names and remove irrelevant visual "
                "details without inventing ingredients."
            ),
            llm=watsonx_llm,
            tools=[FilterIngredientsTool.filter_ingredients],
            verbose=True,
            allow_delegation=False,
        )

        dietary_agent = Agent(
            config=self.agents_config["dietary_filtering_agent"],
            llm=watsonx_llm,
            tools=[DietaryFilterTool.filter_by_dietary_restrictions],
            verbose=True,
            allow_delegation=False,
        )

        nutrition_agent = Agent(
            config=self.agents_config["nutrient_analysis_agent"],
            llm=watsonx_llm,
            tools=[NutrientAnalysisTool.analyze_nutrients],
            verbose=True,
            allow_delegation=False,
        )

        recipe_agent = Agent(
            config=self.agents_config["recipe_suggestion_agent"],
            llm=watsonx_llm,
            tools=[RecipeSuggestionTool.suggest_recipes],
            verbose=True,
            allow_delegation=False,
        )

        return (
            ingredient_agent,
            cleaning_agent,
            dietary_agent,
            nutrition_agent,
            recipe_agent,
        )

    def _create_tasks(self, agents):
        """Create sequential tasks with lab-defined structured outputs."""
        (
            ingredient_agent,
            cleaning_agent,
            dietary_agent,
            nutrition_agent,
            recipe_agent,
        ) = agents

        ingredient_config = self.tasks_config["ingredient_detection_task"]
        cleaning_config = self.tasks_config["ingredient_cleaning_task"]
        dietary_config = self.tasks_config["dietary_filtering_task"]
        nutrition_config = self.tasks_config["nutrient_analysis_task"]
        recipe_config = self.tasks_config["recipe_suggestion_task"]

        ingredient_task = Task(
            description=(
                ingredient_config["description"]
                + "\nUploaded image path: {uploaded_image}"
            ),
            expected_output=ingredient_config["expected_output"],
            agent=ingredient_agent,
        )

        cleaning_task = Task(
            description=cleaning_config["description"],
            expected_output=cleaning_config["expected_output"],
            agent=cleaning_agent,
            context=[ingredient_task],
        )

        dietary_task = Task(
            description=(
                dietary_config["description"]
                + "\nDietary restrictions: {dietary_restrictions}. "
                "Use the cleaned ingredients from the preceding task."
            ),
            expected_output=dietary_config["expected_output"],
            agent=dietary_agent,
            context=[cleaning_task],
        )

        nutrition_task = Task(
            description=(
                nutrition_config["description"]
                + "\nImage path: {uploaded_image}"
                + "\nPortion details: {portion_details}"
                + "\nAnalyze the filtered ingredients supplied by the "
                "preceding task. If the image is a complete dish, identify "
                "the dish and estimate its nutrition. Clearly label estimates."
            ),
            expected_output=nutrition_config["expected_output"],
            agent=nutrition_agent,
            context=[dietary_task],
            output_json=NutrientAnalysisOutput,
        )

        recipe_task = Task(
            description=(
                recipe_config["description"]
                + "\nDietary restrictions: {dietary_restrictions}"
                + "\nCalorie goal: {calorie_goal}"
                + "\nServings: {servings}"
                + "\nUse only ingredients that passed dietary filtering. "
                "Include quantities, instructions, and estimated calories "
                "per serving. Do not invent certainty about nutrition."
            ),
            expected_output=recipe_config["expected_output"],
            agent=recipe_agent,
            context=[dietary_task, nutrition_task],
            output_json=RecipeSuggestionOutput,
        )

        return [
            ingredient_task,
            cleaning_task,
            dietary_task,
            nutrition_task,
            recipe_task,
        ]

    def crew(self):
        """Build the complete sequential multi-agent crew."""
        agents = self._create_agents()
        tasks = self._create_tasks(agents)

        return Crew(
            agents=list(agents),
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )

    def run(self):
        """Execute the complete workflow."""
        return self.crew().kickoff(
            inputs={
                "uploaded_image": str(self.image_data),
                "dietary_restrictions": self.dietary_restrictions,
                "calorie_goal": self.calorie_goal,
                "portion_details": self.portion_details,
                "servings": self.servings,
            }
        )


class NourishBotAnalysisCrew(BaseNourishBotCrew):
    """Run ingredient identification, cleaning, filtering, and nutrition."""

    def crew(self):
        agents = self._create_agents()
        tasks = self._create_tasks(agents)

        return Crew(
            agents=list(agents[:4]),
            tasks=tasks[:4],
            process=Process.sequential,
            verbose=True,
        )


class NourishBotRecipeCrew(BaseNourishBotCrew):
    """Run the complete workflow, including recipe suggestions."""

    pass
