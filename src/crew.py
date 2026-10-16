import os
import yaml
from pathlib import Path

from dotenv import load_dotenv
from crewai import Agent, Crew, LLM, Process, Task

from src.tools import (
    ExtractIngredientsTool,
    FilterIngredientsTool,
    DietaryFilterTool,
    NutrientAnalysisTool,
    RecipeSuggestionTool,
)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY not found. Add it to the project's root .env file."
    )

# Configure CrewAI to use Gemini through LiteLLM.
gemini_llm = LLM(
    model="gemini/gemini-3.8-flash",
    api_key=GEMINI_API_KEY,
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
            llm=gemini_llm,
            tools=[ExtractIngredientsTool.extract_ingredient],
            verbose=True,
            allow_delegation=False,
        )

        cleaning_agent = Agent(
            role="Ingredient Cleaning Specialist",
            goal="Clean and standardize the ingredients identified in the image.",
            backstory=(
                "You normalize ingredient names and remove irrelevant "
                "visual details without inventing ingredients."
            ),
            llm=gemini_llm,
            tools=[FilterIngredientsTool.filter_ingredients],
            verbose=True,
            allow_delegation=False,
        )

        dietary_agent = Agent(
            config=self.agents_config["dietary_filtering_agent"],
            llm=gemini_llm,
            tools=[DietaryFilterTool.filter_by_dietary_restrictions],
            verbose=True,
            allow_delegation=False,
        )

        nutrition_agent = Agent(
            config=self.agents_config["nutrient_analysis_agent"],
            llm=gemini_llm,
            tools=[NutrientAnalysisTool.analyze_nutrients],
            verbose=True,
            allow_delegation=False,
        )

        recipe_agent = Agent(
            config=self.agents_config["recipe_suggestion_agent"],
            llm=gemini_llm,
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
        """Create tasks and connect their outputs in sequence."""

        (
            ingredient_agent,
            cleaning_agent,
            dietary_agent,
            nutrition_agent,
            recipe_agent,
        ) = agents

        ingredient_task = Task(
            description="""
                Use the Extract ingredients tool to identify the ingredients
                visible in this image: {uploaded_image}.

                Pass the image path to the tool. Report the tool's result
                without inventing ingredients.
            """,
            expected_output="A list of ingredients identified in the image.",
            agent=ingredient_agent,
        )

        cleaning_task = Task(
            description="""
                Use the Filter ingredients tool to clean and standardize
                the output of the preceding ingredient identification task.

                Preserve uncertainty and do not add ingredients.
            """,
            expected_output="A cleaned and deduplicated ingredient list.",
            agent=cleaning_agent,
            context=[ingredient_task],
        )

        dietary_task = Task(
            description="""
                Apply these dietary restrictions: {dietary_restrictions}.

                Use the dietary filtering tool with the cleaned ingredient
                list from the preceding task. Exclude incompatible
                ingredients and clearly identify uncertainty.
            """,
            expected_output="Ingredients compatible with the restrictions.",
            agent=dietary_agent,
            context=[cleaning_task],
        )

        nutrition_task = Task(
            description="""
                Analyze the nutritional content of the ingredients remaining
                after dietary filtering.

                Use the nutritional analysis tool. Supply the filtered
                ingredients and these portion details: {portion_details}.

                Clearly distinguish estimates from verified values.
            """,
            expected_output=(
                "Estimated calories, macronutrients, relevant "
                "micronutrients, and assumptions."
            ),
            agent=nutrition_agent,
            context=[dietary_task],
        )

        recipe_task = Task(
            description="""
                Generate recipes using the ingredients that passed dietary
                filtering.

                Dietary restrictions: {dietary_restrictions}
                Calorie goal: {calorie_goal}
                Number of servings: {servings}

                Use the recipe suggestion tool. Incorporate the nutritional
                analysis from the preceding task. Provide ingredient
                quantities, preparation instructions, and estimated
                nutritional values per serving.
            """,
            expected_output=(
                "Recipe suggestions with ingredients, instructions, "
                "servings, estimated nutrition, and a recommendation."
            ),
            agent=recipe_agent,
            context=[dietary_task, nutrition_task],
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