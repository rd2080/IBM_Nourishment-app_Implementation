import unittest

from src.models import (
    MineralInfo,
    NutrientAnalysisOutput,
    NutrientBreakdown,
    Recipe,
    RecipeSuggestionOutput,
    VitaminInfo,
)


class NourishBotModelTests(unittest.TestCase):
    def test_recipe_output_accepts_recipe_list(self):
        output = RecipeSuggestionOutput(
            recipes=[
                Recipe(
                    title="Chickpea Salad",
                    ingredients=["chickpeas", "tomato"],
                    instructions="Mix the ingredients.",
                    calorie_estimate=320,
                )
            ]
        )
        self.assertEqual(output.recipes[0].title, "Chickpea Salad")
        self.assertEqual(output.recipes[0].calorie_estimate, 320)

    def test_nutrient_output_accepts_nested_details(self):
        output = NutrientAnalysisOutput(
            dish="Vegetable rice",
            portion_size="1 bowl",
            estimated_calories=410,
            nutrients=NutrientBreakdown(
                protein="9 g",
                carbohydrates="65 g",
                fats="12 g",
                vitamins=[VitaminInfo(name="Vitamin C", percentage_dv="18%")],
                minerals=[MineralInfo(name="Iron", amount="2 mg")],
            ),
            health_evaluation="Estimated values; portion size affects accuracy.",
        )
        self.assertEqual(output.estimated_calories, 410)
        self.assertEqual(output.nutrients.vitamins[0].name, "Vitamin C")
        self.assertEqual(output.nutrients.minerals[0].amount, "2 mg")

    def test_nutrient_output_supports_missing_optional_details(self):
        output = NutrientAnalysisOutput()
        self.assertIsNone(output.dish)
        self.assertEqual(output.nutrients.vitamins, [])
        self.assertEqual(output.nutrients.minerals, [])


if __name__ == "__main__":
    unittest.main()
