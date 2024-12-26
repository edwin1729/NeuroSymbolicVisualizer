import os

import draco as drc
from draco.schema import Schema
from draco.renderer import AltairRenderer
import altair as alt
from vega_datasets import data
from altair.vegalite.v5.api import FacetChart
from openai import OpenAI
from itertools import combinations

class NeuroSymbolicVisualizer:
    def __init__(self, data_source=data.seattle_weather, img_folder="images"):
        self.draco = drc.Draco()
        self.renderer = AltairRenderer()
        self.img_folder = img_folder
        os.makedirs(self.img_folder, exist_ok=True)

        # Load and preprocess data
        self.df = data_source()
        self.df.rename(columns=str.lower, inplace=True)

        # Create schema and base specification
        self.schema = drc.schema_from_dataframe(self.df)
        self.data_schema_facts = drc.dict_to_facts(self.schema)
        self.input_spec_base = self.data_schema_facts + [
            "entity(view,root,v0).",
            "entity(mark,v0,m0).",
        ]
        self.llm = OpenAI()

    def recommend_charts_asp(self, col1: str, col2: str) -> int:
        """
        Generates and saves recommended charts based on the input specification.

        :param spec: The Draco specification as a list of strings.
        :param num: Number of charts to recommend.
        :return: cost of the model according to draco
        """


        spec_asp = self.input_spec_base + [
            # Encode the `temp_max` field
            "entity(encoding,m0,e0).",
            f"attribute((encoding,field),e0,{col1}).",
            # Encode the `wind` field
            "entity(encoding,m0,e1).",
            f"attribute((encoding,field),e1,{col2}).",
            # Create a faceted chart
            "entity(facet,v0,f0).",
            "attribute((facet,channel),f0,col).",
        ]
        model = next(self.draco.complete_spec(spec_asp))
        spec_answer = drc.answer_set_to_dict(model.answer_set)

        # Render and save the chart
        chart: FacetChart = self.renderer.render(spec=spec_answer, data=self.df)
        chart = chart.configure_view(continuousWidth=130, continuousHeight=130)
        chart.save(os.path.join(self.img_folder, f"{col1}-{col2}.svg"))
        return model.cost

    def recommend_columns_llm(self) -> (str, str):
        columns = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "You're part of visualization recommendation system. You pick two features from a python dict file to plot, whose correlation is insightful. Answer in two words seperated by a space"},
                {
                    "role": "user",
                    "content": str(self.schema)
                }
            ]
        ).choices[0].message.content.split()
        return columns[0], columns[1]

    def all_columns(self) -> list[str]:
        """
        string columns (such as origin) are removed. These are used for multi-faceted plots
        (many side by side plots per origin)
        :return: valid columns (features) in the dataset
        """
        return [feature['name'] for feature in self.schema['field'] if feature['type'] != 'string']

    def run_default_recommendation(self):
        """
        Runs a default chart recommendation based on a predefined input specification.
        """
        (col1, col2) = self.recommend_columns_llm()
        llm_cost = self.recommend_charts_asp(col1, col2)

        all_chart_costs = [self.recommend_charts_asp(*cols) for cols in combinations(self.all_columns(), 2)]
        print(f"LLM cost: {llm_cost}")
        print(f"All chart costs: {all_chart_costs}")

        print(f"llm choice: {(col1, col2)}")
        all_combinations = combinations(self.all_columns(), 2)
        print(f"all combinations: {list(all_combinations)}")

# Example usage
if __name__ == "__main__":
    recommender = NeuroSymbolicVisualizer(data.cars)
    recommender.run_default_recommendation()
