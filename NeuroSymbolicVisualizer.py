import difflib
import os
import base64
from typing import TypedDict

import cairosvg

import draco as drc
from draco.schema import Schema
from draco.renderer import AltairRenderer
import altair as alt
from vega_datasets import data
from altair.vegalite.v5.api import FacetChart
from openai import OpenAI

class NeuroSymbolicVisualizer:
    def __init__(self, data_source, img_folder="images"):
        self.draco = drc.Draco()
        self.renderer = AltairRenderer()
        self.img_folder = img_folder
        os.makedirs(self.img_folder, exist_ok=True)

        # Load and preprocess data
        self.df = data_source()
        self.df.rename(columns=str.lower, inplace=True)

        # Create schema and base specification
        # Schema is a subtype of TypedDict
        self.schema: TypedDict = drc.schema_from_dataframe(self.df)
        self.data_schema_facts: list[str] = drc.dict_to_facts(self.schema)
        self.input_spec_base = self.data_schema_facts + [
            "entity(view,root,v0).",
            "entity(mark,v0,m0).",
        ]
        self.column_choice_llm = OpenAI()
        self.chart_eval_llm = OpenAI()

    def get_img_file_path(self, col1: str, col2: str) -> str:
        return os.path.join(self.img_folder, f"{col1}-{col2}.svg")

    def recommend_columns_llm(self) -> (str, str):
        """
        Recommends two columns for visualization based on the schema and potential correlation.

        :raises IndexError: If fewer than two columns are returned and no matches are available.
        :return: A tuple containing the names of the two recommended columns for visualization.
        """
        column_guesses = self.column_choice_llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "You're part of visualization recommendation system. You pick two features from a python "
                            "dict file to plot, whose correlation is insightful. Answer in two words seperated by a "
                            "space"},
                {
                    "role": "user",
                    "content": str(self.schema)
                }
            ]
        ).choices[0].message.content.split()
        # If the LLM provides an invalid output (which it hasn't done in practice) take the closest one
        columns = [difflib.get_close_matches(word, self.all_columns(), n=1, cutoff=0.0)[0]
                   for word in column_guesses[:2]]
        return columns[0], columns[1]

    def recommend_chart_asp(self, col1: str, col2: str) -> FacetChart:
        """
        Generates and saves recommended charts based on the input specification.

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
        chart.save(self.get_img_file_path(col1, col2))
        return chart

    def encode_img_base64(self, image_path: str) -> str:
        """
        :return: base64 encoded image strings of an svg chart
        """
        with open(image_path, "rb") as image_file:
            png_data = cairosvg.svg2png(url=image_path)
            return base64.b64encode(png_data).decode("utf-8")

    def eval_chart_llm(self, col1: str, col2: str) -> (int, str):
        """
        Uses an LLM capable to evaluate the chart visually (as a png). Additionally also passing the schema of the dataset,
        and asking for an explanation has the effect of making score returned more sensible

        Requires that *recommend_charts_asp* has been called beforehand.

        :return: score of the chart according to llm in the range [1,100]
        """
        base64_img = self.encode_img_base64(self.get_img_file_path(col1, col2))
        response = self.chart_eval_llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content":
                     "You're part of visualization recommendation system. Rank how good a visualization is. "
                     "Answer with a single number in the range 1-100, no other characters are allowed in the first line. "
                     "Provide a brief explanation in the next line"},
                {"role": "user",
                 "content": f"For context, here's the schema of the dataset for which the visualization was generated:\n{self.schema} \n"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
                        },
                    ],
                }
            ]
        ).choices[0].message.content
        (score, explanation) = response.partition("\n")[::2]
        try:
            return int(score), explanation.strip()
        except ValueError:
            print(f"LLM evaluation failed for: {self.get_img_file_path(col1, col2)} with completion: {response}")
            return 0, "error"

    def all_columns(self) -> list[str]:
        """
        string columns (such as origin) are removed. These are used for multi-faceted plots (many
        side by side plots per chart)

        :return: valid columns (features) in the dataset
        """
        return [feature['name'] for feature in self.schema['field'] if feature['type'] != 'string']
