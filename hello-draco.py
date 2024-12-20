import pandas as pd
import os

import draco as drc
from draco.schema import Schema
from draco.renderer import AltairRenderer
import altair as alt
from vega_datasets import data
from altair.vegalite.v5.api import FacetChart

# Setting up renderer and demo data
d = drc.Draco()
renderer = AltairRenderer()
img_folder = "images"
os.makedirs(img_folder, exist_ok=True)

df: pd.DataFrame = data.seattle_weather() # londonTubeLines does not work because the arrays of it are not same length. But I want to do these...
df.rename(columns=str.lower, inplace=True)

# Schema is a subtype of dict
schema: Schema = drc.schema_from_dataframe(df)
data_schema_facts: list[str] = drc.dict_to_facts(schema)
input_spec_base = data_schema_facts + [
    "entity(view,root,v0).",
    "entity(mark,v0,m0).",
]

def recommend_charts(
    spec: list[str], draco: drc.Draco, num: int = 5, labeler=lambda i: f"CHART {i+1}"
) -> dict[str, tuple[list[str], dict]]:
    # Dictionary to store the generated recommendations, keyed by chart name
    chart_specs = {}
    for i, model in enumerate(draco.complete_spec(spec, num)):
        chart_name = labeler(i)
        spec = drc.answer_set_to_dict(model.answer_set)
        chart_specs[chart_name] = drc.dict_to_facts(spec), spec

        print(f"{chart_name} COST: {model.cost}")
        chart: FacetChart = renderer.render(spec=spec, data=df)
        chart = chart.configure_view(continuousWidth=130, continuousHeight=130)
        chart.save(os.path.join(img_folder, f"{chart_name}.svg"))

    return chart_specs

input_spec = input_spec_base + [
    # We want to encode the `date` field
    "entity(encoding,m0,e0).",
    "attribute((encoding,field),e0, temp_max).",
    # We want to encode the `temp_max` field
    "entity(encoding,m0,e1).",
    "attribute((encoding,field),e1,wind).",
    # We want the chart to be a faceted chart
    "entity(facet,v0,f0).",
    "attribute((facet,channel),f0,col).",
]
recommendations = recommend_charts(spec=input_spec, draco=d, num=5)

