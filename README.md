# NeuroSymbolicVisualizaer 
*An experiment in visualization using ASP ([Draco](https://github.com/cmudig/draco2/)) and LLM*

Draco is a visualization recommendation system which uses ASP (answer set programs) to declaritively denote a chart/plot
as logical facts. Draco has built into it domain specific knowledge on what makes a good chart in the form of ASP facts.
But there are some decisions that are subjective and better made by an LLM. In this experiment we:
1) Pick features (`column` in the codebase) to plot using an LLM and use draco to plot it
2) Evaluate a plot by converting it to a png and asking the LLM to give it a score

A decision I've made in this experiment is to use Faceted charts (multiple plots side by side, of a different subcategory).
I do this just so that we can show ASP at work as Draco can pick a feature to categorize over which will minimize the cost.
We can easily make the charts simple and unfaceted by commenting two lines in the `recommend_chart_asp(...)` method, specifically
the `spec_asp` variable

We also try to compare generated plots using Draco's inbuilt cost model. A majority of charts received the same cost 
when using this technique, so we tried to evaluate a recommendation using a visual input to an LLM. However, this
(without further prompt engineering) suffers from the same problem. We have decided to present the LLM based evaluation here.
The Draco cost model based evaluation is also available in the `draco-cost-chart-eval` branch.

## Demo
You can run the `NeuroSymbolicVisualizerDemo.ipynb` to see this in action. The demo requires an openai token to be set 
as an environment variable. You may add the following to your `.xxxshrc` file:
```shell
export OPENAI_API_KEY=<your key here>
```
Also remember to use a python venv satisfying `requirements.txt`