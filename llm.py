from transformers import pipeline

question_answerer = pipeline(task="question-answering", model="deepset/roberta-base-squad2")

preds = question_answerer(
    question="What is the name of the repository?",
    context="The name of the repository is huggingface/transformers",
)

print(
    f"score: {round(preds['score'], 4)}, start: {preds['start']}, end: {preds['end']}, answer: {preds['answer']}"
)
