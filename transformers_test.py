from transformers import pipeline

classifier = pipeline('sentiment-analysis')
print(classifier('We are very love chinese'))
