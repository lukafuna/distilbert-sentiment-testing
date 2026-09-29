from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

result = classifier("I loved this movie!")
print(result)

result2 = classifier("This was a terrible experience.")
print(result2)