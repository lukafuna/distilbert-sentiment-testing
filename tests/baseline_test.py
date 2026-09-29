import pytest
from transformers import pipeline


@pytest.fixture(scope="module")
def classifier():
    return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def test_clearly_positive(classifier):
    result = classifier("I absolutely loved this movie, it was fantastic!")[0]
    assert result["label"] == "POSITIVE"


def test_clearly_negative(classifier):
    result = classifier("This was the worst film I have ever seen.")[0]
    assert result["label"] == "NEGATIVE"

def test_empty_string(classifier):
    result = classifier("")[0]
    assert result["label"] in ["POSITIVE", "NEGATIVE"]


def test_single_word_positive(classifier):
    result = classifier("Great")[0]
    assert result["label"] == "POSITIVE"


def test_single_word_negative(classifier):
    result = classifier("Terrible")[0]
    assert result["label"] == "NEGATIVE"


def test_all_caps(classifier):
    result = classifier("THIS MOVIE WAS AMAZING")[0]
    assert result["label"] == "POSITIVE"


def test_numbers_only(classifier):
    result = classifier("12345")[0]
    assert result["label"] in ["POSITIVE", "NEGATIVE"]


def test_long_text(classifier):
    long_text = "This movie was great. " * 50
    result = classifier(long_text)[0]
    assert result["label"] == "POSITIVE"