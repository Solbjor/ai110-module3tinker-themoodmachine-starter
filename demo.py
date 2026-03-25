#!/usr/bin/env python3
"""Quick demo of the improved Mood Machine."""

from mood_analyzer import MoodAnalyzer

analyzer = MoodAnalyzer()

test_sentences = [
    "I absolutely love this!",
    "This is wicked cool",
    "I'm exhausted but proud",
    "I hate how much I love this",
    "Not the worst outcome",
    "I love getting stuck in traffic",
]

print("MOOD MACHINE - Improved Analyzer Demo\n")
print("=" * 70)

for text in test_sentences:
    score = analyzer.score_text(text)
    label = analyzer.predict_label(text)
    print(f"Text: {text}")
    print(f"Score: {score}, Label: {label}\n")

print("=" * 70)
