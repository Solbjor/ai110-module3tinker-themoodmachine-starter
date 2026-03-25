#!/usr/bin/env python3
"""Quick evaluation script to test the mood analyzer."""

from mood_analyzer import MoodAnalyzer
from dataset import SAMPLE_POSTS, TRUE_LABELS

analyzer = MoodAnalyzer()

print("=== Rule Based Evaluation on SAMPLE_POSTS ===\n")
correct = 0
for post, true_label in zip(SAMPLE_POSTS, TRUE_LABELS):
    pred = analyzer.predict_label(post)
    match = "✓" if pred == true_label else "✗"
    score = analyzer.score_text(post)
    print(f'{match} Score={score:2d} | predicted={pred:8s}, true={true_label:8s} | "{post}"')
    if pred == true_label:
        correct += 1

accuracy = correct / len(SAMPLE_POSTS)
print(f"\n=== Results ===")
print(f"Accuracy: {accuracy:.1%} ({correct}/{len(SAMPLE_POSTS)})")
