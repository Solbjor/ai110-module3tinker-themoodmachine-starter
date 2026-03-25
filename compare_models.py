#!/usr/bin/env python3
"""Compare rule-based vs ML model performance."""

from ml_experiments import train_ml_model, evaluate_on_dataset, predict_single_text
from mood_analyzer import MoodAnalyzer
from dataset import SAMPLE_POSTS, TRUE_LABELS

print("=" * 80)
print("COMPARING RULE-BASED vs ML MODELS")
print("=" * 80)
print()

# Train ML model
print("Training ML model...")
vectorizer, ml_model = train_ml_model(SAMPLE_POSTS, TRUE_LABELS)
print("ML model trained successfully.\n")

# Evaluate both models side by side
rule_analyzer = MoodAnalyzer()

print("=" * 80)
print("DETAILED COMPARISON")
print("=" * 80)
print()

rule_correct = 0
ml_correct = 0

for i, (text, true_label) in enumerate(zip(SAMPLE_POSTS, TRUE_LABELS)):
    rule_pred = rule_analyzer.predict_label(text)
    ml_pred = predict_single_text(text, vectorizer, ml_model)
    
    rule_match = "✓" if rule_pred == true_label else "✗"
    ml_match = "✓" if ml_pred == true_label else "✗"
    
    if rule_pred == true_label:
        rule_correct += 1
    if ml_pred == true_label:
        ml_correct += 1
    
    print(f"Post {i+1}: {text[:50]}...")
    print(f"  True label: {true_label}")
    print(f"  Rule-based {rule_match}: {rule_pred}")
    print(f"  ML model   {ml_match}: {ml_pred}")
    if rule_pred != ml_pred:
        print(f"  *** DISAGREEMENT ***")
    print()

print("=" * 80)
print("ACCURACY SUMMARY")
print("=" * 80)
rule_accuracy = rule_correct / len(SAMPLE_POSTS)
ml_accuracy = ml_correct / len(SAMPLE_POSTS)

print(f"Rule-based: {rule_accuracy:.1%} ({rule_correct}/{len(SAMPLE_POSTS)})")
print(f"ML model:   {ml_accuracy:.1%} ({ml_correct}/{len(SAMPLE_POSTS)})")
print()

if rule_accuracy > ml_accuracy:
    print(f"Winner: Rule-based by {(rule_accuracy - ml_accuracy):.1%}")
elif ml_accuracy > rule_accuracy:
    print(f"Winner: ML model by {(ml_accuracy - rule_accuracy):.1%}")
else:
    print("Tie!")
