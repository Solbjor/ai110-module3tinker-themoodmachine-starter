#!/usr/bin/env python3
"""Simple ML vs Rule-Based comparison."""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from mood_analyzer import MoodAnalyzer
from dataset import SAMPLE_POSTS, TRUE_LABELS

# Train ML model
print("Training ML model on dataset...")
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(SAMPLE_POSTS)
ml_model = LogisticRegression(max_iter=1000)
ml_model.fit(X, TRUE_LABELS)
print()

# Get predictions from both models
rule_analyzer = MoodAnalyzer()
rule_preds = [rule_analyzer.predict_label(text) for text in SAMPLE_POSTS]
ml_preds = ml_model.predict(vectorizer.transform(SAMPLE_POSTS))

# Compare
print("=" * 100)
print("RULE-BASED vs ML MODEL COMPARISON")
print("=" * 100)
print()

disagree_count = 0
rule_correct = 0
ml_correct = 0

for i, (text, true_label, rule_pred, ml_pred) in enumerate(zip(SAMPLE_POSTS, TRUE_LABELS, rule_preds, ml_preds)):
    rule_match = rule_pred == true_label
    ml_match = ml_pred == true_label
    
    if rule_match:
        rule_correct += 1
    if ml_match:
        ml_correct += 1
    
    rule_status = "✓" if rule_match else "✗"
    ml_status = "✓" if ml_match else "✗"
    
    print(f"[{i+1:2d}] {text[:55]}")
    print(f"      TRUE: {true_label:8s}")
    print(f"      Rule {rule_status}: {rule_pred:8s} | ML {ml_status}: {ml_pred:8s}", end="")
    
    if rule_pred != ml_pred:
        print(" <-- DISAGREE")
        disagree_count += 1
    else:
        print()
    print()

print("=" * 100)
print("SUMMARY")
print("=" * 100)
rule_accuracy = rule_correct / len(SAMPLE_POSTS)
ml_accuracy = ml_correct / len(SAMPLE_POSTS)

print(f"Rule-based accuracy: {rule_accuracy:.1%} ({rule_correct}/{len(SAMPLE_POSTS)})")
print(f"ML accuracy:         {ml_accuracy:.1%} ({ml_correct}/{len(SAMPLE_POSTS)})")
print(f"Disagreements:       {disagree_count}/{len(SAMPLE_POSTS)}")
print()

if rule_accuracy > ml_accuracy:
    print(f"WINNER: Rule-based (better by {(rule_accuracy - ml_accuracy):.1%})")
elif ml_accuracy > rule_accuracy:
    print(f"WINNER: ML model (better by {(ml_accuracy - rule_accuracy):.1%})")
else:
    print("TIED!")
