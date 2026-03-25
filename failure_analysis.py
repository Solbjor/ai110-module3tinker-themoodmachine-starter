#!/usr/bin/env python3
"""Deep analysis of a specific failure case."""

from mood_analyzer import MoodAnalyzer

analyzer = MoodAnalyzer()

# The failing example
text = "Ngl i'm not sure how i feel about this lol"
true_label = "neutral"

# Get detailed breakdown
tokens = analyzer.preprocess(text)
score = analyzer.score_text(text)
predicted = analyzer.predict_label(text)

print("=" * 80)
print("DETAILED ANALYSIS: Why does this fail?")
print("=" * 80)
print()

print(f"Text: {text}")
print(f"True label: {true_label}")
print(f"Predicted: {predicted}")
print(f"Score: {score}")
print()

print(f"Tokens: {tokens}")
print()

# Check which words are in the sentiment word lists
print("Token analysis:")
for token in tokens:
    if token in analyzer.positive_words:
        print(f"  '{token}' → POSITIVE WORD (in word list)")
    elif token in analyzer.negative_words:
        print(f"  '{token}' → NEGATIVE WORD (in word list)")
    else:
        print(f"  '{token}' → [no match]")
print()

# Check for explanation
explanation = analyzer.explain(text)
print(f"Model explanation: {explanation}")
print()

print("=" * 80)
print("REASONING:")
print("=" * 80)
print()
print(f"ACTUAL ISSUE:")
print(f"  - 'lol' is treated as POSITIVE. But here, 'lol' is used as a")
print(f"    tone marker meaning 'laugh out loud', not expressing positivity.")
print(f"  - Text structure: 'not sure how I feel' = uncertainty (should be neutral)")
print(f"  - But 'lol' adds +1 to score, making it positive (score=1)")
print()
print(f"DECISION:")
print(f"  This is a LIMITATION of rule-based systems. The word 'lol' has")
print(f"  multiple meanings:")
print(f"    1. Genuine laughter (positive vibes)")
print(f"    2. Tone marker/dismissal (neutral or sarcastic)")
print(f"  Without context, we can't distinguish.")
print()
print(f"POSSIBLE FIXES (with tradeoffs):")
print(f"  A) Remove 'lol' from positive words")
print(f"     - Pro: Fixes this case")
print(f"     - Con: Breaks 'That's hilarious lol' type cases")
print(f"  B) Detect uncertainty words ('not sure', 'unsure') → force neutral")
print(f"     - Pro: More context-aware")
print(f"     - Con: Adds complexity, may break other cases")
print(f"  C) Keep as-is and document as known limitation")
print(f"     - Pro: Honest about what rule-based can't do")
print(f"     - Con: Wrong predictions persist")
print()
print(f"ENGINEERING DECISION: Document as limitation.")
print(f"These edge cases are exactly why ML models work better.")
