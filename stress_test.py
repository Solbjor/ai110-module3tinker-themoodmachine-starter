#!/usr/bin/env python3
"""Stress test suite for the Mood Machine.

Tests designed to break the rule-based classifier and expose weaknesses.
"""

from mood_analyzer import MoodAnalyzer

analyzer = MoodAnalyzer()

# Breaker sentences: designed to confuse the model
breakers = [
    # Sarcasm - positive words but negative intent
    ("I love getting stuck in traffic", "negative (sarcasm)"),
    ("This is just wonderful", "negative (sarcasm)"),
    ("Oh great, another meeting", "negative (sarcasm)"),
    
    # Slang with multiple meanings
    ("That's sick!", "positive (slang)"),
    ("This is wicked cool", "positive (slang)"),
    ("This fire track is amazing", "positive (slang)"),
    
    # Emojis and tone indicators
    ("I'm fine :)", "positive (emoji override)"),
    ("Everything is great :-(", "negative (emoji override)"),
    ("I'm okay |:", "neutral (sarcasm tone)"),
    
    # Mixed/complex emotions
    ("I'm exhausted but proud of myself", "mixed"),
    ("This is hard but I love the challenge", "mixed"),
    ("I hate how much I need this break", "mixed"),
    
    # Negation edge cases
    ("I'm not bad at this", "positive (double negative)"),
    ("This is not the worst thing", "positive (double negative)"),
    ("I don't love the taste", "negative (negated positive)"),
    
    # Empty or minimal signal
    ("Okay then", "neutral"),
    ("Whatever", "neutral"),
    ("Sure", "neutral"),
]

print("=" * 80)
print("MOOD MACHINE STRESS TEST - Breaker Sentences")
print("=" * 80)
print()

failures = []
successes = []

for text, expected_category in breakers:
    score = analyzer.score_text(text)
    predicted = analyzer.predict_label(text)
    tokens = analyzer.preprocess(text)
    
    # Simple heuristic: if expected says "positive/negative/neutral" at start, check if match
    expected_label = expected_category.split()[0].lower()
    is_correct = predicted == expected_label
    
    status = "✓" if is_correct else "✗"
    
    print(f'{status} Predicted: {predicted:8s} | Score: {score:3d} | "{text}"')
    print(f'   Expected: {expected_category}')
    print(f'   Tokens: {tokens}')
    print()
    
    if not is_correct:
        failures.append({
            'text': text,
            'predicted': predicted,
            'expected': expected_label,
            'score': score,
            'tokens': tokens,
            'category': expected_category
        })
    else:
        successes.append(text)

print("=" * 80)
print(f"RESULTS: {len(successes)}/{len(breakers)} correct")
print("=" * 80)
print()

if failures:
    print("FAILURE PATTERNS IDENTIFIED:\n")
    
    # Analyze failure types
    sarcasm_failures = [f for f in failures if "sarcasm" in f['category']]
    slang_failures = [f for f in failures if "slang" in f['category']]
    emoji_failures = [f for f in failures if "emoji" in f['category']]
    mixed_failures = [f for f in failures if "mixed" in f['category']]
    negation_failures = [f for f in failures if "double negative" in f['category']]
    
    if sarcasm_failures:
        print(f"❌ SARCASM FAILURES ({len(sarcasm_failures)}):")
        for f in sarcasm_failures:
            print(f"   '{f['text']}'")
            print(f"   → Predicted {f['predicted']} (score={f['score']}) but should be {f['expected']}")
        print()
    
    if slang_failures:
        print(f"❌ SLANG FAILURES ({len(slang_failures)}):")
        for f in slang_failures:
            print(f"   '{f['text']}'")
            print(f"   → Predicted {f['predicted']} (score={f['score']}) but should be {f['expected']}")
        print()
    
    if emoji_failures:
        print(f"❌ EMOJI FAILURES ({len(emoji_failures)}):")
        for f in emoji_failures:
            print(f"   '{f['text']}'")
            print(f"   → Predicted {f['predicted']} (score={f['score']}) but should be {f['expected']}")
        print()
    
    if negation_failures:
        print(f"❌ NEGATION FAILURES ({len(negation_failures)}):")
        for f in negation_failures:
            print(f"   '{f['text']}'")
            print(f"   → Predicted {f['predicted']} (score={f['score']}) but should be {f['expected']}")
        print()
    
    if mixed_failures:
        print(f"❌ MIXED EMOTION FAILURES ({len(mixed_failures)}):")
        for f in mixed_failures:
            print(f"   '{f['text']}'")
            print(f"   → Predicted {f['predicted']} (score={f['score']}) but should handle mixed")
        print()
