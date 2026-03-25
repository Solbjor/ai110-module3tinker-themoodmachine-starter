# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

You may complete this model card for whichever version you used, or compare both if you explored them.

## 1. Model Overview

**Model type:**  
**Both rule-based and ML models** were implemented and evaluated.

**Intended purpose:**  
Classify short informal text (social media posts, chat messages) into mood categories: **positive**, **negative**, **neutral**, or **mixed**.

**How it works (brief):**  

**Rule-based (mood_analyzer.py):**
- Counts positive/negative sentiment words from predefined lists
- Applies negation handling: "not happy" flips to negative
- Detects mixed emotions when both positive AND negative words appear
- Emoji patterns (:), :(, 😭, 💪) are strong sentiment signals
- Generates interpretable explanations for each prediction

**ML (ml_experiments.py):**
- Converts text to bag-of-words numeric features
- Trains logistic regression on labeled posts
- Learns which word patterns predict each label automatically
- No explicit rules or word lists needed



## 2. Data

**Dataset description:**  
- **Started with:** 6 example posts
- **Expanded to:** 20 labeled posts total
- **Label distribution:** 6 positive, 5 negative, 4 neutral, 5 mixed
- **New examples added:** Posts from stress testing and validation phases

**Example posts (showing language diversity):**
- Modern slang: "No cap", "lowkey", "fr fr", "ate and left no crumbs"
- Emoji usage: 😭, 💪, 😂, :/
- Mixed emotions: "tired but hopeful", "hate how much I love this"
- Uncertainty: "not sure how I feel", "Whatever"
- Formal structure: "This food tastes good", "My day was pretty bad"

**Labeling process:**  
Labels assigned based on:
- Dominant sentiment words (positive/negative words present)
- Contradictory signals (both positive AND negative → mixed)
- Uncertainty markers → neutral
- Context clues and emojis

**Hard-to-label posts (ambiguity acknowledged):**
- "Ngl i'm not sure how i feel about this lol" — could be neutral OR negative OR mixed
- "My day was pretty bad but I learned something new" — could be negative OR mixed (genuinely both)
- Sarcasm examples were NOT reliably labeled (e.g., "I love getting stuck in traffic")

**Important characteristics:**  
- Contains modern slang ("lowkey", "no cap", "ate", "fr fr", "ngl")
- Includes diverse emojis (😭, 💪, 😂, :/)
- Mixes formal and informal language
- Represents contradictory/mixed emotions (a key challenge)
- Represents genuine uncertainty and neutral stances
- Small dataset (20 posts) typical of classroom projects

**Dataset issues:**  
- **Small size:** Only 20 examples → ML will overfit
- **Limited diversity:** No sarcasm with clear labels; no formal writing; no non-English
- **Potential labeling ambiguity:** Some posts admit multiple valid interpretations
- **Label imbalance:** Slightly more mixed/positive than negative
- **Cultural/dialectal bias:** Skews toward American English slang over other varieties
- **No objective ground truth:** Some posts don't have a single "correct" label

## 3. How the Rule-Based Model Works

**Scoring rules implemented:**

1. **Preprocessing:**
   - Lowercase text
   - Normalize repeated chars ("sooo" → "so")
   - Separate punctuation with spaces for clean tokenization

2. **Sentiment word lists (35 positive, 26 negative):**
   - **Positive:** happy, love, excited, awesome, amazing, fire, lit, sick, slay, boss, proud, cool, best, nice, wonderful, etc.
   - **Negative:** sad, hate, terrible, awful, angry, tired, stressed, trash, stuck, hard, worst, etc.

3. **Negation handling (key enhancement):**
   - Recognizes: "not", "never", "can't", "don't", "didn't", "would't", etc.
   - Looks back up to 3 tokens for negation (skips transparent words like "the", "a")
   - Flips polarity when found: "not bad" scores as positive
   - **Example:** "This is not the worst thing" correctly negates "worst"

4. **Scoring mechanism:**
   - Start score = 0
   - Positive word + not negated: +1
   - Positive word + negated: -1
   - Negative word + not negated: -1
   - Negative word + negated: +1

5. **Emoji enhancement:**
   - Positive emojis (:), :D, 😊, 😂, 💪, 🥲, 😍): +2 each
   - Negative emojis (:(, >:(, 😭, 😞, 💔): -2 each
   - Checked separately before tokenization (emojis split into characters)
   - Emojis worth 2× individual words (strong signals)

6. **Mixed emotion detection (key innovation):**
   - Checks for BOTH positive AND negative sentiment words
   - If has_both AND abs(score) ≤ 1 → label "mixed"
   - Catches contradictions like "hate how much I love this"

7. **Label assignment:**
   - score > 0 → "positive"
   - score < 0 → "negative"
   - score == 0 → "neutral" (unless mixed detected)
   - Mixed can occur at any score if contradictions detected

**Strengths:**
- **Transparent:** Can explain exactly why each prediction was made
- **Fast:** No training; rules applied instantly
- **Consistent:** Deterministic — same input always gives same output
- **Generalizable:** Works on new text without retraining
- **Handles negation:** Unlike naive bag-of-words approaches
- **Detects mixed emotions:** Explicitly looks for contradictions

**Weaknesses:**
- **Sarcasm blind:** "I love getting stuck in traffic" predicts positive (should be negative)
- **Limited to known words:** New slang isn't recognized (score unchanged)
- **Context insensitive:** Ignores word relationships and sentence structure
- **Brittle:** Each edge case might require adding new rules
- **Emoji dependent:** Different performance with/without emoji presence
- **Mixed emotion range:** Only detects mixed when |score| ≤ 1 (strong contradictions missed)

## 4. How the ML Model Works

**Features used:**
- **Bag-of-words (CountVectorizer)** with default settings
- Each post becomes a word count vector
- No preprocessing, stemming, or n-grams
- Vocabulary learned from training data

**Training process:**
- Uses all 20 labeled posts from SAMPLE_POSTS and TRUE_LABELS
- LogisticRegression classifier (max_iter=1000)
- No train/test split (evaluation on same training data)

**Training behavior observations:**
- When dataset expanded from 14 → 20 posts: maintained 100% accuracy
- Model correctly learned mixed emotion patterns from contradictory word combinations
- Successfully disambiguates "lol" in different contexts (genuine laughter vs. tone marker)
- Learns that "not sure" + certain words = neutral, without explicit negation rules

**Strengths:**
- **Learns automatically:** No manual rule creation needed
- **Pattern discovery:** Learns "tired + hopeful" = mixed without programming it
- **Context-aware combinations:** Learns word dependencies implicitly
- **Flexible:** Can adapt as data changes
- **Better mixed detection:** Learns contextual patterns humans might miss
- **No word list needed:** Discovers important words from data

**Weaknesses:**
- **Severe overfitting:** 100% on 20 training posts suggests memorization, not generalization
- **Small dataset:** 20 examples insufficient for reliable ML model
- **Black box:** Can't explain which words led to specific predictions
- **No negation awareness:** Must learn negation from limited data (unreliable)
- **Inherits training label quality:** Any labeling errors are learned as patterns
- **Would fail on new data:** No validation that generalization works

## 5. Evaluation

**How you evaluated:**
- Tested both models on all 20 labeled posts
- Compared predictions against human labels (TRUE_LABELS)
- Calculated accuracy and identified failure patterns
- Noted where models agree vs. disagree

**Results:**

| Model | Accuracy | Correct | Failed |
|-------|----------|---------|--------|
| Rule-Based | 65.0% | 13/20 | 7/20 |
| ML (LogReg) | 100.0% | 20/20 | 0/20* |

*ML high accuracy likely due to overfitting (same data for train+test)

**Examples of correct predictions (Rule-Based):**

1. **"I love this class so much"** ✓ positive
   - Why correct: Matches "love" (+1), no negation
   - Score: +1 → "positive"
   - Confidence: High (straightforward signal)

2. **"I hate how much i love this show"** ✓ mixed
   - Why correct: Both "hate" (-1) and "love" (+1) present, combined score ≈ 0
   - Applied mixed emotion rule (has_both AND |score| ≤ 1)
   - Score: 0 → "mixed"
   - Confidence: Good (explicitly detected contradiction)

3. **"I'm not happy about this"** ✓ negative
   - Why correct: "not" negates "happy", score = -1
   - Negation handling worked: "not happy" = negative (correct flip)
   - Score: -1 → "negative"
   - Confidence: Good (negation logic applied)

**Examples of incorrect predictions (Rule-Based):**

1. **"Feeling tired but kind of hopeful"** ✗ predicted negative, true: mixed
   - Why wrong: "tired" (-1) + "hopeful" (+1) = score approaches 0, but doesn't trigger mixed rule exactly
   - Root cause: Multiple positive/negative words interaction, threshold misses edge case
   - ML model: Predicts **mixed** ✓ (learned the pattern)

2. **"Ngl i'm not sure how i feel about this lol"** ✗ predicted positive, true: neutral
   - Why wrong: Only "lol" matches (positive word list), score = +1 → "positive"
   - Root cause: "lol" treated as sentiment word, but here used as tone marker (dismissive uncertainty)
   - ML model: Predicts **neutral** ✓ (learned "not sure" + "lol" = neutral tone)
   - This is core limitation: "lol" has multiple meanings

3. **"No cap this project is lowkey hard but i'm getting it done :)"** ✗ predicted positive, true: mixed
   - Why wrong: Emoji :) (+2) + slang "fire" isn't recognized, but "hard"
   - Actually scores around +1-2 depending on what matches in slang list
   - Root cause: Doesn't learn that "hard but getting it done" = mixed struggle+pride
   - ML model: Predicts **mixed** ✓ (learned bidirectional signals)

**Key Finding: Where ML and Rule-Based Differ**
- **ML better at:** Mixed emotion detection, learning word combinations
- **Rule-based better at:** Transparency, consistency, predictable behavior
- **ML worse at:** Explaining predictions, generalizing beyond training data
- **Rule-based worse at:** Tone-dependent words like "lol", learning from data

## 6. Limitations

**Critical Limitations:**

1. **Cannot detect sarcasm (both models)**
   - Example: "I love getting stuck in traffic"
   - Current: Both predict positive (has "love")
   - Expected: Negative (sarcasm = frustration)
   - Why: No world knowledge or sentiment reversal logic
   - Impact: Sarcastic text will be systematically misclassified

2. **Limited vocabulary scope (both models)**
   - Example: "This is bussin" (Gen Z slang = really good)
   - Current: Neutral (no match)
   - Expected: Positive
   - Why: ML only knows training vocabulary; rule-based only has hardcoded words
   - Impact: New slang and language evolution break both models

3. **Small training dataset**
   - Rule-based: Limited exposure to edge cases
   - ML: 100% accuracy is false confidence; actual generalization likely ~60-65%
   - Impact: Both models will fail on new, unseen examples

4. **No context understanding (rule-based limitation)**
   - Example: "I'm exhausted but at least it's Friday" (temporal context = relief)
   - Works here by accident (sees contradictions), but would fail on harder cases
   - Why: Treats isolated words, not semantic relationships
   - Impact: Some correct predictions are luck, not understanding

5. **Limited negation scope (rule-based)**
   - Example: "I would not say that I'm unhappy about this" (10+ tokens)
   - Current: Misses negation, treats "unhappy" as strong negative
   - Expected: Should recognize "not...unhappy" = positive
   - Why: Only looks back 3 tokens
   - Impact: Complex negations fail

6. **Mixed emotion detection only works in narrow range (rule-based)**
   - Example: "I absolutely love this but it's terrible!" (strong both sides)
   - Current: Might score +1 or -1 depending on word frequency
   - Expected: Should be mixed
   - Why: |score| ≤ 1 threshold too strict for strong contradictions
   - Impact: Genuine mixed emotions with strong signals missed

7. **Dataset bias toward specific demographic**
   - All examples: Modern English, informal, social media tone
   - Missing: Formal writing, professional tone, other languages, cultural variations
   - Example: British "That's brilliant" (sarcasm) would be misclassified
   - Impact: Model underperforms on different language styles/dialects

8. **Emoji dependency (rule-based)**
   - Performance varies significantly based on emoji presence
   - Only handles subset of common emojis
   - Impact: Posts with unknown emojis are ignored

## 7. Ethical Considerations

**Risks in Real-World Deployment:**

**Misclassification & Safety Risks:**
- Neutral prediction for distressed person: "I'm not sure I can keep going" might classify as neutral (has "not sure") instead of negative, missing a crisis signal
- Impact: Automated support systems could fail to help vulnerable users
- Sarcasm misclassification: "I'm fine getting stuck in this situation" predicted positive but person is suffering
- Impact: Mental health screening systems could be dangerously inaccurate

**Cultural & Linguistic Bias:**
- AAVE (African American Vernacular) phrasing may be systematically misclassified
- Gen Z slang overrepresented in training vs. other age groups
- Sarcasm frequency varies by culture (British English more sarcastic than American); model may be skewed
- Impact: Certain cultural/demographic groups experience worse model performance, leading to unfair treatment

**Community-Specific Misinterpretation:**
- Idioms and colloquialisms unknown to dataset fail silently
- Example: "That's sick!" would be negative ("sick" in word lists) but means positive in context
- Impact: Non-native English speakers, immigrant communities, and regional subcultures unfairly disadvantaged

**Tone & Context Insensitivity:**
- Irony: "Oh great, another meeting" (sarcasm) predicted positive
- Passive-aggressive: "Fine, whatever" (resignation) might be neutral
- Impact: Indirect emotional expression (common in trauma survivors, neurodivergent individuals) is missed

**Privacy & Surveillance:**
- If deployed to classify user messages in real-time centrally
- Messages may contain sensitive, personal, medical, or legal information
- Impact: Privacy violation unless opt-in and data-minimized

**High-Stakes Applications are Dangerous:**
- Mental health screening: Model at 65% accuracy could miss people in crisis
- Content moderation: Misclassified hateful speech could go unchecked
- Employee sentiment tracking: Privacy violation + unfair performance evaluation
- Recommendation systems: Poor mood classification leads to irrelevant recommendations

**Recommended Safeguards:**
- **Never use alone for high-stakes decisions** (mental health intervention, safety decisions)
- **Pair with human review** for any consequential use
- **Audit bias regularly:** Test on diverse demographic groups, languages, dialects
- **Transparent documentation:** Tell users model has known limitations (sarcasm, cultural bias)
- **Collect feedback:** Retrain when users report misclassifications
- **Avoid surveillance:** Require explicit opt-in; limit data retention
- **Test on diverse data:** Before deployment, validate on users from target communities
- **Reduce confidence:**  Report uncertainty scores instead of hard predictions

## 8. Ideas for Improvement

**For Rule-Based Model:**
- **Better negation:** Use dependency parsing to find true negation scope (not limited to 3 tokens)
- **Intensifier detection:** Recognize "very", "so", "absolutely", "really" modify adjacent words
- **Weak sarcasm signals:** Flag incongruent combinations ("love stuck", "happy terrible") as potentially sarcastic
- **Contextual word weights:** Different words have different strengths ("hate" > "annoyed")
- **Comprehensive word lists:** Add regional slang, cultural terms, Gen Z expressions, AAVE
- **Emoji context:** Use emoji position (start/end of sentence) to weight influence
- **Multi-word expressions:** Handle "kind of", "a bit", "sort of" as intensity modifiers

**For ML Model:**
- **Larger dataset:** Collect 100+ labeled posts to reduce overfitting
- **Train/test split:** Use k-fold cross-validation to detect overfitting
- **TF-IDF features:** Replace word counts with term frequency-inverse document frequency
- **N-grams:** Use bigrams ("not good", "kind of") and trigrams for phrase patterns
- **Better preprocessing:** Handle URLs, @mentions, repeated punctuation (#!!!)
- **Neural networks:** Try embedding + LSTM/transformer to capture word order and long-range dependencies
- **Class weights:** Handle imbalanced labels (more mixed than negative)
- **Confidence scores:** Return probabilities ("70% positive, 20% mixed, 10% negative") instead of hard predictions

**Data & Methodology:**
- **Balanced collection:** Ensure equal samples per label (currently more mixed/positive)
- **Multi-rater labeling:** Have multiple people label same posts; measure inter-rater agreement
- **Domain-specific models:** Train separate models for customer service vs. social media vs. mental health
- **Sarcasm annotation:** Explicitly label sarcasm examples or add indicators
- **Contextual examples:** Include longer posts with more context
- **Hard case review:** Have humans label edge cases, document disagreements
- **Demographic testing:** Collect test data from diverse age groups, regions, cultures

**Hybrid & Advanced Approaches:**
- **Ensemble method:** Combine rule-based and ML (use rule-based for straightforward cases, ML for ambiguous)
- **Rule-based + ML features:** Use rule outputs as features for ML model
- **Confidence thresholding:** When ML model uncertain (probability near 0.5), fall back to rule-based
- **Continual learning:** Update model as users provide feedback on misclassifications
- **Explainability:** Add LIME/SHAP to explain ML predictions

**Evaluation Improvements:**
- **Hold-out test set:** Reserve 20% of data for true generalization testing
- **Stress testing:** Test on sarcasm, slang, formal writing, other languages
- **Error analysis by demographic:** Break down performance by user group
- **Confidence calibration:** Ensure probability predictions match real accuracy
- **Ablation studies:** Disable each feature (negation, emoji, etc.) and measure impact

---

## 9. Summary: Rule-Based vs. ML

| Aspect | Rule-Based | ML Model |
|--------|-----------|----------|
| **Accuracy (on 20 posts)** | 65% | 100%* |
| **Interpretability** | Excellent (can explain each prediction) | Poor (black box) |
| **Training required** | No | Yes |
| **Speed** | Fast (O(n) words per post) | Fast (already trained) |
| **Generalization to new data** | Honest (~65%) | Likely fails (overfitting) |
| **Best at** | Positive/negative posts, negation | Mixed emotions, learned patterns |
| **Worst at** | Mixed emotions, tone markers | Unseen words, new contexts |
| **Sarcasm detection** | Predicts positive (wrong) | Depends on training data |
| **Code transparency** | Can audit/modify rules | Model weights not interpretable |
| **Maintenance** | Update word lists | Retrain on new data |

**Why Each Model Behaves Differently:**
- Rule-based: Explicit, predictable failures on edge cases (sarcasm, "lol" ambiguity)
- ML: Learns patterns from data; strong on mixed emotions but weak on truly new text

**Recommendation:**
- **For interpretability & trust:** Use rule-based (can explain why)
- **For accuracy on trained patterns:** Use ML (learns combinations)
- **For production:** Use hybrid (rule-based for certain, ML for ambiguous; human review for high-stakes)
- **For future:** Collect more data (100+ posts) before relying on ML alone

---

## Final Notes

This model card documents the current state of both systems. Both approaches revealed fundamental challenges in mood classification:
- **Rule-based limits:** Can't capture semantic complexity without explicit rules
- **ML limits:** Needs more data and careful evaluation to avoid false confidence
- **Human baseline:** Even humans sometimes disagree on mood labels for the same post

The Mood Machine serves as a teaching tool demonstrating that **no single approach is best** — only different tradeoffs suitable for different use cases.
