# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        Improvements:
          - Lowercase for consistency
          - Normalize repeated characters ("soooo" -> "so")
          - Separate punctuation from words (so "happy!" -> ["happy", "!"])
          - Extract emoji patterns like ":)" and ":("
        """
        # Lowercase
        text = text.strip().lower()
        
        # Normalize repeated characters (sooo -> so, hmmm -> hm)
        text = re.sub(r'(\w)\1{2,}', r'\1', text)
        
        # Add spaces around punctuation to separate them
        text = re.sub(r'([!?.,:;])', r' \1 ', text)
        
        # Split into tokens and filter empty strings
        tokens = [t for t in text.split() if t.strip()]
        
        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        Key enhancements:
          1. NEGATION HANDLING: flips polarity for words preceded by negation
             Can skip over determiners like "the", "a", "an"
          2. EMOJI DETECTION: checks for common emoji patterns (:), :(, etc.)
          3. Strong signal words: weight certain expressions higher
        """
        # First, check for emoji patterns before tokenization
        emoji_score = 0
        positive_emoji_patterns = [":)", ":D", "😊", "😂", "💪", "🥲", "😍"]
        negative_emoji_patterns = [":(", ">:(", "😭", "😞", "💔", "😩"]
        
        for emoji in positive_emoji_patterns:
            if emoji in text:
                emoji_score += 2
        for emoji in negative_emoji_patterns:
            if emoji in text:
                emoji_score -= 2
        
        # Now process tokens
        tokens = self.preprocess(text)
        score = 0
        
        # Define negation words that flip polarity
        negation_words = {
            "not", "never", "no", "can't", "cant", "don't", "dont", 
            "didn't", "didnt", "wouldn't", "wouldnt", "shouldn't", "shouldnt",
            "isn't", "isnt", "wasn't", "wasnt", "aren't", "arent"
        }
        
        # Words that can appear between negation and the sentiment word
        # These don't break negation
        determiners_and_adverbs = {"the", "a", "an", "some", "any", "very", "so", "quite", "really"}
        
        for i, token in enumerate(tokens):
            # Skip punctuation tokens
            if len(token) <= 1 and not token.isalpha():
                continue
            
            # Check if this word is negated by looking back
            is_negated = False
            if i > 0:
                # Look back up to 3 tokens (but skip determiners/adverbs)
                for lookback in range(1, min(4, i + 1)):
                    prev_token = tokens[i - lookback]
                    if prev_token in negation_words:
                        # Found a negation marker
                        is_negated = True
                        break
                    elif prev_token not in determiners_and_adverbs and prev_token not in {",", "and", "or"}:
                        # Hit a real word that's not a determiner; stop looking
                        break
            
            # Positive words
            if token in self.positive_words:
                if is_negated:
                    score -= 1  # Flip: "not happy" is negative
                else:
                    score += 1
            
            # Negative words
            elif token in self.negative_words:
                if is_negated:
                    score += 1  # Flip: "not sad" is positive
                else:
                    score -= 1
        
        # Add emoji signals
        score += emoji_score
        
        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        Mapping:
          - score >  0  → "positive"
          - score <  0  → "negative"
          - score == 0  → "neutral" (or "mixed" if conflicting signals detected)

        Enhancement: Detect mixed emotions by checking for both positive
        and negative words in the same text. If score is close to 0 and
        conflicting signals exist, label as "mixed".
        """
        score = self.score_text(text)
        tokens = self.preprocess(text)
        
        # Check for presence of both positive and negative sentiment words
        has_positive = any(token in self.positive_words for token in tokens)
        has_negative = any(token in self.negative_words for token in tokens)
        has_both = has_positive and has_negative
        
        # If text has conflicting signals (both pos and neg words)
        # and score is neutral or very close to neutral, label as mixed
        if has_both and abs(score) <= 1:
            return "mixed"
        
        if score > 0:
            return "positive"
        elif score < 0:
            return "negative"
        else:
            return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
