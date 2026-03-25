"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    # Modern slang
    "fire",
    "lit",
    "sick",
    "dope",
    "slaps",
    "ate",
    "slay",
    "boss",
    "stoked",
    "hyped",
    "blessed",
    "proud",
    # Informal expressions
    "lol",
    "lmao",
    # Additional positive descriptors
    "cool",
    "best",
    "nice",
    "wonderful",
    "break",  # as in "taking a break"
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    # Modern slang
    "trash",
    "garbage",
    "sucks",
    "suck",
    "lame",
    "wack",
    "exhausted",
    "drained",
    "nope",
    "nah",
    # Expressions of struggle
    "cannot",
    # Additional negative descriptors
    "hard",
    "difficult",
    "worst",
    "stuck",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    "No cap this project is lowkey hard but i'm getting it done :)",
    "Exhausted but at least it's friday 😭",
    "I hate how much i love this show",
    "Feeling like a boss today 💪💪",
    "Ngl i'm not sure how i feel about this lol",
    "Literally cannot with this energy rn",
    "Ate and left no crumbs fr fr 😂",
    "Stuck in my own head today :/ ok maybe tomorrow will be better",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    "mixed",     # "No cap this project is lowkey hard but i'm getting it done :)"
    "mixed",     # "Exhausted but at least it's friday 😭"
    "mixed",     # "I hate how much I love this show"
    "positive",  # "Feeling like a boss today 💪💪"
    "neutral",   # "Ngl I'm not sure how I feel about this lol"
    "negative",  # "Literally cannot with this energy rn"
    "positive",  # "Ate and left no crumbs fr fr 😂"
    "mixed",     # "Stuck in my own head today :/ ok maybe tomorrow will be better"
]

# TODO: Add 5-10 more posts and labels.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
