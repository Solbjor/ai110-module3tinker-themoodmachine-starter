# The Mood Machine

The Mood Machine is a simple text classifier that begins with a rule based approach and can optionally be extended with a small machine learning model. It tries to guess whether a short piece of text sounds **positive**, **negative**, **neutral**, or even **mixed** based on patterns in your data.

This lab gives you hands on experience with how basic systems work, where they break, and how different modeling choices affect fairness and accuracy. You will edit code, add data, run experiments, and write a short model card reflection.

---

## Repo Structure

```plaintext
├── dataset.py         # Starter word lists and example posts (you will expand these)
├── mood_analyzer.py   # Rule based classifier with TODOs to improve
├── main.py            # Runs the rule based model and interactive demo
├── ml_experiments.py  # (New) A tiny ML classifier using scikit-learn
├── model_card.md      # Template to fill out after experimenting
└── requirements.txt   # Dependencies for optional ML exploration
```

---

## Getting Started

1. Open this folder in VS Code.
2. Make sure your Python environment is active.
3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the rule-based starter:

    ```bash
    python main.py
    ```

If pieces of the analyzer are not implemented yet, you will see helpful errors that guide you to the TODOs.

To try the ML model later, run:

```bash
python ml_experiments.py
```

---

## What You Will Do

During this lab you will:

- Implement the missing parts of the rule based `MoodAnalyzer`.
- Add new positive and negative words.
- Expand the dataset with more posts, including slang, emojis, sarcasm, or mixed emotions.
- Observe unusual or incorrect predictions and think about why they happen.
- Train a tiny machine learning model and compare its behavior to your rule based system.
- Complete the model card with your findings about data, behavior, limitations, and improvements.
- The goal is to help you reason about how models behave, how data shapes them, and why even small design choices matter.

---

## Tips

- Start with preprocessing before updating scoring rules.
- When debugging, print tokens, scores, or intermediate choices.
- Ask an AI assistant to help create edge case posts or unusual wording.
- Try examples that mislead or confuse your model. Failure cases teach you the most.



[x] Modify a rule based classifier and understand how handcrafted logic shapes predictions.
[x] Build and label a small dataset and see how data choices influence both rule based and ML systems.
[x] Identify where simple models fail, including sarcasm, slang, emojis, ambiguity, and cultural context.
[x] Train and evaluate a tiny ML model to compare learned behavior with manual rules.
[x] Write a model card that clearly communicates how your system works and where it breaks.


# Summary 

Students should understand the core concepts about how to not only work with your own ML model, but also how to tweak additions and revisions properly via AI assistance. Students will learn about things such as preprocessing and tokenizing, key concepts crucial to the operation of AI ML models. I honestly think students are more likely to struggle with section 3 discussing the rule based model due to how it loops or skips determiners depending on linguistic features. Its not just about the code logic, its about learning how to take into account all these considerations when creating a model. AI was very helpful with stress testing the system and analyzing the different results for comparison. In contrast the AI was misleading when it came to some of the logic behind the rules-based-model, and I had to pivot to re-analyzing certain sections of it. 
