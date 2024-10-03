# ragnews ![Status](https://github.com/Jamesduongrx/ragnews/actions/workflows/tests.yml/badge.svg)

The RAGNews Evaluation script has a dataset with texts as masked tokens. The script then evaluates the built RAG system to predict these masked tokens accurately. The script's accuracy is detmined by comparing the predictions against the true labels in the dataset.

# Example:
```
Now, fill in the masked tokens for the following text:
Text: The debate was held at the National Constitution Center in Philadelphia and lasted for about 100 minutes. ABC's debate topics included abortion, the economy, foreign policy, and immigration. There were questions that neither participants answered. Most voters thought [MASK0] won the debate. Republicans attributed Trump's poor debate performance to their perception of unfair treatment by ABC, because the moderators fact-checked him but not [MASK0].
Predictions:
Assistant Output:
MASK0: Harris
MASK0: Biden
```

# Getting Started
Install Groq, and generate a free Groq API key,https://console.groq.com/keys. Create an .env file through the terminal and add the generated key:

```
GROQ_API_KEY=your_api_key_here
```
# Running the code






