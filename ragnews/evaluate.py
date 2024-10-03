import sys
import json
import logging
import re
import time
from __init__ import ArticleDB, run_llm

#  logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)

class RAGClassifier:
    def __init__(self, valid_labels, db_path='ragnews.db'):
        self.valid_labels = valid_labels
        self.db = ArticleDB(db_path)

    def predict(self, masked_text):
        system_prompt = (
            "You are an AI assistant tasked with filling in masked tokens in a given text. "
            "Your responses should be only the predictions for the masked tokens, each prefixed by the mask identifier, one per line, nothing else. "
            "Ensure each prediction is one of the valid options provided. "
            "Do not include any explanations or extra text."
        )
        user_prompt = '''
Fill in the masked tokens in the following text.
Valid options are: {}.

Example 1:
Text: [MASK0] is the democratic nominee.
Predictions:
MASK0: Harris

Example 2:
Text: [MASK0] is the democratic nominee and [MASK1] is the republican nominee.
Predictions:
MASK0: Harris
MASK1: Trump

Now, fill in the masked tokens for the following text:
Text: {}
Predictions:
'''.format(", ".join(self.valid_labels), masked_text)
        output = self.run_with_retries(system_prompt, user_prompt)
        if not output:
            masks = re.findall(r'\[MASK\d+\]', masked_text)
            unknowns = []
            i = 0
            while i < len(masks):
                unknowns.append('unknown')
                i = i + 1
            return unknowns
        return self.extract_predictions(output)

    def run_with_retries(self, system_prompt, user_prompt, max_retries=5):
        for attempt in range(max_retries):
            try:
                output = run_llm(system=system_prompt, user=user_prompt)
                logging.info("Input Prompt:\n" + user_prompt)
                logging.info("Assistant Output:\n" + output)
                return output
            except Exception as e:
                logging.warning("Attempt " + str(attempt + 1) + " failed with error: " + str(e) + ". Retrying in 10 seconds.")
                time.sleep(10)
        logging.error("Failed after " + str(max_retries) + " attempts.")
        return None

    def extract_predictions(self, output):
        matches = re.findall(r'MASK\d+:\s*(\w+)', output)
        predictions = []
        for word in matches:
            if word in self.valid_labels:
                predictions.append(word)
            else:
                predictions.append('unknown')
        return predictions

#labels should include everything the data set has 
# not just Trump, Kamala
def pull_labels(path):
    labels = set()
    with open(path) as fin:
        for inner_line in fin:
            dp = json.loads(inner_line)
            labels.update(dp['masks'])
    return sorted(labels)

def main(path):
    labels = pull_labels(path)
    logging.info('Valid labels: ' + str(labels))
    classifier = RAGClassifier(valid_labels=labels)
    correct = 0
    total = 0
    with open(path) as json_file:
        for line in json_file:
            current_line = json.loads(line)
            masked_text = current_line.get('masked_text', '')
            true_labels = current_line.get('masks', [])
            if not masked_text or not true_labels:
                continue
            pred = classifier.predict(masked_text)
            i = 0
            while i < len(pred) and i < len(true_labels):
                if pred[i] == true_labels[i]:
                    correct = correct + 1
                i = i + 1
            total = total + len(true_labels)
    if total > 0:
        accuracy = correct / total
    else:
        accuracy = 0
    print("Accuracy: " + str(accuracy * 100) + "%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <data_file_path>")
        sys.exit(1)
    data_file_path = sys.argv[1]
    main(data_file_path)
