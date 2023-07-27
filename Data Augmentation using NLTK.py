# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#
# # Choose a pre-trained model suitable for paraphrasing, for example, T5
# model_name = "t5-base"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#
# def paraphrase_sentence(sentence):
#     inputs = tokenizer.encode("paraphrase: " + sentence, return_tensors="pt")
#     outputs = model.generate(inputs, max_length=50, num_return_sequences=1, num_beams=5, temperature=1.0)
#     paraphrased_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return paraphrased_sentence
#
# # Read the CSV file containing the input sentences
# input_csv_path = r'C:\Users\WIN10\PycharmProjects\pythonProject\synthetic_final_data_sensitive_analysis.csv'
# output_csv_path = 'paraphrased_sentences.csv'
#
# df = pd.read_csv(input_csv_path)
#
# # Perform paraphrasing on each sentence
# paraphrased_data = []
# for sentence in df['Sentences']:
#     paraphrased_sentence = paraphrase_sentence(sentence)
#     paraphrased_data.append(paraphrased_sentence)
#
# # Create a new DataFrame with the paraphrased sentences
# paraphrased_df = pd.DataFrame({'Paraphrased Sentences': paraphrased_data})
#
# # Write the paraphrased DataFrame to a new CSV file
# paraphrased_df.to_csv(output_csv_path, index=False)
#
# print("Paraphrasing completed. Paraphrased sentences written to:", output_csv_path)
import nltk
import random
from nltk.corpus import wordnet

# Download WordNet dataset if not already present
nltk.download('wordnet')

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def synonym_replacement(sentence, num_replacements=1):
    words = nltk.word_tokenize(sentence)
    augmented_sentences = [sentence]

    for _ in range(num_replacements):
        selected_word = None
        while not selected_word:
            random_word = nltk.word_tokenize(words[random.randint(0, len(words) - 1)])
            synonyms = get_synonyms(random_word[0])
            if len(synonyms) > 1:
                selected_word = random.choice(synonyms)

        augmented_sentence = " ".join([selected_word if word == random_word[0] else word for word in words])
        augmented_sentences.append(augmented_sentence)

    return augmented_sentences

# Input sentence
original_sentence = "Data augmentation helps improve natural language processing models."

# Perform synonym replacement data augmentation with 3 replacements
augmented_sentences = synonym_replacement(original_sentence, num_replacements=3)

print("Original Sentence:", original_sentence)
print("Augmented Sentences:")
for idx, sentence in enumerate(augmented_sentences, 1):
    print(f"{idx}. {sentence}")

