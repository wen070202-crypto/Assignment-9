# NLP Comparative Analysis Project

## Purpose of the Project
The primary purpose of this project is to perform basic Natural Language Processing (NLP) analysis using the Natural Language Toolkit (NLTK) in Python. Unstructured text data is inherently difficult for machines to understand. This project utilizes NLP techniques—such as tokenization, stemming, lemmatization, Named Entity Recognition (NER), and n-gram analysis—to extract meaning, determine the subject matter of three different texts, and predict the authorship of a fourth text based on linguistic patterns.

## Class Design and Implementation
To ensure the code is robust, readable, and highly reusable (adhering to Python best practices), I implemented an Object-Oriented Programming (OOP) approach by creating a `TextAnalyzer` class. 

Instead of writing repetitive procedural code for each of the four text files, the class encapsulates the data (the text) and the behaviors (preprocessing, analyzing) into a single entity. This design choice makes the main execution block clean and easy to follow.

### Class Attributes
* `self.file_path`: Stores the path of the text file being analyzed.
* `self.text`: Stores the raw string content of the text file. It is populated using a robust private method `_read_file()` that handles `FileNotFoundError` and `IOError`.
* `self.lemmatized_tokens`: An array that caches the tokens after they have passed through tokenization, stopword removal, and lemmatization. Caching this prevents redundant processing when multiple analytical methods are called.

### Class Methods
* `_read_file(self)`: A private helper method with `try-except` blocks to ensure the program does not crash if a file is missing (Robustness).
* `preprocess(self)`: Converts text to lowercase, tokenizes it, removes non-alphanumeric characters and stopwords, applies the `PorterStemmer` (as required), and applies the `WordNetLemmatizer` for cleaner downstream analysis.
* `get_top_tokens(self, n=20)`: Calculates the frequency distribution of the lemmatized tokens and returns the top 20 most common words.
* `get_named_entities(self)`: Uses NLTK's `pos_tag` and `ne_chunk` on the *raw, un-lowercased* text (since capitalization is key for NER) to extract and count proper nouns and entities.
* `get_trigrams(self, n=3, top_k=20)`: Generates n-grams (trigrams) to capture sequential word patterns, which are indicative of an author's writing style.

## Subject Analysis (Texts 1, 2, and 3)
Based on the extraction of top tokens and Named Entities (NER), the subjects of the first three texts are all variations of the classic "Romeo and Juliet" story, but written in distinct literary genres:

* **Text 1 Subject:** A **Cosmic Horror / Lovecraftian** adaptation of Romeo and Juliet. This is deduced from the high frequency of tokens like "cosmic", "eldritch", "abominations", alongside the core entities "Romeo", "Juliet", and "Verona".
* **Text 2 Subject:** A **High Fantasy / Elven** adaptation of Romeo and Juliet. The subject is evident from terms like "elven", "emerald", and "enchanted slumber", mixed with traditional entities like "House Capulet" and "Friar Laurence".
* **Text 3 Subject:** A **Dark Political Fantasy** adaptation of Romeo and Juliet. The text heavily features tokens representing political intrigue such as "treachery", "blood", "feuds", and "machinations", characterizing a darker, politically driven narrative.

## Authorship Analysis (Text 4)
Using trigram (n=3) overlap analysis, I compared the linguistic patterns of Text 4 against Texts 1, 2, and 3. 

* **Conclusion:** The author of **Text 3** is the most likely author of **Text 4**. 
* **Reasoning:** Text 4 is a dark epic fantasy narrative (featuring lords, treason, and icy creatures). Its writing style, tone, and specific word groupings (trigrams)—especially the recurring use of "House [Name]", and themes of dark politics, blood, and shadows—share the highest n-gram intersection with Text 3.

## Limitations Added / Discovered
1. **NER Inaccuracies:** NLTK's `ne_chunk` is a foundational tool but is not perfect. It sometimes misclassifies the start of sentences as entities simply because they are capitalized, or misses complex multi-word entities. A limitation of this project is relying solely on `ne_chunk` rather than a more advanced model like spaCy.
2. **Stemming vs. Lemmatization:** While the assignment requested stemming, stemming often chops off ends of words abruptly (e.g., "historical" to "histor"). I implemented both but primarily utilized lemmatization for the final analysis to maintain readable and accurate n-grams.
3. **Authorship Attribution Limitations:** Using pure intersection of top trigrams is a basic heuristic for authorship attribution. While it worked well here due to distinct genre vocabularies, authors might write about different subjects using different words. A more robust limitation added here is that trigram overlap alone might not be enough for extremely short texts, making advanced Machine Learning models (like Naive Bayes or SVM) necessary for definitive proof.

## Generative AI Use
The use of Generative AI (Gemini) was utilized to brainstorm the OOP class structure, ensure robust error handling `try-except` patterns, and format this README file. The chat log detailing this interaction is included in the repository as `AI_Chat_Log.pdf` / `AI_Chat_Log.txt`.