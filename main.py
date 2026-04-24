import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.util import ngrams
from collections import Counter

# 下载必要的 NLTK 数据包（如果已下载，NLTK 会自动跳过）
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception as e:
            print(f"Error downloading {res}: {e}")

class TextAnalyzer:
    """
    A class used to perform Natural Language Processing analysis on unstructured text files.
    """
    
    def __init__(self, file_path):
        """
        Initializes the TextAnalyzer with a file path and reads its content.
        
        Attributes:
            file_path (str): The path to the text file.
            text (str): The raw string content of the file.
            lemmatized_tokens (list): A list of processed tokens.
        """
        self.file_path = file_path
        self.text = self._read_file()
        self.lemmatized_tokens = []
        
    def _read_file(self):
        """
        Reads the file safely with robust error handling.
        Returns the text as a string.
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            print(f"Error: The file {self.file_path} was not found.")
            return ""
        except IOError:
            print(f"Error: An I/O error occurred while reading {self.file_path}.")
            return ""
        except Exception as e:
            print(f"Unexpected error occurred: {e}")
            return ""

    def preprocess(self):
        """
        Performs tokenization, stopword/punctuation removal, stemming, and lemmatization.
        Returns the list of lemmatized tokens.
        """
        if not self.text:
            return []

        # 1. Tokenization
        raw_tokens = word_tokenize(self.text.lower())
        
        # 2. Remove punctuation and stopwords
        stop_words = set(stopwords.words('english'))
        clean_tokens = [word for word in raw_tokens if word.isalnum() and word not in stop_words]
        
        # 3. Stemming (Included to satisfy assignment requirements)
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in clean_tokens]
        
        # 4. Lemmatization (Used for deeper analysis as it produces actual words)
        lemmatizer = WordNetLemmatizer()
        self.lemmatized_tokens = [lemmatizer.lemmatize(word) for word in clean_tokens]
        
        return self.lemmatized_tokens

    def get_top_tokens(self, n=20):
        """
        Returns the 'n' most common lemmatized tokens.
        """
        if not self.lemmatized_tokens:
            self.preprocess()
        freq_dist = Counter(self.lemmatized_tokens)
        return freq_dist.most_common(n)

    def get_named_entities(self):
        """
        Extracts Named Entities using NLTK's ne_chunk.
        Returns the total count of named entities and a list of the entities.
        """
        if not self.text:
            return 0, []

        # NER requires original capitalization, so we use self.text instead of lowercased tokens
        raw_tokens = word_tokenize(self.text)
        tagged_tokens = pos_tag(raw_tokens)
        entities = ne_chunk(tagged_tokens)
        
        named_entities = []
        for chunk in entities:
            # If the chunk has a label, it is a named entity
            if hasattr(chunk, 'label'):
                entity_name = ' '.join(c[0] for c in chunk)
                named_entities.append(entity_name)
                
        return len(named_entities), Counter(named_entities).most_common(10)

    def get_trigrams(self, n=3, top_k=20):
        """
        Generates N-grams (default n=3) from the lemmatized tokens.
        Returns the most common trigrams.
        """
        if not self.lemmatized_tokens:
            self.preprocess()
        
        trigrams = list(ngrams(self.lemmatized_tokens, n))
        return Counter(trigrams).most_common(top_k)


def compare_authorship(text4_trigrams, candidates_trigrams):
    """
    Compares the trigrams of Text 4 with Texts 1, 2, and 3 to determine authorship
    based on the highest number of overlapping trigrams.
    """
    text4_set = set([trigram for trigram, count in text4_trigrams])
    
    best_match = None
    max_overlap = 0
    
    print("\n--- Authorship Analysis (Trigram Overlap with Text 4) ---")
    for name, candidate_trigrams in candidates_trigrams.items():
        candidate_set = set([trigram for trigram, count in candidate_trigrams])
        overlap = len(text4_set.intersection(candidate_set))
        print(f"Overlap with {name}: {overlap} trigrams")
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = name
            
    if max_overlap > 0:
        print(f"\nConclusion: Based on n-gram analysis, {best_match} is the most likely author of Text 4.")
    else:
        print("\nConclusion: No significant trigram overlap found. Authorship cannot be conclusively determined.")

if __name__ == "__main__":
    download_nltk_resources()
    
    files = ['Text_1.txt', 'Text_2.txt', 'Text_3.txt', 'Text_4.txt']
    analyzers = {}
    
    # 1. Analyze Text 1, 2, and 3 for Subject and Entities
    for file in files[:3]:
        print(f"\n{'='*40}\nAnalyzing {file}\n{'='*40}")
        analyzer = TextAnalyzer(file)
        analyzers[file] = analyzer
        
        # Get top tokens
        top_tokens = analyzer.get_top_tokens(20)
        print("Top 20 Tokens:")
        print(top_tokens)
        
        # Get Named Entities
        ne_count, top_nes = analyzer.get_named_entities()
        print(f"\nTotal Named Entities found: {ne_count}")
        print("Most common Named Entities:")
        print(top_nes)
        
        print("\n* Subject Deduction: Look at the top tokens and entities above to determine the text's subject.")

    # 2. Authorship Analysis using Trigrams (Including Text 4)
    print(f"\n{'='*40}\nExtracting Trigrams for Authorship Analysis\n{'='*40}")
    
    # Load and process Text 4
    analyzer_4 = TextAnalyzer('Text_4.txt')
    text4_trigrams = analyzer_4.get_trigrams(n=3, top_k=50) # Get top 50 for a broader comparison base
    
    candidate_trigrams = {}
    for file in files[:3]:
        candidate_trigrams[file] = analyzers[file].get_trigrams(n=3, top_k=50)
        
    compare_authorship(text4_trigrams, candidate_trigrams)