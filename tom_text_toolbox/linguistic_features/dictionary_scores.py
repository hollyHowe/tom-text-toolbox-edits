import pandas as pd
import numpy as np
import re
import json
import os
import string
import nltk
from collections import Counter
from typing import List, Dict, Optional

# Download resources for tokenization / lemmatization
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

class TermCounter:
    def __init__(self, term_dict: Dict[str, List[str]]):
        """Initialize TermCounter with a dictionary of term categories."""
        if not isinstance(term_dict, dict):
            raise ValueError("term_dict must be a dictionary.")
        if not all(isinstance(v, list) for v in term_dict.values()):
            raise ValueError("Each value in term_dict must be a list of terms.")
        self.term_dict = term_dict
        self.patterns = {name: self.build_pattern(terms) for name, terms in term_dict.items()}

    @classmethod
    def from_json(cls, json_path: str = "linguistic_dictionaries/term_dict.json"):
        """Load TermCounter from a JSON file."""
        if not os.path.isabs(json_path):
            base_dir = os.path.join(os.path.dirname(__file__), "..")
            json_path = os.path.abspath(os.path.join(base_dir, json_path))
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Cannot find term dictionary file at: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            term_dict = json.load(f)

        if not isinstance(term_dict, dict):
            raise ValueError("JSON must contain a dictionary at the top level.")
        if not all(isinstance(v, list) for v in term_dict.values()):
            raise ValueError("Each value in the JSON must be a list of terms.")

        return cls(term_dict)

    def build_pattern(self, terms: List[str]) -> re.Pattern:
        """Compile a regex pattern for a list of terms (supports '*' wildcard)."""
        pattern_parts = [
            rf"\b{re.escape(term[:-1])}\w*" if term.endswith("*") else rf"\b{re.escape(term)}\b"
            for term in terms
        ]
        return re.compile(rf"(?:{'|'.join(pattern_parts)})", re.IGNORECASE)

    def count_terms(self, captions: pd.Series, category: str) -> pd.Series:
        """
        Count term matches for a specific category.

        Preserves case-insensitive matching from the compiled regex pattern.
        """
        if category not in self.patterns:
            raise ValueError(f"Category '{category}' not found in term_dict.")

        captions = captions.fillna("").astype(str)
        pattern = self.patterns[category]

        return captions.str.count(pattern.pattern, flags=pattern.flags)

    @staticmethod
    def exclamation_count(captions: pd.Series) -> pd.Series:
        return captions.str.count(r'!')

    @staticmethod
    def question_count(captions: pd.Series) -> pd.Series:
        return captions.str.count(r'\?')

    @staticmethod
    def hashtag_count(captions: pd.Series) -> pd.Series:
        return captions.str.count(r'#\S+')

    @staticmethod
    def mention_count(captions: pd.Series) -> pd.Series:
        return captions.str.count(r'@\w+')

    @staticmethod
    def caption_length(captions: pd.Series) -> pd.Series:
        return captions.str.len()

    def type_token_ratio(self, captions: pd.Series, segment_size: int = 5) -> pd.Series:
        """Calculate segmental type-token ratio (TTR) for each caption."""
        def calculate_segmental_ttr(text: str) -> Optional[float]:
            words = str(text).lower().split()
            if not words:
                return None
            segments = [words[i:i + segment_size] for i in range(0, len(words), segment_size)]
            ttrs = [len(set(seg)) / len(seg) for seg in segments]
            return round(float(np.mean(ttrs)), 3)
        return captions.apply(calculate_segmental_ttr)

    # -------------------------------------------------------
    # New Features: Alliteration & Repetition
    # -------------------------------------------------------
    @staticmethod
    def alliteration_count(captions: pd.Series) -> pd.Series:
        """Count occurrences of alliteration per caption."""
        stop_words = set(stopwords.words("english"))

        def count_alliteration(text):
            tokens = [
                w.lower().strip(string.punctuation)
                for w in word_tokenize(str(text))
                if w.isalpha()
            ]
            content_words = [w for w in tokens if w not in stop_words]
            count = 0
            for i in range(len(content_words) - 1):
                if content_words[i][0] == content_words[i + 1][0]:
                    count += 1
            return count

        return captions.apply(count_alliteration)

    @staticmethod
    def repetition_count(captions: pd.Series) -> pd.Series:
        """Count repeated words per caption."""
        def count_repetition(text):
            tokens = [
                w.lower().strip(string.punctuation)
                for w in word_tokenize(str(text))
                if w.isalpha()
            ]
            counts = Counter(tokens)
            return sum(v - 1 for v in counts.values() if v > 1)

        return captions.apply(count_repetition)

    # -------------------------------------------------------
    # Main Counting Function (Extended)
    # -------------------------------------------------------
    def count_all(self, captions: pd.Series) -> pd.DataFrame:
        """
        Count matches for all dictionary categories and include additional text features.

        This version preserves case-insensitive matching.
        """

        captions = captions.fillna("").astype(str)

        df_counts = pd.DataFrame(
            {
                cat: captions.str.count(pat.pattern, flags=pat.flags)
                for cat, pat in self.patterns.items()
            },
            index=captions.index
        )

        df_counts["exclamation_count"] = self.exclamation_count(captions)
        df_counts["question_count"] = self.question_count(captions)
        df_counts["hashtag_count"] = self.hashtag_count(captions)
        df_counts["mention_count"] = self.mention_count(captions)
        df_counts["caption_length"] = self.caption_length(captions)
        df_counts["type_token_ratio"] = self.type_token_ratio(captions)
        df_counts["alliteration_count"] = self.alliteration_count(captions)
        df_counts["repetition_count"] = self.repetition_count(captions)

        return df_counts


if __name__ == "__main__":
    tc = TermCounter.from_json("tom_text_toolbox/dictionaries/term_dict.json")
    df = pd.read_csv("tom_text_toolbox/text_data_TEST.csv")

    # Count all term categories and linguistic features
    term_counts_df = tc.count_all(df["caption"])

    # Merge results with the original data
    df = pd.concat([df, term_counts_df], axis=1)

    print(df[["caption", "alliteration_count", "repetition_count"]].head())
