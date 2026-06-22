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
for package in [
    "punkt",
    "punkt_tab",
    "stopwords",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "wordnet",
    "omw-1.4"
]:
    try:
        nltk.download(package, quiet=True)
    except Exception:
        pass

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer


class TermCounter:
    def __init__(
        self,
        term_dict: Dict[str, List[str]],
        preprocessed_categories: Optional[List[str]] = None,
        change_dict: Optional[Dict[str, str]] = None
    ):
        """
        Initialize TermCounter with a dictionary of term categories.

        By default, these categories are preprocessed before counting:
        - nostalgia_terms
        - reg_promotion
        - reg_prevention
        """

        if not isinstance(term_dict, dict):
            raise ValueError("term_dict must be a dictionary.")

        if not all(isinstance(v, list) for v in term_dict.values()):
            raise ValueError("Each value in term_dict must be a list of terms.")

        self.term_dict = term_dict

        self.patterns = {
            name: self.build_pattern(terms)
            for name, terms in term_dict.items()
        }

        self.preprocessed_categories = set(
            cat.lower()
            for cat in (
                preprocessed_categories
                or ["nostalgia_terms", "reg_promotion", "reg_prevention"]
            )
        )

        # Optional spelling fixes.
        # Add more entries here if needed.
        default_change_dict = {
            "freind": "friend",
            "freinds": "friend"
        }

        if change_dict:
            default_change_dict.update(change_dict)

        self.change_dict = {
            str(k).lower(): str(v).lower()
            for k, v in default_change_dict.items()
        }

        self.wnl = WordNetLemmatizer()


    @classmethod
    def from_json(
        cls,
        json_path: str = "linguistic_dictionaries/term_dict.json",
        preprocessed_categories: Optional[List[str]] = None,
        change_dict: Optional[Dict[str, str]] = None
    ):
        """
        Load TermCounter from a JSON file.
        """

        if not os.path.isabs(json_path):
            try:
                base_dir = os.path.join(os.path.dirname(__file__), "..")
            except NameError:
                base_dir = os.getcwd()

            json_path = os.path.abspath(os.path.join(base_dir, json_path))

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Cannot find term dictionary file at: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            term_dict = json.load(f)

        if not isinstance(term_dict, dict):
            raise ValueError("JSON must contain a dictionary at the top level.")

        if not all(isinstance(v, list) for v in term_dict.values()):
            raise ValueError("Each value in the JSON must be a list of terms.")

        return cls(
            term_dict=term_dict,
            preprocessed_categories=preprocessed_categories,
            change_dict=change_dict
        )


    def build_pattern(self, terms: List[str]) -> re.Pattern:
        """
        Compile a regex pattern for a list of terms.

        Supports '*' wildcard.
        Example:
            remember* matches remember, remembers, remembered, remembering
        """

        if not terms:
            return re.compile(r"a^", re.IGNORECASE)

        pattern_parts = [
            rf"\b{re.escape(term[:-1])}\w*"
            if term.endswith("*")
            else rf"\b{re.escape(term)}\b"
            for term in terms
        ]

        return re.compile(
            rf"(?:{'|'.join(pattern_parts)})",
            re.IGNORECASE
        )


    @staticmethod
    def get_wordnet_pos(tag: str):
        """
        Convert NLTK POS tags to WordNet POS tags.
        """

        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("N"):
            return wordnet.NOUN
        elif tag.startswith("R"):
            return wordnet.ADV
        else:
            return None


    def preprocess_for_dictionary(self, captions: pd.Series) -> pd.Series:
        """
        Preprocess captions before applying selected dictionaries.

        Does:
        1. Lowercase text.
        2. Fix selected misspellings.
        3. Lemmatize verbs to base form.
           Example: felt -> feel, achieved -> achieve, escaping -> escape
        4. Lemmatize nouns to singular form.
           Example: memories -> memory
        """

        captions = captions.fillna("").astype(str)

        def process_one_caption(text: str) -> str:
            tokens = word_tokenize(str(text))

            tokens = [token.lower() for token in tokens]

            # Fix spelling before POS tagging
            tokens = [
                self.change_dict.get(token, token)
                for token in tokens
            ]

            tagged_tokens = pos_tag(tokens)

            lemmas = []

            for word, tag in tagged_tokens:
                wordnet_pos = self.get_wordnet_pos(tag) or wordnet.NOUN
                lemma = self.wnl.lemmatize(word, pos=wordnet_pos)

                # Fix spelling again after lemmatizing, just in case
                lemma = self.change_dict.get(lemma, lemma)

                lemmas.append(lemma)

            return " ".join(lemmas)

        return captions.apply(process_one_caption)


    def count_terms(self, captions: pd.Series, category: str) -> pd.Series:
        """
        Count term matches for one category.

        If category is in self.preprocessed_categories, count on preprocessed text.
        Otherwise, count on original text.

        Also fixes the case-sensitivity issue.
        """

        if category not in self.patterns:
            raise ValueError(f"Category '{category}' not found in term_dict.")

        captions = captions.fillna("").astype(str)

        if category.lower() in self.preprocessed_categories:
            captions_to_count = self.preprocess_for_dictionary(captions)
        else:
            captions_to_count = captions

        pattern = self.patterns[category]

        return captions_to_count.str.count(
            pattern.pattern,
            flags=pattern.flags
        )


    @staticmethod
    def exclamation_count(captions: pd.Series) -> pd.Series:
        captions = captions.fillna("").astype(str)
        return captions.str.count(r"!")


    @staticmethod
    def question_count(captions: pd.Series) -> pd.Series:
        captions = captions.fillna("").astype(str)
        return captions.str.count(r"\?")


    @staticmethod
    def hashtag_count(captions: pd.Series) -> pd.Series:
        captions = captions.fillna("").astype(str)
        return captions.str.count(r"#\S+")


    @staticmethod
    def mention_count(captions: pd.Series) -> pd.Series:
        captions = captions.fillna("").astype(str)
        return captions.str.count(r"@\w+")


    @staticmethod
    def caption_length(captions: pd.Series) -> pd.Series:
        captions = captions.fillna("").astype(str)
        return captions.str.len()


    def type_token_ratio(self, captions: pd.Series, segment_size: int = 5) -> pd.Series:
        """
        Calculate segmental type-token ratio for each caption.
        """

        captions = captions.fillna("").astype(str)

        def calculate_segmental_ttr(text: str) -> Optional[float]:
            words = str(text).lower().split()

            if not words:
                return None

            segments = [
                words[i:i + segment_size]
                for i in range(0, len(words), segment_size)
            ]

            ttrs = [
                len(set(segment)) / len(segment)
                for segment in segments
                if len(segment) > 0
            ]

            if not ttrs:
                return None

            return round(float(np.mean(ttrs)), 3)

        return captions.apply(calculate_segmental_ttr)


    @staticmethod
    def alliteration_count(captions: pd.Series) -> pd.Series:
        """
        Count adjacent alliterative content words per caption.
        """

        captions = captions.fillna("").astype(str)
        stop_words = set(stopwords.words("english"))

        def count_alliteration(text):
            tokens = [
                w.lower().strip(string.punctuation)
                for w in word_tokenize(str(text))
                if w.isalpha()
            ]

            content_words = [
                w for w in tokens
                if w and w not in stop_words
            ]

            count = 0

            for i in range(len(content_words) - 1):
                if content_words[i][0] == content_words[i + 1][0]:
                    count += 1

            return count

        return captions.apply(count_alliteration)


    @staticmethod
    def repetition_count(captions: pd.Series) -> pd.Series:
        """
        Count repeated words per caption.

        This counts repeat uses beyond the first use.
        Example:
            "love love music music music" = 3 repeats
        """

        captions = captions.fillna("").astype(str)

        def count_repetition(text):
            tokens = [
                w.lower().strip(string.punctuation)
                for w in word_tokenize(str(text))
                if w.isalpha()
            ]

            counts = Counter(tokens)

            return sum(v - 1 for v in counts.values() if v > 1)

        return captions.apply(count_repetition)


    def count_all(self, captions: pd.Series) -> pd.DataFrame:
        """
        Count all dictionary categories and additional text features.

        Important:
        - Categories in self.preprocessed_categories are counted on
          lemmatized/preprocessed captions.
        - All other dictionary categories are counted on original captions.
        - Regex matching is case-insensitive.
        """

        captions = captions.fillna("").astype(str)

        df_counts = pd.DataFrame(index=captions.index)

        preprocessed_captions = None

        for category, pattern in self.patterns.items():

            if category.lower() in self.preprocessed_categories:

                # Only preprocess once, even if there are multiple preprocessed categories.
                if preprocessed_captions is None:
                    preprocessed_captions = self.preprocess_for_dictionary(captions)

                captions_to_count = preprocessed_captions

            else:
                captions_to_count = captions

            df_counts[category] = captions_to_count.str.count(
                pattern.pattern,
                flags=pattern.flags
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

    term_counts_df = tc.count_all(df["caption"])

    df = pd.concat([df, term_counts_df], axis=1)

    print(df[["caption", "alliteration_count", "repetition_count"]].head())
