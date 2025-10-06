import os
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords as nltk_stopwords

# -----------------------------
# Load Brysbaert dictionary
# -----------------------------
def load_brysbaert_dictionary(dict_path=None):
    """
    Loads the Brysbaert concreteness dictionary.

    If dict_path is None, automatically finds it in the package's
    linguistic_dictionaries folder relative to this script.
    """
    if dict_path is None:
        # Locate the linguistic_dictionaries folder relative to this file
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "linguistic_dictionaries"))
        dict_path = os.path.join(base_dir, "ac_brysbaert_dict.csv")

    brys_df = pd.read_csv(dict_path, usecols=["Word", "Conc.M"])
    return dict(zip(brys_df["Word"].str.lower().astype(str), brys_df["Conc.M"]))

# -----------------------------
# Get stopwords if needed
# -----------------------------
def get_stopwords(remove_stopwords=False):
    if remove_stopwords:
        nltk.download("stopwords", quiet=True)
        return set(nltk_stopwords.words("english"))
    return set()

# -----------------------------
# Score one tokenized caption
# -----------------------------
def concreteness_score_tokens(tokens, brys_dict, stop_words=set()):
    if not tokens:
        return np.nan

    tokens_lower = [t.lower() for t in tokens]
    if stop_words:
        tokens_lower = [t for t in tokens_lower if t not in stop_words]

    scores = [brys_dict[t] for t in tokens_lower if t in brys_dict]
    return np.mean(scores) if scores else np.nan

# -----------------------------
# Apply scoring to Series of tokenized captions
# -----------------------------
def classify_abstract_concrete(token_captions: pd.Series, remove_stopwords: bool = False) -> pd.Series:
    brys_dict = load_brysbaert_dictionary()
    stop_words = get_stopwords(remove_stopwords)

    return pd.Series(
        [concreteness_score_tokens(tokens, brys_dict, stop_words) for tokens in token_captions],
        index=token_captions.index
    )

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    from nltk.tokenize import word_tokenize

    # Load test CSV (update path if needed)
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "text_data_TEST.csv"))

    # Ensure nltk punkt tokenizer is available
    nltk.download("punkt", quiet=True)
    df["token_captions"] = df["caption"].fillna("").apply(word_tokenize)

    # Compute abstract/concrete scores
    df["scores"] = classify_abstract_concrete(df["token_captions"], remove_stopwords=True)

    # Preview results
    print(df[["caption", "scores"]].head(10))

