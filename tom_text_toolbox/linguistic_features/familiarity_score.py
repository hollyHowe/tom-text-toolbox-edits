import os
import pandas as pd
from typing import Optional, Union, List
from tqdm import tqdm

# -----------------------------
# Load familiarity dictionary
# -----------------------------
def load_familiarity_dict(dict_path: str = None) -> dict:
    """
    Loads the familiarity dictionary.

    If dict_path is None, automatically finds it in the package's
    linguistic_dictionaries folder relative to this script.
    """
    if dict_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "linguistic_dictionaries"))
        dict_path = os.path.join(base_dir, "fam_peatzold_dict.csv")

    df = pd.read_csv(dict_path)
    df = df[df["Word"].apply(lambda x: isinstance(x, str))]
    return dict(zip(df["Word"].str.lower(), df["Familiarity"]))

# -----------------------------
# Score a single caption
# -----------------------------
def score_caption(tokens_or_text: Union[str, List[str]], fam_dict: dict) -> Optional[float]:
    # Handle NaNs / empty values
    if tokens_or_text is None or (isinstance(tokens_or_text, float) and pd.isna(tokens_or_text)):
        return None
    
    # If it's already tokenized
    if isinstance(tokens_or_text, list):
        tokens = [str(t).lower() for t in tokens_or_text]
    else:
        # Assume it's a string
        tokens = str(tokens_or_text).lower().split()
    
    if not tokens:
        return None
    
    # Compute familiarity score
    scores = [fam_dict[token] for token in tokens if token in fam_dict]
    return round(sum(scores) / len(scores), 2) if scores else None

# -----------------------------
# Apply scoring to a pandas Series
# -----------------------------
def classify_familiarity(captions: pd.Series, show_progress: bool = False) -> pd.Series:
    fam_dict = load_familiarity_dict()
    if show_progress:
        tqdm.pandas(desc="Calculating familiarity scores")
        return captions.progress_apply(lambda x: score_caption(x, fam_dict))
    else:
        return captions.apply(lambda x: score_caption(x, fam_dict))

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    # Load test CSV relative to this script
    test_csv_path = os.path.join(os.path.dirname(__file__), "text_data_TEST.csv")
    df = pd.read_csv(test_csv_path)
    
    # This will now work whether "caption" is tokenized (list) or a raw string
    df["familiarity_score"] = classify_familiarity(df["caption"])
    print(df[["caption", "familiarity_score"]].head(10))
