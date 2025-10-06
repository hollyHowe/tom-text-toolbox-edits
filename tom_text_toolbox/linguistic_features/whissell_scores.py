import pandas as pd
from tqdm import tqdm
import re
from typing import Union
from pathlib import Path
import tom_text_toolbox as ttt  # needed to find the package folder

def classify_whissell_scores(
    captions: pd.Series, 
    dictionary: Union[str, pd.DataFrame, None] = None
) -> pd.DataFrame:
    """
    Calculate Whissell scores for a series of captions.

    Parameters:
        captions (pd.Series): Series of caption strings or tokenized captions (list of words).
        dictionary (str or pd.DataFrame or None): Whissell dictionary with 'pleas', 'activ', 'image' columns, 
                                                  indexed by 'word'. If None, uses the default dictionary
                                                  inside tom_text_toolbox package.

    Returns:
        pd.DataFrame: DataFrame with columns:
            - 'whissell_pleasant'
            - 'whissell_active'
            - 'whissell_image'
    """
    # Automatically locate default dictionary if none provided
    if dictionary is None:
        dictionary = Path(ttt.__file__).parent / "linguistic_dictionaries" / "whissell_dict.csv"

    if isinstance(dictionary, str) or isinstance(dictionary, Path):
        dictionary = pd.read_csv(dictionary)

    if dictionary.index.name != 'word':
        dictionary = dictionary.set_index('word')

    pleasant = []
    active = []
    image = []

    for caption in tqdm(captions, desc="Scoring captions"):
        # If caption is string, tokenize
        if isinstance(caption, str):
            words = re.findall(r"\b[a-zA-Z]+\b", caption.lower())
        else:
            words = [str(w).lower() for w in caption]

        matched_words = [w for w in words if w in dictionary.index]

        if matched_words:
            scores = dictionary.loc[matched_words]
            means = scores[["pleas", "activ", "image"]].mean()
            pleasant.append(means["pleas"])
            active.append(means["activ"])
            image.append(means["image"])
        else:
            pleasant.append(0.0)
            active.append(0.0)
            image.append(0.0)

    return pd.DataFrame({
        "whissell_pleasant": pd.Series(pleasant, index=captions.index),
        "whissell_active": pd.Series(active, index=captions.index),
        "whissell_image": pd.Series(image, index=captions.index),
    })


if __name__ == "__main__":
    df = pd.DataFrame({
        "caption": ["This is a test.", "Another sentence!", "Third one."]
    })

    # Add Whissell scores
    whissell_df = classify_whissell_scores(df["caption"])

    # Combine with original captions
    final_df = pd.concat([df, whissell_df], axis=1)
    print(final_df.head())

