import pandas as pd
from tqdm import tqdm
import re
from typing import Union

def classify_whissell_scores(
    captions: pd.Series, 
    dictionary: Union[str, pd.DataFrame] = "tom_text_toolbox/linguistic_dictionaries/whissell_dict.csv"
) -> pd.DataFrame:
    """
    Calculate Whissell scores for a series of captions.

    Parameters:
        captions (pd.Series): Series of caption strings or tokenized captions (list of words).
        dictionary (str or pd.DataFrame): Whissell dictionary with 'pleas', 'activ', 'image' columns, indexed by 'word'.

    Returns:
        pd.DataFrame: DataFrame with columns:
            - 'whissell_pleasant'
            - 'whissell_active'
            - 'whissell_image'
    """
    if isinstance(dictionary, str):
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
