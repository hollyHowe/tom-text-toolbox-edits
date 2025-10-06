import os
import pandas as pd
import re

def avg_emo_scores(caption, emo_dicts):
    """
    caption: list of words (tokens)
    emo_dicts: dict of {emotion_name: {word: score}}

    Returns:
    dict of {emotion_name: average_score}
    """
    result = {}
    for emo_name, emo_dict in emo_dicts.items():
        scores = [emo_dict[word] for word in caption if word in emo_dict]
        result[emo_name] = sum(scores) / len(scores) if scores else float("nan")
    return result

def classify_nrc_dict(captions: list[str] | pd.Series) -> pd.DataFrame:
    """
    Calculate average joy and anger scores from EmoLex for a list of captions.
    """
    # Get the directory where the linguistic_dictionaries folder is located
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "linguistic_dictionaries"))

    joy_path = os.path.join(base_dir, "joy-NRC-EmoIntv1-withZeroIntensityEntries.txt")
    anger_path = os.path.join(base_dir, "anger-NRC-EmoIntv1-withZeroIntensityEntries.txt")

    # Load the emotion dictionaries
    joy = pd.read_csv(joy_path, sep="\t")
    anger = pd.read_csv(anger_path, sep="\t")

    joy_dict = dict(zip(joy["English Word"].astype(str).str.lower(), joy["Emotion-Intensity-Score"]))
    anger_dict = dict(zip(anger["English Word"].astype(str).str.lower(), anger["Emotion-Intensity-Score"]))

    emotion_dicts = {
        "joy": joy_dict,
        "anger": anger_dict
    }

    # Process each caption
    results = []
    for caption in captions:
        tokens = re.findall(r"\b[a-zA-Z]+\b", caption.lower())
        emo_scores = avg_emo_scores(tokens, emotion_dicts)
        results.append(emo_scores)

    return pd.DataFrame(results)


if __name__ == "__main__":
    captions = [
        "I love my joyous and wonderful life",
        "I like bananas and chocolate milk a lot.",
        "He is always so happy ecstatic lovely and kind."
    ]

    df = classify_nrc_dict(captions)
    print(df)
