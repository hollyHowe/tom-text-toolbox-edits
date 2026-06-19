import numpy as np
import pandas as pd
from emosent import get_emoji_sentiment_rank_multiple

def _get_text_series(data, column=None):
    """
    Convert input data into a pandas Series of text.

    Supports:
    - a DataFrame plus column name
    - a Series
    - a list/tuple/array of strings
    """
    if isinstance(data, pd.DataFrame):
        if column is None:
            if "caption" in data.columns:
                column = "caption"
            else:
                raise ValueError(
                    "You passed a DataFrame to classify_emosent(), but no column was specified. "
                    "Use classify_emosent(df, column='your_text_column')."
                )

        if column not in data.columns:
            raise ValueError(
                f"Column '{column}' not found in the DataFrame. "
                f"Available columns are: {list(data.columns)}"
            )

        return data[column]

    if isinstance(data, pd.Series):
        return data

    return pd.Series(data)


def score_emosent_text(text, empty_value=np.nan):
    """
    Calculate average EmoSent sentiment score for emojis in one text.

    Parameters
    ----------
    text : str
        Text containing zero or more emojis.

    empty_value : float, default np.nan
        Value returned when the text has no emojis recognized by EmoSent.

    Returns
    -------
    dict
        Dictionary containing average emoji sentiment and emoji counts.
    """
    if pd.isna(text):
        text = ""

    text = str(text)

    emoji_results = get_emoji_sentiment_rank_multiple(text)

    scores = []

    for item in emoji_results:
        rank = item.get("emoji_sentiment_rank")

        if isinstance(rank, dict) and "sentiment_score" in rank:
            score = rank["sentiment_score"]

            try:
                scores.append(float(score))
            except TypeError:
                continue
            except ValueError:
                continue

    if len(scores) == 0:
        avg_sentiment = empty_value
    else:
        avg_sentiment = float(np.mean(scores))

    return {
        "emosent_avg_sentiment": avg_sentiment,
        "emosent_emoji_count": len(emoji_results),
        "emosent_scored_emoji_count": len(scores),
    }


def classify_emosent(data, column=None, empty_value=np.nan):
    """
    Calculate average emoji sentiment for each row of text.

    Parameters
    ----------
    data : pd.DataFrame, pd.Series, list, tuple, or array
        Text data to score.

        Recommended use inside tom-text-toolbox:
            classify_emosent(df, column=column)

        Also works:
            classify_emosent(df["caption"])

    column : str, optional
        Name of the text column to score when data is a DataFrame.

    empty_value : float, default np.nan
        Value used when a row has no emojis recognized by EmoSent.
        Use 0.0 if you prefer no-emoji rows to be coded as neutral/zero.

    Returns
    -------
    pd.DataFrame
        DataFrame with:
        - emosent_avg_sentiment
        - emosent_emoji_count
        - emosent_scored_emoji_count
    """
    text_series = _get_text_series(data, column=column)

    results = [
        score_emosent_text(text, empty_value=empty_value)
        for text in text_series
    ]

    return pd.DataFrame(results, index=text_series.index)
