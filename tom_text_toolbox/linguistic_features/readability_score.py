import pandas as pd
import readability
from tqdm import tqdm


READABILITY_METHOD = "Kincaid"
KINCAID_KEY = "Kincaid"


def _check_readability_package():
    """
    Make sure we imported the correct `readability` package.

    This code needs the package named `readability`, not `readability-lxml`.
    """
    if not hasattr(readability, "getmeasures"):
        raise ImportError(
            "Wrong readability package is installed. "
            "This code needs the PyPI package named `readability`, not `readability-lxml`.\n\n"
            "Run this inside your conda environment:\n"
            "python -m pip uninstall -y readability-lxml readability\n"
            "python -m pip install readability"
        )


def parse_readability_measures(measures: dict, method: str = READABILITY_METHOD) -> dict:
    """
    Extracts the specified readability score from the 'readability grades' section.
    """
    grades = measures.get("readability grades", {})

    if method == READABILITY_METHOD and KINCAID_KEY in grades:
        return {f"readability_{KINCAID_KEY.lower()}": grades[KINCAID_KEY]}

    return {f"readability_{KINCAID_KEY.lower()}": pd.NA}


def get_readability_safe(text):
    """
    Safely computes readability measures for a single text.
    Returns NA if computation fails.
    """
    empty_result = {f"readability_{KINCAID_KEY.lower()}": pd.NA}

    if pd.isna(text):
        return empty_result

    text = str(text).strip()

    if text == "":
        return empty_result

    try:
        measures = readability.getmeasures(text)
        return parse_readability_measures(measures)

    except ValueError:
        return empty_result


def readability_scores(captions):
    """
    Computes readability scores for a pandas Series of captions/text.
    Returns a DataFrame aligned with the original Series index.
    """
    _check_readability_package()

    tqdm.pandas(desc="Computing readability")

    results = captions.progress_apply(get_readability_safe)

    readability_scores_df = pd.DataFrame(
        results.tolist(),
        index=captions.index
    )

    skipped = readability_scores_df[f"readability_{KINCAID_KEY.lower()}"].isna().sum()
    success_rate = ((len(readability_scores_df) - skipped) / len(readability_scores_df)) * 100 if len(readability_scores_df) else 0

    print(f"Skipped {skipped} rows, success rate: {success_rate:.2f}%")

    return readability_scores_df
