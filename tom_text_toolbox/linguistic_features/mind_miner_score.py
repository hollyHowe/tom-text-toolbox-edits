# Import libraries
import pandas as pd
from transformers import pipeline
import torch

# Define max chunk size (safe below model's max of 512)
MAX_TOKENS = 480

def chunk_tokens(tokens: list[str], chunk_size: int = MAX_TOKENS) -> list[list[str]]:
    """
    Split a list of tokens into chunks of up to `chunk_size` tokens.
    Returns a list of token chunks (each a list of strings).
    """
    return [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]


def classify_mind_miner(captions: pd.Series | list) -> pd.Series:
    """
    Analyze tokenized captions using MindMiner with token-level chunking (no truncation),
    and compute a weighted average score based on chunk length.
    """
    model_name = "j-hartmann/MindMiner"
    mindminer = pipeline(
        model=model_name,
        function_to_apply="none",
        device=0 if torch.cuda.is_available() else -1
    )

    # Check if the GPU is available
    if mindminer.device == 0:
        print("⚡ Using GPU for inference.")
    else:
        print("🐢 Using CPU for inference. This may be slower.")

    results = []
    for token_list in captions:
        # Normalize input
        if isinstance(token_list, str):
            token_list = token_list.split()
        elif not isinstance(token_list, list):
            token_list = [str(token_list)]

        if len(token_list) == 0:
            results.append(float('nan'))
            continue

        # Split into chunks (as lists of tokens)
        chunks = chunk_tokens(token_list, chunk_size=MAX_TOKENS)

        total_tokens = 0
        weighted_sum = 0.0

        for chunk_tokens_list in chunks:
            chunk_text = " ".join(chunk_tokens_list)
            try:
                output = mindminer(chunk_text)
                score = output[0]['score']
                weight = len(chunk_tokens_list)
                weighted_sum += score * weight
                total_tokens += weight
            except Exception as e:
                print(f"❌ Error processing chunk (length {len(chunk_tokens_list)}): {e}")
                continue

        if total_tokens == 0:
            results.append(float('nan'))
        else:
            weighted_avg = weighted_sum / total_tokens
            results.append(weighted_avg)

    return pd.Series(results)


if __name__ == "__main__":
    # Example tokenized captions
    df = pd.DataFrame({
        "caption": [
            ["This", "product", "is", "amazing", "!"],
            ["I'm", "not", "sure", "how", "I", "feel", "about", "this", "."],
            ["Worst", "experience", "ever", "."],
            ["A"] * 1200  # long tokenized caption to test chunking
        ]
    })

    results = classify_mind_miner(df["caption"])
    print(results)

