# Import libraries
import pandas as pd
from transformers import pipeline
import torch

# Define max tokens (typical for most transformer models like RoBERTa)
MAX_TOKENS = 512

def chunk_tokens(tokens: list[str], chunk_size: int = MAX_TOKENS) -> list[str]:
    """
    Split a list of tokens into chunks of up to `chunk_size` tokens,
    then join each chunk into a string for model inference.
    """
    return [" ".join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]


def classify_mind_miner(captions: pd.Series | list) -> pd.Series:
    """
    Analyze tokenized captions using MindMiner with token-level chunking.
    
    Parameters:
        captions (pd.Series | list): Series or list of token lists.
    Returns:          
        pd.Series: List of average scores for each caption.
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
        # Ensure we're working with a list of tokens
        if isinstance(token_list, str):
            token_list = token_list.split()
        elif not isinstance(token_list, list):
            token_list = [str(token_list)]

        # Chunk the token list
        chunks = chunk_tokens(token_list, chunk_size=MAX_TOKENS)

        # Run inference on each chunk
        chunk_scores = []
        for chunk in chunks:
            output = mindminer(chunk)
            chunk_scores.append(output[0]['score'])

        # Average score for the caption
        avg_score = sum(chunk_scores) / len(chunk_scores)
        results.append(avg_score)

    return pd.Series(results)


if __name__ == "__main__":
    # Example tokenized captions
    df = pd.DataFrame({
        "caption": [
            ["This", "product", "is", "amazing", "!"],
            ["I'm", "not", "sure", "how", "I", "feel", "about", "this", "."],
            ["Worst", "experience", "ever", "."],
            ["A"] * 1200  # long tokenized caption
        ]
    })

    # Run MindMiner
    results = classify_mind_miner(df["caption"])

    # Print results
    print(results)


