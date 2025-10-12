import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm  # progress bar

def chunk_text(text: str, max_tokens: int = 400) -> list:
    """Split a text string into chunks of up to max_tokens words (approximate tokens)."""
    words = text.split()
    return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

def classify_mind_miner(captions: pd.Series | list, max_tokens_per_chunk: int = 400, batch_size: int = 8):
    """
    Analyze captions using MindMiner, safely handling very long captions with batching, weighted average, and progress bar.
    
    Parameters:
        captions (pd.Series | list): Series or list of caption strings.
        max_tokens_per_chunk (int): Maximum number of tokens per chunk.
        batch_size (int): Number of chunks to process at once in the pipeline.
        
    Returns:
        pd.Series: Series of weighted-average scores for each caption.
    """
    
    model_name = "j-hartmann/MindMiner"
    device = 0 if torch.cuda.is_available() else -1
    mindminer = pipeline(model=model_name, function_to_apply="none", device=device)

    print(f"{'Using GPU' if device == 0 else 'Using CPU'} for inference.")

    all_scores = []

    # Loop over captions with progress bar
    for caption in tqdm(captions, desc="Processing captions", unit="caption"):
        chunks = chunk_text(caption, max_tokens=max_tokens_per_chunk)
        chunk_scores = []

        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_results = mindminer(batch)

            # Safely extract scores from batch
            chunk_scores.extend([res.get('score', 0.0) for res in batch_results])

        # Weighted average by chunk length
        chunk_lengths = [len(chunk.split()) for chunk in chunks]
        total_length = sum(chunk_lengths)
        weighted_avg = sum(score * length for score, length in zip(chunk_scores, chunk_lengths)) / total_length

        all_scores.append(weighted_avg)

    return pd.Series(all_scores)

# ------------------ Usage Example ------------------
if __name__ == "__main__":
    df = pd.DataFrame({
        "caption": [
            "This product is amazing! " * 100,  # very long caption
            "I'm not sure how I feel about this.",
            "Worst experience ever."
        ]
    }) 

    results = classify_mind_miner(df["caption"])
    print(results)

