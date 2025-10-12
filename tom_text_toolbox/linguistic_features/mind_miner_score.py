import pandas as pd
from transformers import pipeline, AutoTokenizer
import torch
from tqdm import tqdm

def chunk_text_with_token_lengths(text: str, tokenizer, max_tokens: int = 300):
    """
    Split text into chunks and return both the chunk text and its token length.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    chunk_texts = [tokenizer.decode(chunk, clean_up_tokenization_spaces=True) for chunk in chunks]
    chunk_lengths = [len(chunk) for chunk in chunks]
    return chunk_texts, chunk_lengths

def classify_mind_miner(captions: pd.Series | list, max_tokens_per_chunk: int = 300, batch_size: int = 8):
    model_name = "j-hartmann/MindMiner"
    device = 0 if torch.cuda.is_available() else -1

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mindminer = pipeline(model=model_name, function_to_apply="none", device=device)

    print(f"{'Using GPU' if device == 0 else 'Using CPU'} for inference.")

    all_scores = []

    for caption in tqdm(captions, desc="Processing captions", unit="caption"):
        try:
            chunks, chunk_lengths = chunk_text_with_token_lengths(caption, tokenizer, max_tokens=max_tokens_per_chunk)
            chunk_scores = []

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_results = mindminer(batch)
                chunk_scores.extend([res.get('score', 0.0) for res in batch_results])

            total_length = sum(chunk_lengths)
            weighted_avg = sum(score * length for score, length in zip(chunk_scores, chunk_lengths)) / total_length
            all_scores.append(weighted_avg)

        except Exception as e:
            # Print the caption that caused the error
            print(f"\n❌ Failed on caption:\n{caption}\nError: {e}")
            # Append a placeholder score
            all_scores.append(0.0)

    return pd.Series(all_scores)
