# Full, copy-pasteable implementation
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple, Union

MODEL_NAME = "j-hartmann/MindMiner"

def classify_mind_miner(captions: Union[pd.Series, List]) -> pd.Series:
    """
    Tokenizer-aware chunking (no truncation). Accepts captions that are either:
      - lists of tokens (e.g., ['This', 'is', 'a', ...]) OR
      - strings (will be split on whitespace into word tokens).
    Behavior:
      - Uses tokenizer to measure subword length of words and builds chunks so
        that when special tokens are added the model input length <= model_max.
      - Runs model on each chunk (no truncation) and gets a confidence score
        for the top label (like the HF pipeline's 'score').
      - Returns a weighted average score per caption where weights = original token counts.
    """
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # Determine model limits and number of special tokens
    try:
        model_max = int(getattr(model.config, "max_position_embeddings", tokenizer.model_max_length))
    except Exception:
        model_max = tokenizer.model_max_length
    try:
        n_special = tokenizer.num_special_tokens_to_add(pair=False)
    except Exception:
        n_special = 2  # conservative fallback
    # available space for input_ids BEFORE adding special tokens
    chunk_input_id_limit = max(1, model_max - n_special)

    def process_word_list_to_chunks(words: List[str]) -> List[List[str]]:
        """
        Build chunks of original words such that tokenizer(word-chunk, add_special_tokens=False)
        input_ids length <= chunk_input_id_limit. This uses tokenizer(word, add_special_tokens=False)
        per word to account for subword expansion.
        """
        chunks: List[List[str]] = []
        current: List[str] = []
        current_ids_len = 0

        for w in words:
            # get subword length for this word
            try:
                wid = tokenizer(w, add_special_tokens=False)["input_ids"]
            except Exception:
                # fallback: treat word as a single token (rare)
                wid = tokenizer(str(w), add_special_tokens=False)["input_ids"]
            wlen = len(wid)

            # If single word alone is larger than limit, we still create a chunk with it.
            # It will be handled by recursive splitter later.
            if current_ids_len + wlen > chunk_input_id_limit:
                if current:
                    chunks.append(current)
                    current = [w]
                    current_ids_len = wlen
                else:
                    # current is empty but word itself exceeds limit - create single-word chunk
                    chunks.append([w])
                    current = []
                    current_ids_len = 0
            else:
                current.append(w)
                current_ids_len += wlen

        if current:
            chunks.append(current)

        return chunks

    def score_chunk_words(words_chunk: List[str]) -> Tuple[float, int]:
        """
        Score a chunk represented as a list of original words.
        Returns (weighted_sum, n_original_tokens) where weighted_sum = score * n_original_tokens.
        This function ensures the encoded length with special tokens <= model_max by recursively
        splitting a chunk if necessary.
        """
        if not words_chunk:
            return (0.0, 0)

        chunk_text = " ".join(words_chunk)

        # Tokenize with special tokens to get final length
        encoded = tokenizer(chunk_text, add_special_tokens=True, return_tensors="pt")
        input_ids = encoded["input_ids"]
        seq_len = input_ids.shape[1]

        # If safe length, run model and return score * weight
        if seq_len <= model_max:
            input_ids = input_ids.to(device)
            attention_mask = encoded["attention_mask"].to(device) if "attention_mask" in encoded else torch.ones_like(input_ids).to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # outputs.logits shape: (1, num_labels)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                score = float(probs.max(dim=-1).values.item())  # match pipeline score behavior
            weight = len(words_chunk)  # weighting by original token count (words)
            return (score * weight, weight)
        else:
            # If too long, split into halves and recurse (no truncation)
            if len(words_chunk) == 1:
                # Extremely rare: single "word" token still produces > model_max tokens.
                # As last resort, break the word into characters or subparts to ensure progress;
                # we will split the word into character-based micro-tokens to allow processing.
                word = words_chunk[0]
                # split into chunks of characters which tokenizer will subword-tokenize
                # we choose a safe character chunk size (heuristic)
                char_chunk_size = max(1, chunk_input_id_limit // 4)
                sub_chunks = [word[i:i+char_chunk_size] for i in range(0, len(word), char_chunk_size)]
                total_sum = 0.0
                total_w = 0
                for sc in sub_chunks:
                    s_sum, s_w = score_chunk_words([sc])
                    total_sum += s_sum
                    total_w += s_w
                return (total_sum, total_w)
            else:
                mid = len(words_chunk) // 2
                left_sum, left_w = score_chunk_words(words_chunk[:mid])
                right_sum, right_w = score_chunk_words(words_chunk[mid:])
                return (left_sum + right_sum, left_w + right_w)

    results = []

    for idx, caption in enumerate(captions):
        # normalize input into list of words (original tokens)
        if pd.isna(caption):
            results.append(float("nan"))
            continue
        if isinstance(caption, str):
            words = caption.split()
        elif isinstance(caption, (list, tuple)):
            # ensure strings
            words = [str(w) for w in caption]
        else:
            words = [str(caption)]

        if len(words) == 0:
            results.append(float("nan"))
            continue

        # Build initial chunks using tokenizer-aware estimation
        chunks_by_words = process_word_list_to_chunks(words)

        total_weighted_sum = 0.0
        total_weight = 0

        for chunk_words in chunks_by_words:
            try:
                w_sum, w_count = score_chunk_words(chunk_words)
                total_weighted_sum += w_sum
                total_weight += w_count
            except Exception as e:
                # If something unexpected happens, try safe fallback: score words one-by-one
                # (still no truncation — slow fallback)
                # NOTE: this is extremely unlikely, but keeps behavior robust
                for w in chunk_words:
                    try:
                        s_sum, s_count = score_chunk_words([w])
                        total_weighted_sum += s_sum
                        total_weight += s_count
                    except Exception:
                        # give up on this tiny piece and skip it
                        continue

        if total_weight == 0:
            results.append(float("nan"))
        else:
            weighted_avg = total_weighted_sum / total_weight
            results.append(weighted_avg)

    return pd.Series(results)


if __name__ == "__main__":
    # Quick smoke test
    df = pd.DataFrame({
        "caption": [
            ["This", "product", "is", "amazing", "!"],
            ["I'm", "not", "sure", "how", "I", "feel", "about", "this", "."],
            ["Worst", "experience", "ever", "."],
            ["A"] * 1200  # very long tokenized caption
        ]
    })

    out = classify_mind_miner(df["caption"])
    print(out)


