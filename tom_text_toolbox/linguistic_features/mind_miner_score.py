import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple, Union

MODEL_NAME = "j-hartmann/MindMiner"

# 🧠 Load model & tokenizer ONCE at import time (not inside the function)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Model max length info cached here
try:
    MODEL_MAX_LEN = int(getattr(model.config, "max_position_embeddings", tokenizer.model_max_length))
except Exception:
    MODEL_MAX_LEN = tokenizer.model_max_length

try:
    SPECIAL_TOKENS = tokenizer.num_special_tokens_to_add(pair=False)
except Exception:
    SPECIAL_TOKENS = 2  # fallback

CHUNK_LIMIT = max(1, MODEL_MAX_LEN - SPECIAL_TOKENS)


def classify_mind_miner(captions: Union[pd.Series, List]) -> pd.Series:
    """
    Tokenizer-aware chunking without truncation.
    Computes a weighted average confidence score for each caption.
    """

    def process_word_list_to_chunks(words: List[str]) -> List[List[str]]:
        """Build chunks so that subword length + specials <= model limit."""
        chunks = []
        current = []
        current_len = 0

        for w in words:
            try:
                w_len = len(tokenizer(w, add_special_tokens=False)["input_ids"])
            except Exception:
                w_len = len(tokenizer(str(w), add_special_tokens=False)["input_ids"])

            if current_len + w_len > CHUNK_LIMIT:
                if current:
                    chunks.append(current)
                    current = [w]
                    current_len = w_len
                else:
                    chunks.append([w])
                    current = []
                    current_len = 0
            else:
                current.append(w)
                current_len += w_len

        if current:
            chunks.append(current)

        return chunks

    def score_chunk(words_chunk: List[str]) -> Tuple[float, int]:
        """Score one chunk recursively if too long."""
        if not words_chunk:
            return (0.0, 0)

        chunk_text = " ".join(words_chunk)
        encoded = tokenizer(chunk_text, add_special_tokens=True, return_tensors="pt")
        input_ids = encoded["input_ids"]
        seq_len = input_ids.shape[1]

        if seq_len <= MODEL_MAX_LEN:
            input_ids = input_ids.to(device)
            attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = F.softmax(outputs.logits, dim=-1)
                score = float(probs.max(dim=-1).values.item())
            weight = len(words_chunk)
            return (score * weight, weight)

        # if too long still, split recursively
        if len(words_chunk) == 1:
            # emergency char splitting
            word = words_chunk[0]
            char_chunk_size = max(1, CHUNK_LIMIT // 4)
            sub_chunks = [word[i:i + char_chunk_size] for i in range(0, len(word), char_chunk_size)]
            total_sum = 0.0
            total_w = 0
            for sc in sub_chunks:
                s_sum, s_w = score_chunk([sc])
                total_sum += s_sum
                total_w += s_w
            return (total_sum, total_w)
        else:
            mid = len(words_chunk) // 2
            left_sum, left_w = score_chunk(words_chunk[:mid])
            right_sum, right_w = score_chunk(words_chunk[mid:])
            return (left_sum + right_sum, left_w + right_w)

    results = []
    for caption in captions:
        if pd.isna(caption):
            results.append(float("nan"))
            continue

        # normalize input to list of words
        if isinstance(caption, str):
            words = caption.split()
        elif isinstance(caption, (list, tuple)):
            words = [str(w) for w in caption]
        else:
            words = [str(caption)]

        if len(words) == 0:
            results.append(float("nan"))
            continue

        chunks = process_word_list_to_chunks(words)
        total_sum = 0.0
        total_weight = 0

        for chunk_words in chunks:
            try:
                s_sum, s_weight = score_chunk(chunk_words)
                total_sum += s_sum
                total_weight += s_weight
            except Exception:
                # fallback: score word by word
                for w in chunk_words:
                    try:
                        s_sum, s_weight = score_chunk([w])
                        total_sum += s_sum
                        total_weight += s_weight
                    except Exception:
                        continue

        if total_weight == 0:
            results.append(float("nan"))
        else:
            results.append(total_sum / total_weight)

    return pd.Series(results)


if __name__ == "__main__":
    # Smoke test
    df = pd.DataFrame({
        "caption": [
            ["This", "product", "is", "amazing", "!"],
            ["I'm", "not", "sure", "how", "I", "feel", "about", "this", "."],
            ["Worst", "experience", "ever", "."],
            ["A"] * 1200
        ]
    })

    out = classify_mind_miner(df["caption"])
    print(out)

