import torch
from transformers import T5Tokenizer, MT5ForConditionalGeneration
import pandas as pd
from tqdm import tqdm

# Load once globally to avoid reloading for every function call
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = T5Tokenizer.from_pretrained('laihuiyuan/MMFLD', legacy = True)
MODEL = MT5ForConditionalGeneration.from_pretrained('laihuiyuan/MMFLD', use_safetensors=True).to(DEVICE)

def classify_figures_of_speech(captions, tasks=None, batch_size=4):
    if tasks is None:
        tasks = ['Idiom', 'Hyperbole', 'Metaphor']
    
    # Ensure list format
    if isinstance(captions, pd.Series):
        captions = captions.tolist()
    
    prompt_template = 'Which figure of speech does this text contain? (A) Literal. (B) {}. | Text: {}'
    all_results = {"Caption": captions}

    # Loop over each task once 
    for task in tasks:
        binary_preds_all = []
        
        for i in tqdm(range(0, len(captions), batch_size), desc=f"Processing {task}"):
            batch = captions[i:i+batch_size]
            prompts = [prompt_template.format(task, cap) for cap in batch]
            
            inputs = TOKENIZER(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            outputs = MODEL.generate(**inputs)
            preds = [TOKENIZER.decode(o, skip_special_tokens=True).strip().lower() for o in outputs]

            binary_preds = [1 if task.lower() in p else 0 for p in preds]
            binary_preds_all.extend(binary_preds)
        
        all_results[task] = binary_preds_all
    
    return pd.DataFrame(all_results)

if __name__ == "__main__":
    df = pd.DataFrame({
        'captions': [
            "This is a perfect way to break the ice and start the conversation.",
            "He’s a ticking time bomb ready to explode.",
            "The data speaks for itself.",
            "She’s walking on sunshine after the news.",
            "They are burning the midnight oil again.",
        ]
    })

    binary_df = classify_figures_of_speech(df['captions'])
    print(binary_df)
