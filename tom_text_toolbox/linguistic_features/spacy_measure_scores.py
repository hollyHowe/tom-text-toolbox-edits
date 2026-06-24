import spacy
from spacy.symbols import NOUN, VERB, ADJ, ADV
import pandas as pd
import json
import os

class SpacyAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")

    def score_spacy_measures(self, captions: pd.Series) -> pd.DataFrame:
        docs = list(self.nlp.pipe(captions.astype(str), batch_size=2000, n_process=4))

        informativeness, boastful, syntax_complexity, tense_data = [], [], [], []
        
        for doc in docs:
            alpha_tokens = [t for t in doc if t.is_alpha]
            n_tokens = len(alpha_tokens)

            # ---- Informativeness ----
            content_count = doc.count_by(spacy.attrs.POS)
            content_tokens = content_count.get(NOUN, 0) + content_count.get(VERB, 0) + \
                             content_count.get(ADJ, 0) + content_count.get(ADV, 0)
            n_words = sum(1 for token in doc if token.is_alpha)
            informativeness.append(round(content_tokens / n_words, 3) if n_words else 0.0)

            # ---- Syntax complexity ----
            dep_counts = doc.count_by(spacy.attrs.DEP)
            num_clauses = sum(dep_counts.get(doc.vocab.strings[dep], 0) for dep in ["ccomp", "advcl", "acl", "relcl"])
            max_depth = max((len(list(t.ancestors)) for t in alpha_tokens), default=0)
            num_subtrees = sum(1 for t in alpha_tokens if len(list(t.children)) > 1)
            syntax_complexity.append(round(num_clauses * 1.5 + max_depth * 1.2 + num_subtrees * 1.0, 2))

            # ---- Verb tenses ----
            counts = {"Past": 0, "Present": 0}
            for t in alpha_tokens:
                if t.pos_ == "VERB" and "VerbForm=Fin" in t.morph:
                    if "Tense=Past" in t.morph:
                        counts["Past"] += 1
                    elif "Tense=Pres" in t.morph:
                        counts["Present"] += 1
            tense_data.append(counts)

            # ---- Boastful Language ----
            boast_count = sum(1 for token in doc if token.tag_ in ["JJS", "RBS"])
            boastful.append(boast_count)

        # Build final DataFrame
        df = pd.DataFrame({
            "informativeness": informativeness,
            "boastful_language": boastful,
            "syntax_complexity": syntax_complexity,
            "tense_past": [t["Past"] for t in tense_data],
            "tense_present": [t["Present"] for t in tense_data]
        }, index=captions.index)

        return df
