import json
import numpy as np
from tensorflow import keras
from utils import normalize_text, encode_sentence
from custom_layers import ScaledDotProductAttention


LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


class SentimentModel:
    def __init__(self, model_path="bilstm_att_correct_mask_final.keras", vocab_path="vocab.json"):
        print("→ Loading vocab...")
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        print("→ Loading model...")
        self.model = keras.models.load_model(
            model_path,
            custom_objects={"ScaledDotProductAttention": ScaledDotProductAttention},
            compile=False
        )

        self.max_len = self.model.input_shape[1]
        print("✓ Model loaded successfully.")

    def predict(self, text: str):
        cleaned = normalize_text(text)
        toks, seq = encode_sentence(cleaned, self.vocab, self.max_len)

        preds = self.model.predict(seq)[0]     # (20,3)

        sentence_pred = preds.mean(axis=0)      # (3,)

        pred_id = int(np.argmax(sentence_pred))
        pred_label = LABEL_MAP[pred_id]

        return {
            "input_text": text,
            "cleaned_text": cleaned,
            "prediction_label": pred_label,
            "prediction_raw": sentence_pred.tolist(),
        }


sentiment_model = SentimentModel()
