import numpy as np

from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import onnxruntime


class Onnx():

    def __init__(self, model="GPTCache/paraphrase-albert-onnx"):
        save_directory = "/workspace/my_tokenizer_directory"
        self.tokenizer = AutoTokenizer.from_pretrained(save_directory)

        self.model = model
        onnx_model_path = "/workspace/model.onnx"

        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)
        self.__dimension = 768

    def to_embeddings(self, data, **_):
        encoded_text = self.tokenizer.encode_plus(data, padding="max_length")

        ort_inputs = {
            "input_ids": np.array(encoded_text["input_ids"]).astype("int64").reshape(1, -1),
            "attention_mask": np.array(encoded_text["attention_mask"]).astype("int64").reshape(1, -1),
            "token_type_ids": np.array(encoded_text["token_type_ids"]).astype("int64").reshape(1, -1),
        }

        ort_outputs = self.ort_session.run(None, ort_inputs)
        ort_feat = ort_outputs[0]
        emb = self.post_proc(ort_feat, ort_inputs["attention_mask"])
        return emb.flatten()

    def post_proc(self, token_embeddings, attention_mask):
        input_mask_expanded = (
            np.expand_dims(attention_mask, -1)
            .repeat(token_embeddings.shape[-1], -1)
            .astype(float)
        )
        sentence_embs = np.sum(token_embeddings * input_mask_expanded, 1) / np.maximum(
            input_mask_expanded.sum(1), 1e-9
        )
        return sentence_embs

    @property
    def dimension(self):
        return self.__dimension

if __name__ == "__main__":
    model = Onnx()
    res = model.to_embeddings("hello world")
    print(res)



