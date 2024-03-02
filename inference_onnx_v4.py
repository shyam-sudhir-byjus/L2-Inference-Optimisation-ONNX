import onnxruntime
from transformers import AutoTokenizer
import torch
from torch.nn import functional as F
import numpy as np
from common import ONNX_MODEL_PATH, TOKENIZER_PATH


class ONNXInferenceEngine:

    def __init__(
        self, model_path: str = ONNX_MODEL_PATH, tokenizer_path: str = TOKENIZER_PATH
    ):
        """
        Initialize the ONNX Inference Engine.

        Args:
        - model_path (str): Path to the ONNX model file.
        - tokenizer_path (str): Path to the tokenizer used by the model.
        """
        self.sess_options = onnxruntime.SessionOptions()
        self.session = onnxruntime.InferenceSession(
            model_path, self.sess_options, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, do_lower_case=True
        )

    def mean_pooling(
        self, model_output: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mean pooling of token embeddings.

        Args:
        - model_output: Model output containing token embeddings.
        - attention_mask: Attention mask for token embeddings.

        Returns:
        - Mean-pooled embeddings normalized along dimension 1.
        """
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        temp = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        return F.normalize(temp, p=2, dim=1)

    def compute_embeddings(self, data: list[str]) -> np.ndarray:
        """
        Compute embeddings for input data.

        Args:
        - data: Input data of sentences

        Returns:
        - Embeddings computed for the input data.
        """
        inputs = self.tokenizer(
            data, padding=True, truncation=True, return_tensors="pt"
        )

        ort_inputs = {k: v.cpu().numpy() for k, v in inputs.items()}

        op = self.session.run(None, ort_inputs)
        op = torch.from_numpy(op[0])
        similarity_matrix = (
            self.mean_pooling([op], inputs["attention_mask"]).cpu().detach().numpy()
        )
        return similarity_matrix

    def compute_similarity_many_to_many(self, sentences: list[str]) -> np.ndarray:
        """
        Compute similarity matrix for many-to-many comparisons.

        Args:
        - sentences: List of sentences for comparison.

        Returns:
        - Similarity matrix computed for the input sentences.
        """
        embeddings = self.compute_embeddings(sentences)
        similarity_matrix = np.dot(embeddings, embeddings.T)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarity_matrix /= norm
        similarity_matrix /= norm.T
        return similarity_matrix

    def compute_similarity_one_to_many(
        self, query: str, sentences: list[str]
    ) -> np.ndarray:
        """
        Compute similarity matrix for one-to-many comparisons.

        Args:
        - query: Input query text
        - sentences: List of sentences for comparison.

        Returns:
        - Similarity matrix computed for the input sentences.
        """
        all_sentences = [query] + sentences
        embeddings = self.compute_embeddings(all_sentences)
        query_embeddings = embeddings[0]
        sentences_embeddings = embeddings[1:]
        similarity_scores = np.dot(query_embeddings, sentences_embeddings.T)
        norm_query = np.linalg.norm(query_embeddings)
        norm_sentences = np.linalg.norm(sentences_embeddings, axis=1)
        similarity_scores /= norm_query
        similarity_scores /= norm_sentences
        return similarity_scores
