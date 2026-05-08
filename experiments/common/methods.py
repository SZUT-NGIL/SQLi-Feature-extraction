from __future__ import annotations

import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

from experiments.common.bootstrap import ROOT_DIR, bootstrap_repo_paths
from experiments.common.metrics import scores_from_model, to_probabilities


bootstrap_repo_paths()

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
OURS_MODEL_DIR = ROOT_DIR / "ours" / "model" / "1"
FE_DIR = ROOT_DIR / "fe"
NUM_COLS = [
    "qlen",
    "wcount",
    "sq",
    "dq",
    "puncts",
    "comments",
    "spaces",
    "logic",
    "arith",
    "alpha",
    "sqlkw",
    "sqlfunc",
]


def _legacy_tfidf_tokenizer(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower().strip())


def _install_legacy_tfidf_tokenizer() -> None:
    for module_name in ("__main__", "__mp_main__"):
        module_obj = sys.modules.get(module_name)
        if module_obj is not None:
            setattr(module_obj, "tokenizer", _legacy_tfidf_tokenizer)


@dataclass
class MethodStatus:
    method_id: str
    display_name: str
    notes: str = ""
    artifact_files: list[Path] = field(default_factory=list)


class BaseMethod:
    method_id = "base"
    display_name = "Base"
    default_threshold = 0.5

    def __init__(self) -> None:
        self.model: Any | None = None
        self.threshold = self.default_threshold
        self.status = MethodStatus(method_id=self.method_id, display_name=self.display_name)

    def mandatory_paths(self) -> Iterable[Path]:
        return []

    def load(self) -> MethodStatus:
        missing = [path for path in self.mandatory_paths() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{self.display_name} artifacts missing: {missing}")
        self._load_impl()
        return self.status

    def _load_impl(self) -> None:
        raise NotImplementedError

    def featurize(self, texts: list[str]) -> Any:
        raise NotImplementedError

    def predict_scores(self, features: Any) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self.display_name} must call load() before predict.")
        return scores_from_model(self.model, features)

    def total_model_size_mb(self) -> float:
        total_bytes = 0
        for path in self.status.artifact_files:
            if path.is_file():
                total_bytes += path.stat().st_size
            elif path.is_dir():
                total_bytes += sum(
                    child.stat().st_size for child in path.rglob("*") if child.is_file()
                )
        return total_bytes / (1024 * 1024)

    @staticmethod
    def _normalize_texts(texts: list[str]) -> list[str]:
        return [str(text).lower().strip() for text in texts]


class OursMethod(BaseMethod):
    method_id = "ours"
    display_name = "Ours"

    def mandatory_paths(self) -> Iterable[Path]:
        return [
            OURS_MODEL_DIR / "scaler_for_numeric.pkl",
            OURS_MODEL_DIR / "numeric_features" / "model_XGBoost.pkl",
            OURS_MODEL_DIR / "numeric_features" / "best_threshold.pkl",
        ]

    def _load_impl(self) -> None:
        from utils.hfes import extract_struct_features_single

        self.extract_struct_features_single = extract_struct_features_single
        scaler_path = OURS_MODEL_DIR / "scaler_for_numeric.pkl"
        model_path = OURS_MODEL_DIR / "numeric_features" / "model_XGBoost.pkl"
        threshold_path = OURS_MODEL_DIR / "numeric_features" / "best_threshold.pkl"

        with open(scaler_path, "rb") as file_obj:
            self.scaler = pickle.load(file_obj)
        with open(threshold_path, "rb") as file_obj:
            self.threshold = float(pickle.load(file_obj))

        self.model = joblib.load(model_path)
        self.booster = self.model.get_booster()
        self.scaler_mean = np.asarray(
            getattr(self.scaler, "mean_", np.zeros(len(NUM_COLS))),
            dtype=np.float64,
        )
        self.scaler_scale = np.asarray(
            getattr(self.scaler, "scale_", np.ones(len(NUM_COLS))),
            dtype=np.float64,
        )
        self.status.artifact_files = [scaler_path, model_path, threshold_path]
        self.status.notes = "12D handcrafted features + XGBoost"

    def featurize(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, len(NUM_COLS)), dtype=np.float64)

        # 单样本特征直通，并手动完成标准化，减少 sklearn/pandas 包装层开销。
        feature_rows = [
            self.extract_struct_features_single("" if text is None else str(text))
            for text in texts
        ]
        feature_array = np.asarray(feature_rows, dtype=np.float64)
        return (feature_array - self.scaler_mean) / self.scaler_scale

    def predict_scores(self, features: Any) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self.display_name} must call load() before predict.")
        scores = self.booster.inplace_predict(features, validate_features=False)
        return to_probabilities(scores)


class BowMethod(BaseMethod):
    method_id = "bow"
    display_name = "BoW"

    def mandatory_paths(self) -> Iterable[Path]:
        model_dir = FE_DIR / "bow" / "model" / "1"
        return [model_dir / "bow_vectorizer.pkl", model_dir / "model_XGB.pkl"]

    def _load_impl(self) -> None:
        model_dir = FE_DIR / "bow" / "model" / "1"
        vec_path = model_dir / "bow_vectorizer.pkl"
        model_path = model_dir / "model_XGB.pkl"
        self.vectorizer = joblib.load(vec_path)
        self.model = joblib.load(model_path)
        self.status.artifact_files = [vec_path, model_path]
        self.status.notes = "CountVectorizer + XGBoost"

    def featurize(self, texts: list[str]) -> Any:
        return self.vectorizer.transform(self._normalize_texts(texts))


class TfidfMethod(BaseMethod):
    method_id = "tfidf"
    display_name = "TF-IDF"

    def mandatory_paths(self) -> Iterable[Path]:
        model_dir = FE_DIR / "tfidf" / "model" / "1"
        return [model_dir / "tfidf_vectorizer.pkl", model_dir / "model_XGB.pkl"]

    def _load_impl(self) -> None:
        model_dir = FE_DIR / "tfidf" / "model" / "1"
        vec_path = model_dir / "tfidf_vectorizer.pkl"
        model_path = model_dir / "model_XGB.pkl"
        _install_legacy_tfidf_tokenizer()
        with open(vec_path, "rb") as file_obj:
            self.vectorizer = pickle.load(file_obj)
        self.model = joblib.load(model_path)
        self.status.artifact_files = [vec_path, model_path]
        self.status.notes = "TF-IDF + XGBoost"

    def featurize(self, texts: list[str]) -> Any:
        return self.vectorizer.transform(self._normalize_texts(texts))


class W2VMethod(BaseMethod):
    method_id = "w2v"
    display_name = "Word2Vec"

    def mandatory_paths(self) -> Iterable[Path]:
        model_dir = FE_DIR / "w2v" / "model" / "1"
        return [model_dir / "word2vec.model", model_dir / "model_XGB.pkl"]

    def _load_impl(self) -> None:
        from gensim.models import Word2Vec

        model_dir = FE_DIR / "w2v" / "model" / "1"
        w2v_path = model_dir / "word2vec.model"
        model_path = model_dir / "model_XGB.pkl"
        self.embedding_model = Word2Vec.load(str(w2v_path))
        self.model = joblib.load(model_path)
        self.status.artifact_files = [w2v_path, model_path]
        self.status.notes = "Word2Vec sentence average + XGBoost"

    def featurize(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            tokens = TOKEN_RE.findall(str(text).lower().strip())
            if not tokens:
                vectors.append(np.zeros(self.embedding_model.vector_size, dtype=np.float32))
                continue
            embeds = [self.embedding_model.wv[token] for token in tokens if token in self.embedding_model.wv]
            if embeds:
                vectors.append(np.mean(embeds, axis=0).astype(np.float32))
            else:
                vectors.append(np.zeros(self.embedding_model.vector_size, dtype=np.float32))
        return np.vstack(vectors)


class FastTextMethod(BaseMethod):
    method_id = "fasttext"
    display_name = "FastText"

    def mandatory_paths(self) -> Iterable[Path]:
        model_dir = FE_DIR / "fasttext" / "model" / "1"
        return [
            model_dir / "fasttext.model",
            model_dir / "fasttext.model.wv.vectors_ngrams.npy",
            model_dir / "model_XGB.pkl",
        ]

    def _load_impl(self) -> None:
        from gensim.models import FastText

        model_dir = FE_DIR / "fasttext" / "model" / "1"
        ft_path = model_dir / "fasttext.model"
        model_path = model_dir / "model_XGB.pkl"
        self.embedding_model = FastText.load(str(ft_path))
        self.model = joblib.load(model_path)
        self.status.artifact_files = [ft_path, model_path]
        self.status.notes = "FastText sentence average + XGBoost"

    def featurize(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            tokens = TOKEN_RE.findall(str(text).lower().strip())
            if not tokens:
                vectors.append(np.zeros(self.embedding_model.vector_size, dtype=np.float32))
                continue
            embeds = [self.embedding_model.wv[token] for token in tokens if token in self.embedding_model.wv]
            if embeds:
                vectors.append(np.mean(embeds, axis=0).astype(np.float32))
            else:
                vectors.append(np.zeros(self.embedding_model.vector_size, dtype=np.float32))
        return np.vstack(vectors)


class BertMethod(BaseMethod):
    method_id = "bert"
    display_name = "BERT"

    def __init__(self) -> None:
        super().__init__()
        self._backbone_size_mb = 0.0

    def mandatory_paths(self) -> Iterable[Path]:
        return [FE_DIR / "bert" / "model" / "1" / "model_XGB.pkl"]

    def _load_impl(self) -> None:
        import torch
        from transformers import BertModel, BertTokenizer

        model_dir = FE_DIR / "bert" / "model" / "1"
        xgb_path = model_dir / "model_XGB.pkl"
        domain_dir = model_dir / "bert_domain"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = 64
        self.batch_size = 32
        self.model = joblib.load(xgb_path)

        if domain_dir.exists() and any(domain_dir.iterdir()):
            self.tokenizer = BertTokenizer.from_pretrained(str(domain_dir), local_files_only=True)
            self.bert_model = BertModel.from_pretrained(str(domain_dir), local_files_only=True)
            self.status.artifact_files = [xgb_path, domain_dir]
            self.status.notes = "Domain BERT + XGBoost"
        else:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
            self.bert_model = BertModel.from_pretrained("bert-base-uncased", local_files_only=True)
            self.status.artifact_files = [xgb_path]
            self.status.notes = "bert-base-uncased(cache) + XGBoost"

        self.bert_model.to(self.device)
        self.bert_model.eval()
        self._backbone_size_mb = sum(
            parameter.nelement() * parameter.element_size()
            for parameter in self.bert_model.parameters()
        ) / (1024 * 1024)

    def total_model_size_mb(self) -> float:
        return super().total_model_size_mb() + self._backbone_size_mb

    def featurize(self, texts: list[str]) -> np.ndarray:
        import torch

        all_embeddings = []
        normalized = self._normalize_texts(texts)
        for start in range(0, len(normalized), self.batch_size):
            batch_texts = normalized[start : start + self.batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu().numpy())
        return np.vstack(all_embeddings).astype(np.float32)


METHOD_FACTORIES = {
    "ours": OursMethod,
    "bow": BowMethod,
    "tfidf": TfidfMethod,
    "w2v": W2VMethod,
    "fasttext": FastTextMethod,
    "bert": BertMethod,
}


def build_methods(method_ids: list[str]) -> list[BaseMethod]:
    methods = []
    for method_id in method_ids:
        if method_id not in METHOD_FACTORIES:
            raise ValueError(f"Unknown method: {method_id}. Choices: {sorted(METHOD_FACTORIES)}")
        methods.append(METHOD_FACTORIES[method_id]())
    return methods
