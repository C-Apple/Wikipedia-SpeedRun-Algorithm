from __future__ import annotations

import numpy as np


class EmbeddingProjector3D:
    """Project embedding vectors into a stable 3D PCA space."""

    def __init__(
        self,
        embedding_matrix: np.ndarray,
        *,
        sample_size: int = 10_000,
        seed: int = 42,
    ) -> None:
        if embedding_matrix.ndim != 2:
            raise ValueError(
                "embedding_matrix must have shape "
                "(vocabulary_size, embedding_dimension)"
            )

        rng = np.random.default_rng(seed)

        sample_count = min(
            sample_size,
            len(embedding_matrix),
        )

        sample_indices = rng.choice(
            len(embedding_matrix),
            size=sample_count,
            replace=False,
        )

        sample = embedding_matrix[
            sample_indices
        ].astype(
            np.float32,
            copy=True,
        )

        # Normalize because the model uses cosine similarity.
        norms = np.linalg.norm(
            sample,
            axis=1,
            keepdims=True,
        )

        norms[norms == 0] = 1.0
        sample /= norms

        self.mean = sample.mean(axis=0)

        centered = sample - self.mean

        # Principal component analysis using SVD.
        _, _, right_singular_vectors = np.linalg.svd(
            centered,
            full_matrices=False,
        )

        self.components = right_singular_vectors[:3]

        sample_coordinates = (
            centered @ self.components.T
        )

        distances = np.linalg.norm(
            sample_coordinates,
            axis=1,
        )

        self.scale = float(
            np.percentile(distances, 95)
        )

        if (
            not np.isfinite(self.scale)
            or self.scale <= 0
        ):
            self.scale = 1.0

    def project(
        self,
        vector: np.ndarray,
    ) -> list[float]:
        """Project one embedding vector into three dimensions."""

        vector = np.asarray(
            vector,
            dtype=np.float32,
        )

        if vector.ndim != 1:
            raise ValueError(
                "vector must be one-dimensional"
            )

        norm = np.linalg.norm(vector)

        if norm > 0:
            vector = vector / norm

        coordinates = (
            (vector - self.mean)
            @ self.components.T
        )

        coordinates = coordinates / self.scale

        return [
            float(coordinate)
            for coordinate in coordinates
        ]