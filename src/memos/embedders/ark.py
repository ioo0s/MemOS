from volcenginesdkarkruntime import Ark
from volcenginesdkarkruntime.types.multimodal_embedding import (
    EmbeddingInputParam,
    MultimodalEmbeddingContentPartImageParam,
    MultimodalEmbeddingContentPartTextParam,
    MultimodalEmbeddingResponse,
)
from volcenginesdkarkruntime.types.multimodal_embedding.embedding_content_part_image_param import (
    ImageURL,
)

from memos.configs.embedder import ArkEmbedderConfig
from memos.embedders.base import BaseEmbedder
from memos.log import get_logger


logger = get_logger(__name__)


class ArkEmbedder(BaseEmbedder):
    """Arl Embedder class."""

    def __init__(self, config: ArkEmbedderConfig):
        self.config = config

        if self.config.embedding_dims is not None:
            logger.warning(
                "Ark does not support specifying embedding dimensions. "
                "The embedding dimensions is determined by the model."
                "`embedding_dims` will be set to None."
            )
            self.config.embedding_dims = None

        # Default model if not specified
        if not self.config.model_name_or_path:
            self.config.model_name_or_path = "doubao-embedding-vision-250615"

        # Initialize ollama client
        self.client = Ark(api_key=self.config.api_key, base_url=self.config.api_base)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for the given texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embeddings, each represented as a list of floats.
        """
        texts_input = [
            MultimodalEmbeddingContentPartTextParam(text=text, type="text") for text in texts
        ]
        return self.multimodal_embeddings(texts_input, chunk_size=self.config.chunk_size)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_images(self, urls: list[str], chunk_size: int | None = None) -> list[list[float]]:
        chunk_size_ = chunk_size or self.config.chunk_size
        images_input = [
            MultimodalEmbeddingContentPartImageParam(image_url=ImageURL(url=url), type="image_url")
            for url in urls
        ]
        return self.multimodal_embeddings(images_input, chunk_size=chunk_size_)

    def multimodal_embeddings(
        self, inputs: list[EmbeddingInputParam], chunk_size: int | None = None
    ) -> list[list[float]]:
        chunk_size_ = chunk_size or self.config.chunk_size
        embeddings: list[list[float]] = []

        for i in range(0, len(inputs), chunk_size_):
            response: MultimodalEmbeddingResponse = self.client.multimodal_embeddings.create(
                model=self.config.model_name_or_path,
                input=inputs[i : i + chunk_size_],
            )

            data = [response.data] if isinstance(response.data, dict) else response.data
            embeddings.extend(r["embedding"] for r in data)

        return embeddings
