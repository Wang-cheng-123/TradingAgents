import os
import requests
import chromadb
from chromadb.config import Settings

try:
    # 仅在需要 OpenAI 兼容端点时导入
    from openai import OpenAI
except Exception:
    OpenAI = None


class FinancialSituationMemory:
    """
    记忆/相似检索模块，支持两类嵌入后端：
      1) OpenAI 兼容:   <base>/v1/embeddings
      2) Ollama 原生:   <base>/api/embeddings
    通过 config 中的 embedding_backend_url / embedding_model / embedding_api_key 控制。
    """

    def __init__(self, name, config):
        self.config = config or {}

        # ---- 嵌入后端的配置（你在 config 里已提供） ----
        self.embedding_backend_url = self.config.get("embedding_backend_url")
        self.embedding_model = self.config.get("embedding_model", "text-embedding-3-small")
        # 优先使用 embedding_api_key；否则退回到全局 OPENAI_API_KEY / api_key（如果存在）
        self.embedding_api_key = (
            self.config.get("embedding_api_key")
            or os.getenv("EMBEDDING_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or self.config.get("api_key")
            or ""
        )

        # ---- 自动识别后端类型 ----
        self._embedding_mode = self._detect_embedding_mode(self.embedding_backend_url)

        # ---- 初始化嵌入客户端 ----
        self.embedding_client = None
        if self._embedding_mode == "openai":
            if OpenAI is None:
                raise RuntimeError(
                    "OpenAI SDK 未安装，但 embedding_backend_url 指向 /v1。请 `pip install openai` "
                    "或改用 Ollama 原生 /api/embeddings。"
                )
            # 注意：这里用 embedding_backend_url，不再复用对话的 backend_url
            base_url = self._normalize_openai_base(self.embedding_backend_url)
            self.embedding_client = OpenAI(base_url=base_url, api_key=self.embedding_api_key)

        # ---- 启动本地向量库 ----
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        # 用 get_or_create 避免重复创建抛错
        self.situation_collection = self.chroma_client.get_or_create_collection(name=name)

    # -------------------------- 工具方法 --------------------------

    @staticmethod
    def _detect_embedding_mode(url: str) -> str:
        """
        简单判断使用 OpenAI 兼容还是 Ollama 原生：
        - 包含 '/v1' → 视为 openai 兼容
        - 包含 '/api/embeddings' 或以 '/api' 结尾 → 视为 ollama 原生
        """
        if not url:
            # 没给就默认 openai（与项目原逻辑兼容），但更建议明确配置
            return "openai"
        u = url.rstrip("/")
        if "/v1" in u:
            return "openai"
        if u.endswith("/api") or "/api/embeddings" in u:
            return "ollama"
        # 默认按 openai 处理
        return "openai"

    @staticmethod
    def _normalize_openai_base(url: str) -> str:
        """
        规范化 OpenAI 兼容端点的 base_url：
        - 允许传 'https://xxx/v1' 或 'https://xxx'
        - 返回必须以 '/v1' 结尾，满足 OpenAI SDK 预期
        """
        if not url:
            return "https://api.openai.com/v1"
        u = url.rstrip("/")
        if not u.endswith("/v1"):
            u = u + "/v1"
        return u

    def _post_ollama_embeddings(self, text: str):
        """
        调用 Ollama 原生 /api/embeddings
        文档格式：POST /api/embeddings  { "model": "...", "prompt": "..." }
        响应常见: { "embedding": [ ... ], "model": "..." }
        """
        url = self.embedding_backend_url.rstrip("/")
        # 允许用户给到主机根 '/api'，这里自动补成 '/api/embeddings'
        if url.endswith("/api"):
            url = url + "/embeddings"

        headers = {}
        if self.embedding_api_key:
            headers["Authorization"] = f"Bearer {self.embedding_api_key}"

        payload = {"model": self.embedding_model, "prompt": text}
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # 优先取原生字段
        if "embedding" in data:
            return data["embedding"]

        # 兼容某些代理返回 OpenAI 风格
        if "data" in data and data["data"]:
            first = data["data"][0]
            if "embedding" in first:
                return first["embedding"]

        raise RuntimeError(f"Unexpected embeddings response schema from {url}: {data}")

    # -------------------------- 对外方法 --------------------------

    def get_embedding(self, text: str):
        """返回文本嵌入向量（使用配置的后端）。"""
        if self._embedding_mode == "openai":
            # OpenAI 兼容：/v1/embeddings
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            # openai>=1.0 的返回
            return response.data[0].embedding

        # Ollama 原生：/api/embeddings
        return self._post_ollama_embeddings(text)

    def add_situations(self, situations_and_advice):
        """
        添加若干 (situation, recommendation) 对。
        """
        situations, advice, ids, embeddings = [], [], [], []
        offset = self.situation_collection.count()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))
            embeddings.append(self.get_embedding(situation))

        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in advice],
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation, n_matches=1):
        """
        用嵌入做相似检索，返回最匹配的若干条建议。
        """
        query_embedding = self.get_embedding(current_situation)

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_matches,
            include=["metadatas", "documents", "distances"],
        )

        matched_results = []
        for i in range(len(results["documents"][0])):
            matched_results.append(
                {
                    "matched_situation": results["documents"][0][i],
                    "recommendation": results["metadatas"][0][i]["recommendation"],
                    "similarity_score": 1 - results["distances"][0][i],
                }
            )
        return matched_results


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory()

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors 
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")

