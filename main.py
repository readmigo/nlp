"""
spaCy NLP 服务
提供英文文本分词、词性标注、命名实体识别等功能
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import spacy
import logging
from sentence_transformers import SentenceTransformer
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="readmigo NLP Service",
    description="spaCy-based NLP service for bilingual reading",
    version="1.0.0",
)

# 加载 spaCy 模型
try:
    nlp = spacy.load("en_core_web_lg")
    logger.info("Loaded spaCy model: en_core_web_lg")
except OSError:
    logger.warning("en_core_web_lg not found, trying en_core_web_sm")
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded spaCy model: en_core_web_sm")
    except OSError:
        logger.error("No spaCy model found. Please install: python -m spacy download en_core_web_lg")
        nlp = None

# 加载多语言句子嵌入模型
try:
    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    logger.info("Loaded sentence embedding model: paraphrase-multilingual-MiniLM-L12-v2")
except Exception as e:
    logger.error(f"Failed to load sentence embedding model: {e}")
    embed_model = None


# ============================================================
# 请求/响应模型
# ============================================================


class TokenizeRequest(BaseModel):
    """分词请求"""
    texts: List[str]
    include_entities: bool = True


class TokenData(BaseModel):
    """单个 token 数据"""
    text: str
    start: int
    end: int
    lemma: str
    pos: str
    tag: str
    dep: str
    is_stop: bool
    is_punct: bool
    is_space: bool
    ent_type: Optional[str] = None


class EntityData(BaseModel):
    """命名实体数据"""
    text: str
    start: int
    end: int
    label: str


class TokenizedText(BaseModel):
    """分词结果"""
    original: str
    tokens: List[TokenData]
    entities: List[EntityData]


class TokenizeResponse(BaseModel):
    """分词响应"""
    results: List[TokenizedText]


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model: str
    version: str


class EmbedRequest(BaseModel):
    """嵌入请求"""
    texts: List[str]


class EmbedResponse(BaseModel):
    """嵌入响应"""
    embeddings: List[List[float]]
    dimension: int


# ============================================================
# API 端点
# ============================================================


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    if nlp is None:
        raise HTTPException(status_code=503, detail="NLP model not loaded")

    return HealthResponse(
        status="healthy",
        model=nlp.meta.get("name", "unknown"),
        version=nlp.meta.get("version", "unknown"),
    )


@app.post("/tokenize", response_model=TokenizeResponse)
async def tokenize(request: TokenizeRequest):
    """
    批量分词接口

    对每个文本进行分词，返回 tokens 和实体
    """
    if nlp is None:
        raise HTTPException(status_code=503, detail="NLP model not loaded")

    results = []

    for text in request.texts:
        doc = nlp(text)

        # 提取 tokens
        tokens = []
        for token in doc:
            token_data = TokenData(
                text=token.text,
                start=token.idx,
                end=token.idx + len(token.text),
                lemma=token.lemma_,
                pos=token.pos_,
                tag=token.tag_,
                dep=token.dep_,
                is_stop=token.is_stop,
                is_punct=token.is_punct,
                is_space=token.is_space,
                ent_type=token.ent_type_ if token.ent_type_ else None,
            )
            tokens.append(token_data)

        # 提取实体
        entities = []
        if request.include_entities:
            for ent in doc.ents:
                entity_data = EntityData(
                    text=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    label=ent.label_,
                )
                entities.append(entity_data)

        results.append(TokenizedText(
            original=text,
            tokens=tokens,
            entities=entities,
        ))

    return TokenizeResponse(results=results)


@app.post("/lemmatize")
async def lemmatize(texts: List[str]):
    """
    批量词元化

    返回每个词的原形
    """
    if nlp is None:
        raise HTTPException(status_code=503, detail="NLP model not loaded")

    results = []
    for text in texts:
        doc = nlp(text)
        lemmas = [token.lemma_ for token in doc if not token.is_space]
        results.append({"original": text, "lemmas": lemmas})

    return {"results": results}


@app.get("/stopwords")
async def get_stopwords():
    """获取停用词列表"""
    if nlp is None:
        raise HTTPException(status_code=503, detail="NLP model not loaded")

    # 返回前 100 个最常见的停用词
    stopwords = list(nlp.Defaults.stop_words)[:100]
    return {"stopwords": sorted(stopwords)}


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """
    批量句子嵌入接口

    使用多语言模型将文本转换为 384 维向量
    支持中英文混合输入
    """
    if embed_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")

    if not request.texts:
        return EmbedResponse(embeddings=[], dimension=384)

    # 批量编码
    embeddings = embed_model.encode(request.texts, convert_to_numpy=True)

    # 转换为列表格式
    embeddings_list = embeddings.tolist()

    return EmbedResponse(
        embeddings=embeddings_list,
        dimension=embeddings.shape[1] if len(embeddings.shape) > 1 else 384
    )


# ============================================================
# 启动配置
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
