"""Factory module returning either Ollama, Groq, or OpenAI depending on configuration."""
import os
from langchain_core.language_models.chat_models import BaseChatModel

def get_chat_model(temperature: float = 0.0) -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        return ChatOllama(model=model_name, temperature=temperature)

    elif provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is required when LLM_PROVIDER=groq")
        return ChatGroq(model_name=model_name, temperature=temperature, api_key=api_key)

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model_name, temperature=temperature)

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}. Use 'ollama', 'groq', or 'openai'.")
