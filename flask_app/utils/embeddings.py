# Author: Prof. MM Ghassemi <ghassem3@msu.edu>

import os
import requests

def generate_embedding(text: str):
    """Generate 1536-dimensional embedding using OpenAI / OpenRouter API"""
    if not text or not str(text).strip():
        return [0.0] * 1536
    
    api_key = os.getenv('OPENAI_API_KEY')
    
    # 1. محاولة استخدام OpenAI Direct
    try:
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "text-embedding-3-small",
                "input": str(text).strip()
            },
            timeout=15
        )
        if response.status_code == 200:
            return response.json()['data'][0]['embedding']
    except Exception:
        pass

    # 2. محاولة احتياطية عبر OpenRouter
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/text-embedding-3-small",
                "input": str(text).strip()
            },
            timeout=15
        )
        if response.status_code == 200:
            return response.json()['data'][0]['embedding']
    except Exception:
        pass

    return [0.0] * 1536