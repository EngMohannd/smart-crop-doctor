# Author: Prof. MM Ghassemi <ghassem3@msu.edu>
# Updated for Homework 2: Semantic Search & Human Validation Workflow

import os
import json
import re
import requests
from typing import Dict, List, Optional
from flask import current_app, session, jsonify
from flask_app import socketio
from .socket_events import process_and_emit_message
from .embeddings import generate_embedding

try:
    from .database import query_db, modify_db
except ImportError:
    from database import query_db, modify_db


class ChatGPTClient:
    """Client for interacting with OpenAI / OpenRouter Chat API"""
    def __init__(self, api_key: Optional[str] = None, model: str = None, max_tokens: int = None, temperature: float = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("API key is required.")
        
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "HW2 AI Multi-Agent App"
        }
        
        self.model = "google/gemini-2.5-flash"
        self.max_tokens = max_tokens or 700
        self.temperature = temperature or 0.2

    def send_message(self, message: str, conversation_history: Optional[List[Dict]] = None, system_prompt: Optional[str] = None) -> Dict:
        if conversation_history is None:
            conversation_history = []
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": message})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return {
                "success": True,
                "response": result['choices'][0]['message']['content'].strip(),
                "model": self.model
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": "I'm having trouble connecting right now."
            }


# ==========================================
# 🛡️ Step 2: Human Validation Workflow
# ==========================================

DANGEROUS_KEYWORDS = ['delete', 'remove', 'clear', 'drop', 'destroy', 'erase', 'truncate']

def assess_message_risk(message: str) -> bool:
    """Check if the user request contains dangerous destructive keywords"""
    msg_lower = message.lower()
    return any(re.search(r'\b' + re.escape(word) + r'\b', msg_lower) for word in DANGEROUS_KEYWORDS)


# ==========================================
# 🧠 Step 1.4: Multi-Agent Semantic Experts
# ==========================================

DB_SCHEMA_PROMPT = """
PostgreSQL Database Schema with pgvector (vector(1536)):
- institutions(inst_id SERIAL PRIMARY KEY, type VARCHAR, name VARCHAR, department VARCHAR, address VARCHAR, city VARCHAR, state VARCHAR, zip VARCHAR, embedding vector(1536))
- positions(position_id SERIAL PRIMARY KEY, inst_id INT REFERENCES institutions(inst_id), title VARCHAR, start_date DATE, end_date DATE, responsibilities TEXT, embedding vector(1536))
- experiences(experience_id SERIAL PRIMARY KEY, position_id INT REFERENCES positions(position_id), name VARCHAR, description TEXT, start_date DATE, end_date DATE, embedding vector(1536))
- skills(skill_id SERIAL PRIMARY KEY, experience_id INT REFERENCES experiences(experience_id), name VARCHAR, skill_level INT, embedding vector(1536))
- users(user_id SERIAL PRIMARY KEY, email VARCHAR, password VARCHAR, role VARCHAR, embedding vector(1536))
"""

def clean_sql_output(raw_sql: str) -> str:
    sql = re.sub(r'```sql', '', raw_sql, flags=re.IGNORECASE)
    sql = re.sub(r'```', '', sql).strip()
    return sql

def orchestrator_agent(llm_client: ChatGPTClient, user_message: str) -> str:
    system_prompt = """
    You are the Orchestrator Router. Classify the user message into exactly one category:
    - READ: Querying, viewing, searching information, semantic matching, or resume queries.
    - WRITE: Adding, updating, or modifying database records.
    - CONTENT: General greetings, polite chit-chat, or questions not related to resume data.
    
    Respond ONLY with READ, WRITE, or CONTENT.
    """
    res = llm_client.send_message(user_message, system_prompt=system_prompt)
    decision = res.get("response", "CONTENT").upper()
    if "READ" in decision:
        return "READ"
    elif "WRITE" in decision:
        return "WRITE"
    return "CONTENT"

def db_read_expert(llm_client: ChatGPTClient, user_message: str) -> str:
    """Semantic & Relational Read Expert using vector cosine distance (<=>)"""
    query_emb = generate_embedding(user_message)
    
    semantic_data = {}
    try:
        # 1. البحث الدلالي في المؤسسات (Semantic Search on Institutions)
        inst_matches = query_db(
            "SELECT name, department, (embedding <=> %s::vector) as distance FROM institutions WHERE embedding IS NOT NULL ORDER BY distance ASC LIMIT 2",
            (query_emb,)
        )
        # 2. البحث الدلالي في الخبرات (Semantic Search on Experiences)
        exp_matches = query_db(
            "SELECT name, description, (embedding <=> %s::vector) as distance FROM experiences WHERE embedding IS NOT NULL ORDER BY distance ASC LIMIT 3",
            (query_emb,)
        )
        # 3. البحث الدلالي في المهارات (Semantic Search on Skills)
        skill_matches = query_db(
            "SELECT name, skill_level, (embedding <=> %s::vector) as distance FROM skills WHERE embedding IS NOT NULL ORDER BY distance ASC LIMIT 5",
            (query_emb,)
        )
        semantic_data = {
            "matched_institutions": inst_matches,
            "matched_experiences": exp_matches,
            "matched_skills": skill_matches
        }
    except Exception as e:
        semantic_data = {"error": f"Semantic query error: {str(e)}"}

    synthesis_prompt = f"""
    You are an intelligent Assistant powered by Semantic Vector Search.
    The user asked: "{user_message}"
    
    Retrieved semantic similarity records from PostgreSQL pgvector:
    {json.dumps(semantic_data, default=str, ensure_ascii=False)}
    
    Instructions:
    1. Provide an accurate and comprehensive response answering the user's intent.
    2. If the user used an abbreviation (such as 'MSU' -> 'Michigan State University' or 'AUG' -> 'Al-Azhar University'), resolve and explain it using the retrieved records.
    3. If the user asked for a broad concept (like 'AI skills' or 'embedded systems'), present all relevant skills found via vector similarity.
    """
    res = llm_client.send_message(user_message, system_prompt=synthesis_prompt)
    return res.get("response", "I could not retrieve semantic information.")

def db_write_expert(llm_client: ChatGPTClient, user_message: str) -> str:
    system_prompt = f"""
    You are a PostgreSQL Write Expert.
    Schema: {DB_SCHEMA_PROMPT}
    Generate the SQL query to perform the modification (INSERT, UPDATE, DELETE).
    Output ONLY the raw SQL query without markdown formatting.
    """
    res = llm_client.send_message(user_message, system_prompt=system_prompt)
    sql_query = clean_sql_output(res.get("response", ""))
    
    try:
        modify_db(sql_query)
        return f"Operation executed successfully! (Executed SQL: `{sql_query}`)"
    except Exception as e:
        return f"Database execution error: {str(e)}"

def content_expert(llm_client: ChatGPTClient, user_message: str, conversation_history: list, page_content: dict = None) -> str:
    clean_content = page_content.get('text', '') if page_content else ''
    system_prompt = f"You are a helpful portfolio assistant. Context: {clean_content}"
    res = llm_client.send_message(user_message, conversation_history=conversation_history, system_prompt=system_prompt)
    return res.get("response", "I am here to assist with information about the resume.")


# ==========================================
# 🚀 Entry Point: Handle AI Chat Request
# ==========================================

def handle_ai_chat_request(llm_client: ChatGPTClient, message: str, system_prompt: str = None, room: str = 'main', page_content: dict = None):
    try:
        msg_clean = message.strip()
        msg_lower = msg_clean.lower()
        
        # ----------------------------------------------------
        # 1. التحقق من وجود طلب معلق للمصادقة البشرية (Step 2.2)
        # ----------------------------------------------------
        if 'pending_validation' in session:
            pending_msg = session.pop('pending_validation')
            
            # موافقة المستخدم
            if msg_lower in ['yes', 'y', 'confirm', 'sure']:
                ai_response = db_write_expert(llm_client, pending_msg)
                confirmed_output = f"Operation confirmed by user. {ai_response}"
                process_and_emit_message(socketio, confirmed_output, 'ai', room)
                return jsonify({"success": True, "response": confirmed_output})
                
            # رفض المستخدم
            elif msg_lower in ['no', 'n', 'cancel']:
                cancel_msg = "Action cancelled safely. No changes were made to the database."
                process_and_emit_message(socketio, cancel_msg, 'ai', room)
                return jsonify({"success": True, "response": cancel_msg})
                
            # رد غير مفهوم
            else:
                session['pending_validation'] = pending_msg
                prompt_msg = "Please respond with 'yes' to proceed with deletion or 'no' to cancel."
                process_and_emit_message(socketio, prompt_msg, 'ai', room)
                return jsonify({"success": True, "response": prompt_msg})

        # ----------------------------------------------------
        # 2. تقييم خطورة الرسالة الجديدة (Step 2.1 Risk Assessment)
        # ----------------------------------------------------
        if assess_message_risk(msg_clean):
            session['pending_validation'] = msg_clean
            warn_msg = f"⚠️ Warning: Your request ('{msg_clean}') involves a potentially dangerous action. Are you sure you want to proceed? (yes/no)"
            process_and_emit_message(socketio, warn_msg, 'ai', room)
            return jsonify({"success": True, "response": warn_msg, "validation_required": True})

        # ----------------------------------------------------
        # 3. التدفق الطبيعي عبر الوكلاء متعددي الخبراء
        # ----------------------------------------------------
        conversation_history = session.get('chat_history', [])
        intent = orchestrator_agent(llm_client, msg_clean)
        
        if intent == "READ":
            ai_response = db_read_expert(llm_client, msg_clean)
        elif intent == "WRITE":
            ai_response = db_write_expert(llm_client, msg_clean)
        else:
            ai_response = content_expert(llm_client, msg_clean, conversation_history, page_content)

        conversation_history.append({"role": "user", "content": msg_clean})
        conversation_history.append({"role": "assistant", "content": ai_response})
        session['chat_history'] = conversation_history[-10:]
        
        process_and_emit_message(socketio, ai_response, 'ai', room)
        return jsonify({"success": True, "response": ai_response, "intent": intent})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "response": f"An error occurred: {str(e)}"}), 500