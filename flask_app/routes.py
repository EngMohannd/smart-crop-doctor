# Author: Prof. MM Ghassemi <ghassem3@msu.edu>

import os
import base64
import json
import openai
from openai import OpenAI
from flask import current_app as app
from flask import render_template, redirect, request, session, url_for, jsonify, send_from_directory
from .config import Config
from .utils.database import database
from .utils.llm import ChatGPTClient
from .utils.llm import handle_ai_chat_request
from bs4 import BeautifulSoup

#--------------------------------------------------
# GLOBAL INSTANCES
#--------------------------------------------------
db = database()

#--------------------------------------------------
# MAIN APPLICATION ROUTES
#--------------------------------------------------

@app.route('/')
def home():
    return render_template('dynamic-page.html', user=db.get_user_email(), page_type='home')

@app.route('/agents')
def agents():
    return render_template('dynamic-page.html', user=db.get_user_email(), page_type='agents')

@app.route('/agents/resume')
def resume():
    return render_template('dynamic-page.html', user=db.get_user_email(), page_type='resume')

@app.route('/api/resume')
def api_resume():
    """API endpoint to serve resume data as JSON for Vue.js frontend."""
    resume_data = db.getResumeData()
    return jsonify({ "success": True, "data": resume_data})

# ---- مسارات طبيب المحاصيل الذكي (Plant Doctor) ----



@app.route('/plant-doctor')
def plant_doctor_page():
    return render_template('plant_doctor.html')

@app.route('/api/diagnose', methods=['POST'])
def diagnose_crop():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    crop_hint = request.form.get('crop_hint', 'Tomato')
    
    image_bytes = file.read()
    b64_image = base64.b64encode(image_bytes).decode('utf-8')

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost:5000",
                "X-Title": "Smart Crop Doctor"
            }
        )

        # 1. المرحلة الأولى: استخراج التشخيص واسم المرض من الصورة
        vision_prompt = f"""
        You are an expert agronomist. Look at this leaf image for crop '{crop_hint}'.
        Identify:
        1. Disease/Condition Name (e.g., Early Blight, Powdery Mildew, Blossom End Rot, or Healthy).
        2. Brief observed symptoms.
        Keep your answer focused and state the primary suspected disease name explicitly on the first line as:
        DISEASE: <disease_name>
        """

        vision_response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                    ]
                }
            ],
            max_tokens=250
        )
        vision_text = vision_response.choices[0].message.content

        # استخراج اسم المرض المقترح للبحث به في قاعدة البيانات
        detected_disease = "General"
        for line in vision_text.splitlines():
            if line.upper().startswith("DISEASE:"):
                detected_disease = line.split(":", 1)[1].strip()
                break

        # 2. المرحلة الثانية: جلب التضمين (Embedding) والبحث في pgvector
        query_text = f"{crop_hint} {detected_disease}"
        
        # استدعاء Embedding
        try:
            emb_res = client.embeddings.create(
                model="text-embedding-3-small",
                input=query_text
            )
            embedding_vector = str(emb_res.data[0].embedding)
        except Exception:
            embedding_vector = None

        # استعلام قاعدة البيانات
        db_records = db.search_plant_treatment(detected_disease, embedding_vector)

        context_data = ""
        if db_records:
            context_data = "\n\n--- LOCAL KNOWLEDGE BASE PROTOCOLS ---\n"
            for rec in db_records:
                context_data += (
                    f"• Crop: {rec[0]} | Target: {rec[1]} | Type: {rec[2]}\n"
                    f"  Active Ingredient: {rec[3]} ({rec[4]})\n"
                    f"  Dosage: {rec[5]} | PHI Safety Period: {rec[6]} days\n"
                    f"  Instructions: {rec[7]}\n"
                )
        else:
            context_data = "\n(No direct matching record in local DB. Applying generalized agronomic protocol.)"

        # 3. المرحلة الثالثة: صياغة التقرير النهائي التفاعلي من قِبل الوكيل
        agent_synthesis_prompt = f"""
You are the Smart Crop Doctor AI Agent, an expert agronomic decision-support system.

Target Crop Context: {crop_hint}
Visual Diagnostic Findings:
{vision_text}

Verified Knowledge Base Records from Database:
{context_data}

Instructions:
- Provide a rigorous, fully fleshed-out Agricultural Diagnostic Report.
- Strictly DO NOT output placeholders, bracketed templates, or incomplete text (e.g., never write '[Specify Crop]', '[Insert Fungicide]', or '[Dosage]').
- If a verified local database protocol is present above, adopt its exact active ingredients, dosages, and Pre-Harvest Interval (PHI).
- If no local record exists, supply standard calibrated agronomic compounds (e.g., Copper Hydroxide, Chlorothalonil, Tebuconazole, or appropriate bio-fungicides) with explicit dosage per liter/hectare and definitive PHI days.

Structure your report under these exact sections:
### 1. Diagnostic Summary
- Crop: (Name the identified or selected crop explicitly)
- Suspected Disease / Pathogen: (Scientific and common name)
- Severity Level: (Mild / Moderate / Severe)

### 2. Visual Symptoms Observed
(Concrete bullet points describing lesions, coloration, margins, and tissue degradation)

### 3. Recommended Treatment Plan
- Primary Chemical Intervention: (Active ingredient, commercial example, precise application dosage, and spray volume)
- Organic / Bio-control Option: (Bio-fungicide, microbial agent, or sulfur/copper formulation with dosage)
- Application Method & Frequency: (Foliar coverage, timing, interval between applications)
- Safety & Pre-Harvest Interval (PHI): (Exact mandatory waiting period in days before harvest)

### 4. Cultural & Preventative Management
(Targeted cultural practices: sanitation, irrigation methods, canopy management, and soil amendments)
"""
        final_response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional agronomist agent providing safe, calibrated crop prescriptions."},
                {"role": "user", "content": agent_synthesis_prompt}
            ],
            max_tokens=650
        )

        final_report = final_response.choices[0].message.content
        return jsonify({'diagnosis': final_report})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

#--------------------------------------------------
# CHAT ROUTES
#--------------------------------------------------
@app.route('/chat/ai', methods=['POST'])
def chat_with_ai():
    data          = request.get_json()
    message       = data.get('message', '').strip()
    page_content  = data.get('pageContent', {})
    
    print(f"Received message: {message}")
    print(f"Received page content: {page_content}")
    
    chatGPT = ChatGPTClient()
    
    if page_content and page_content.get('content'):
        clean_content = clean_html_content(page_content.get('content', ''))
        
        system_prompt = f""" 
            You are a helpful AI assistant. You have access to the current page content that the user is viewing.
            IMPORTANT INSTRUCTIONS:
            1. If the user's question is related to the content on the current page, use that content to provide accurate and relevant answers.
            2. Reference specific information from the page when it helps answer the user's question.
            3. If the user asks about something not covered on the current page, provide general helpful information.
            4. Be conversational and helpful while maintaining accuracy.

            CURRENT PAGE CONTENT:
            Title:   {page_content.get('title', 'Unknown page')}
            URL:     {page_content.get('url', 'N/A')}
            Content: {clean_content}

            Use this content to provide contextually relevant responses when appropriate.
        """
    else:
        system_prompt = "You are a helpful AI assistant."
        print("Using fallback system prompt (no page content available)")

    return handle_ai_chat_request(llm_client=chatGPT, message=message, system_prompt=system_prompt, room='main', page_content=page_content)

#--------------------------------------------------
# AUTHENTICATION ROUTES
#--------------------------------------------------
@app.route('/login')
def login():
    return render_template('dynamic-page.html', user=db.get_user_email(), page_type='login')

@app.route('/processlogin', methods=["POST", "GET"])
def processlogin():
    data     = request.get_json()
    email    = data.get('email')
    password = data.get('password')

    if not email or not password:
        return json.dumps({"success": 0,"error": "Email and password are required"})
    
    status = db.authenticate(email=email, password=password)
    session['email'] = db.reversibleEncrypt('encrypt', email) 

    return json.dumps(status)

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

#--------------------------------------------------
# UTILITY ROUTES
#--------------------------------------------------

@app.route("/static/<path:path>")
def static_dir(path):
    return send_from_directory("static", path)

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r

def clean_html_content(html_content):
    if not html_content:
        return ""
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        text = ' '.join(text.split())
        return text
    except Exception as e:
        print(f"Error cleaning HTML content: {e}")
        return html_content