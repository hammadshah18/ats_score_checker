# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
import os
import re
import json

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Cerevo ATS Resume Analyzer")

# Allow CORS (for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini model lazily
llm = None

def get_llm():
    global llm
    if llm is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured")
        llm = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    return llm

# ---------- PROMPT TEMPLATE ----------
prompt = PromptTemplate(
    input_variables=["resume_text"],
    template="""
You are an advanced Applicant Tracking System (ATS) evaluator and professional resume analyst.

Your job is to analyze the provided resume text as if you are a next-generation AI ATS, focusing on structure, readability, keyword optimization, and recruiter appeal.

Evaluate the resume on the following 14 key criteria, each scored from 0 to 100, and provide short but specific feedback for each:

1. ats_parse_rate — How accurately the resume text can be parsed by modern ATS systems (layout simplicity, use of standard sections, etc.).
2. quantifying_impact — How effectively the resume quantifies achievements with numbers, metrics, or measurable results.
3. repetition — Penalize repeated phrases, overused buzzwords, or redundant sentences.
4. spelling_grammar — Grammar, spelling, and sentence clarity.
5. essential_sections — Whether essential sections (Summary, Experience, Education, Skills) are present and complete.
6. contact_information — Presence and clarity of name, phone, email, and location.
7. ats_essentials — Use of standard section headers and keyword structure recognized by ATS.
8. file_format_size — Simulate scoring based on ideal resume formats (PDF/DOCX, size under 2MB, readable text — assume good if text is parsable).
9. design_layout — Readability, consistent font usage, spacing, and professional visual design.
10. email_address — Whether the email address appears professional (avoid nicknames, numbers, etc.).
11. hyperlink_in_header — Check if any URLs, emails, or hyperlinks appear in the header (these often break ATS parsing).
12. tailoring — How well the resume seems customized toward a role (vs generic template).
13. hard_skills — Relevance and strength of technical or domain-specific skills.
14. soft_skills — Communication, teamwork, and leadership skills visibility.
15. action_verbs — Usage of strong, impactful verbs like "developed," "led," "optimized," etc.
16. tailored_title — Whether the job title or summary aligns with a specific target position.

After scoring all factors, calculate:
- overall_score (average of all above metrics)

Return output as **valid JSON only**, in the format:

{{
  "overall_score": 0-100,
  "criteria": {{
    "ats_parse_rate": {{ "score": 0-100, "feedback": "..." }},
    "quantifying_impact": {{ "score": 0-100, "feedback": "..." }},
    "repetition": {{ "score": 0-100, "feedback": "..." }},
    "spelling_grammar": {{ "score": 0-100, "feedback": "..." }},
    "essential_sections": {{ "score": 0-100, "feedback": "..." }},
    "contact_information": {{ "score": 0-100, "feedback": "..." }},
    "ats_essentials": {{ "score": 0-100, "feedback": "..." }},
    "file_format_size": {{ "score": 0-100, "feedback": "..." }},
    "design_layout": {{ "score": 0-100, "feedback": "..." }},
    "email_address": {{ "score": 0-100, "feedback": "..." }},
    "hyperlink_in_header": {{ "score": 0-100, "feedback": "..." }},
    "tailoring": {{ "score": 0-100, "feedback": "..." }},
    "hard_skills": {{ "score": 0-100, "feedback": "..." }},
    "soft_skills": {{ "score": 0-100, "feedback": "..." }},
    "action_verbs": {{ "score": 0-100, "feedback": "..." }},
    "tailored_title": {{ "score": 0-100, "feedback": "..." }}
  }},
  "final_feedback": "Write 3-5 lines of professional overall feedback summarizing strengths and areas for improvement."
}}

Analyze deeply and respond only with JSON.

RESUME TEXT:
{resume_text}
"""
)

# ---------- PDF & DOCX Extraction ----------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# ---------- Helper: Clean JSON response ----------
def extract_json(text: str):
    """Extract and clean JSON from LLM response."""
    # Remove markdown fences
    text = re.sub(r"```json|```", "", text).strip()
    # Extract first {...} block
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No valid JSON found in LLM output.")
    return json.loads(match.group(0))

# ---------- Routes ----------

@app.get("/")
def root():
    return {
        "message": "Cerevo ATS Resume Analyzer API",
        "status": "running",
        "endpoints": {
            "/": "API info",
            "/home": "Welcome message",
            "/analyze_resume": "POST - Analyze resume (PDF/DOCX)"
        }
    }

@app.get("/home")
def home():
    return {"message": "Welcome to the Cerevo ATS Resume Analyzer API", "status": "ok"}

@app.post("/analyze_resume")
async def analyze_resume(file: UploadFile = File(...)):
    filename = file.filename.lower()

    if not (filename.endswith(".pdf") or filename.endswith(".docx")):
        return {"error": "Invalid file type. Please upload a PDF or DOCX resume."}

    # Extract text
    if filename.endswith(".pdf"):
        resume_text = extract_text_from_pdf(file.file)
    elif filename.endswith(".docx"):
        resume_text = extract_text_from_docx(file.file)
    else:
        return {"error": "Unsupported file format."}

    # Prepare prompt
    filled_prompt = prompt.format(resume_text=resume_text)

    # Get LLM response
    model = get_llm()
    response = model.invoke(filled_prompt)
    raw_output = response.content if hasattr(response, "content") else str(response)

    # Parse and clean JSON safely
    try:
        ats_result = extract_json(raw_output)
    except Exception as e:
        ats_result = {"error": "Invalid JSON response from LLM.", "raw_output": raw_output}

    return ats_result

# Vercel serverless handler
handler = app
