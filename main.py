import logging
import os
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import GenerativeModel

# Load environment variables
load_dotenv()

# Enable logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI app
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://career-guidance-avud.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-api-key")
genai.configure(api_key=GEMINI_API_KEY)

# Define the ChatRequest class
class ChatRequest(BaseModel):
    query: str

# Models for user profile
class UserProfile(BaseModel):
    uid: str
    email: str
    academic_background: str
    skills: List[str]
    interests: List[str]
    qualifications: List[str]

@app.get("/")
def read_root():
    return {"message": "FastAPI backend is running!"}

@app.post("/submit-profile")
def submit_profile(profile: UserProfile):
    logging.debug(f"Received profile: {profile}")

    # Combine skills, interests, and academic background into text
    skills_text = " ".join(profile.skills)
    interests_text = " ".join(profile.interests)
    academics_text = profile.academic_background

    # Get career recommendations
    career_recommendations = get_job_recommendations(skills_text, interests_text, academics_text)

    # Log the career recommendations
    logging.debug(f"Career Recommendations: {career_recommendations}")

    return {
        "message": "Profile received and analyzed!",
        "timestamp": datetime.utcnow().isoformat(),
        "user_uid": profile.uid,
        "career_recommendations": career_recommendations
    }

# Function to generate job recommendations
def get_job_recommendations(skills_text: str, interests_text: str, academics_text: str):
    query = f"""
    Skills: {skills_text} 
    Interests: {interests_text} 
    Academic Background: {academics_text}
    
    Based on the above information, recommend a career field, job specialization, and a brief description of the job role.
    Provide them as:
    Career Field: <Career Field>
    Job Specialization: <Job Specialization>
    Description: <Job Description>
    
    Ensure each part is on a new line and starts with the exact label (e.g., "Career Field:").
    Example:
    Career Field: Software Engineering
    Job Specialization: Backend Development
    Description: Backend developers build server-side systems to manage data and requests.
    """

    # Use the correct Gemini model
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",  # Updated to a valid model name
    )

    try:
        # Generate the content
        response = model.generate_content(query)
        response_text = response.text.strip()

        # Log the raw response for debugging
        logging.debug(f"Raw Gemini Response: {response_text}")

        # Use regex to extract fields
        career_field_match = re.search(r'^Career Field:\s*(.+)$', response_text, re.MULTILINE)
        job_specialization_match = re.search(r'^Job Specialization:\s*(.+)$', response_text, re.MULTILINE)
        description_match = re.search(r'^Description:\s*(.+)$', response_text, re.MULTILINE)

        # Extract values or set defaults
        career_field = career_field_match.group(1).strip() if career_field_match else "No career field found"
        job_specialization = job_specialization_match.group(1).strip() if job_specialization_match else "No job specialization found"
        description = description_match.group(1).strip() if description_match else "No description available"

        # Remove markdown (e.g., **)
        career_field = re.sub(r'\*\*|\*', '', career_field)
        job_specialization = re.sub(r'\*\*|\*', '', job_specialization)
        description = re.sub(r'\*\*|\*', '', description)

        # Truncate description if too long
        description_words = description.split()
        if len(description_words) > 50:
            description = " ".join(description_words[:50]) + "..."

        return {
            "career_field": career_field,
            "job_specialization": job_specialization,
            "description": description,
        }

    except Exception as e:
        logging.error(f"Gemini API error: {str(e)}")
        return {"error": f"Gemini API error: {str(e)}"}

@app.post("/chatbot")
async def chatbot(query: ChatRequest):
    gemini_response = get_gemini_response(query.query)
    if gemini_response.get("error"):
        raise HTTPException(status_code=500, detail=gemini_response["error"])
    return {"response": gemini_response["response"]}

def get_gemini_response(query: str):
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
    )
    prompt = f"""
    You are a helpful and concise chatbot. Respond in 2-3 short, friendly sentences.
    User query: "{query}"
    Provide a brief, clear answer, and ask for clarification if needed.
    """
    try:
        response = model.generate_content(prompt)
        reply = response.text.strip()
        if len(reply.split()) > 50:
            reply = "I will keep it short: " + ' '.join(reply.split()[:50]) + "..."
        return {"response": reply}
    except Exception as e:
        logging.error(f"Gemini API error: {str(e)}")
        return {"error": f"Gemini API error: {str(e)}"}