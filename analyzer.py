import json
import os

from google import genai
from google.genai import types


class ResumeAnalyzer:

    def __init__(self):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing from the .env file."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash"
        )

    def analyze(
        self,
        resume_context,
        job_description
    ):

        prompt = f"""
You are an expert technical recruiter and resume analyzer.

Your task is to analyze a candidate's resume against a
given job description.

IMPORTANT RULES:

1. Use ONLY the provided resume context for information
   about the candidate.

2. Never invent skills, projects, experience,
   certifications, education, or technologies.

3. If information is not present in the resume context,
   say "Not found in resume".

4. Be honest and practical.

5. Generate interview questions based on the candidate's
   actual resume and the job requirements.

RESUME CONTEXT
==============
{resume_context}

JOB DESCRIPTION
===============
{job_description}

Return ONLY valid JSON.

Use exactly this structure:

{{
    "match_score": 0,
    "strong_skills": [],
    "missing_skills": [],
    "matching_projects": [],
    "experience_analysis": "",
    "resume_improvements": [],
    "interview_questions": [],
    "summary": ""
}}

Rules for the JSON:

- match_score must be an integer between 0 and 100.
- strong_skills must contain skills clearly supported
  by the resume.
- missing_skills must contain important job requirements
  not clearly found in the resume.
- matching_projects must contain relevant projects
  actually present in the resume.
- experience_analysis must explain the relevance of
  the candidate's experience.
- resume_improvements must contain actionable suggestions.
- interview_questions must contain exactly 8 questions.
- summary must provide a concise overall assessment.
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json"
            )
        )

        output = response.text

        return self.parse_json(output)

    @staticmethod
    def parse_json(output):

        try:

            return json.loads(output)

        except json.JSONDecodeError:

            start = output.find("{")
            end = output.rfind("}")

            if start != -1 and end != -1:

                try:

                    return json.loads(
                        output[start:end + 1]
                    )

                except json.JSONDecodeError:
                    pass

            return {
                "match_score": 0,
                "strong_skills": [],
                "missing_skills": [],
                "matching_projects": [],
                "experience_analysis":
                    "Unable to parse the AI response.",
                "resume_improvements": [],
                "interview_questions": [],
                "summary": output
            }