prompt_template = """
You are an expert technical interviewer.

Analyze the following document content and generate interview questions
based only on the information provided in the document.

Document:
{text}

Instructions:
- Generate important interview questions from the document.
- Cover important concepts, definitions, processes, goals, and examples.
- Generate a mixture of basic, intermediate, and scenario-based questions.
- Questions must be suitable for an interview.
- Do not create questions about information that is not present in the document.
- Avoid duplicate questions.
- Return only the interview questions.

Interview Questions:
"""


refine_template = """
You are an expert technical interviewer.

You already have a list of interview questions generated from the document.

Existing Interview Questions:
{existing_answer}

Additional Document Content:
{text}

Instructions:
- Review the existing interview questions.
- Use the additional document content to generate new relevant questions.
- Add important questions that are missing.
- Remove duplicate or very similar questions.
- Improve unclear questions.
- Keep the questions suitable for interviews.
- Cover important concepts from the additional document content.
- Do not create questions about information that is not present in the document.
- Keep the final list organized and concise.
- Return only the final interview questions.

Final Interview Questions:
"""