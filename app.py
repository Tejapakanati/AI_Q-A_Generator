
from fastapi import FastAPI, Form, Request, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from xml.sax.saxutils import escape

import os
import uvicorn

from src.main import llm_pipeline


app = FastAPI()

# Create required folders
os.makedirs("static/docs", exist_ok=True)
os.makedirs("static/output", exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


# Home page
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# Upload PDF
@app.post("/upload")
async def upload_pdf(
    pdf_file: UploadFile = File(...),
    file_name: str = Form(...)
):
    if not file_name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file."
        )

    safe_filename = os.path.basename(file_name)
    pdf_path = os.path.join("static", "docs", safe_filename)

    try:
        content = await pdf_file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF is empty."
            )

        with open(pdf_path, "wb") as f:
            f.write(content)

        return JSONResponse({
            "message": "PDF uploaded successfully",
            "pdf_filename": safe_filename
        })

    finally:
        await pdf_file.close()


# Generate questions and answers, then save them to PDF
def get_pdf(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    answer_generation_chain, ques_list = llm_pipeline(file_path)

    output_file = os.path.join(
        "static", "output", "QA.pdf"
    )

    styles = getSampleStyleSheet()

    question_style = ParagraphStyle(
        "QuestionStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        spaceAfter=8
    )

    answer_style = ParagraphStyle(
        "AnswerStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        spaceAfter=18
    )

    content = []

    content.append(
        Paragraph(
            "Question and Answer Report",
            styles["Title"]
        )
    )
    content.append(Spacer(1, 0.3 * inch))

    for index, question in enumerate(ques_list, start=1):
        print(f"Question {index}:", question)

        result = answer_generation_chain.invoke({
            "query": question
        })

        if isinstance(result, dict):
            answer = (
                result.get("result")
                or result.get("answer")
                or str(result)
            )
        else:
            answer = str(result)

        print("Answer:", answer)

        safe_question = escape(str(question)).replace(
            "\n", "<br/>"
        )
        safe_answer = escape(str(answer)).replace(
            "\n", "<br/>"
        )

        content.append(
            Paragraph(
                f"Question {index}: {safe_question}",
                question_style
            )
        )

        content.append(
            Paragraph(
                f"<b>Answer:</b><br/>{safe_answer}",
                answer_style
            )
        )

    document = SimpleDocTemplate(
        output_file,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )

    document.build(content)

    print("PDF created successfully:", output_file)

    return output_file


# Analyze PDF and return PDF download link
@app.post("/analyze")
async def analyze_pdf(pdf_filename: str = Form(...)):
    try:
        safe_filename = os.path.basename(pdf_filename)
        pdf_path = os.path.join(
            "static", "docs", safe_filename
        )

        print("PDF received for analysis:", pdf_path)

        output_file = get_pdf(pdf_path)

        return JSONResponse({
            "message": "Questions and answers generated successfully",
            "output_file": output_file,
            "download_url": "/static/output/QA.pdf"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "error_type": type(e).__name__
            }
        )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )