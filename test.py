import os
import base64
import markdown
from markupsafe import Markup
from groq import Groq

# === CONFIGURATION ===
GROQ_API_KEY = "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"  
IMAGE_PATH = "image.jpg" 
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "dicom"}

# === Medical Analysis Prompt ===
MEDICAL_QUERY = """
You are a highly skilled medical imaging expert...

The uploaded image is here:  
![Uploaded Image](data:image/jpeg;base64,{image_base64})

Please analyze it and respond in the following structured format:
### 1. Diagnosis
- Identify the most likely diagnosis based on the uploaded image.
- Provide a brief explanation of how you reached this diagnosis by observing the image.

### 2. Cause of the Condition
- Explain why this condition occurs.
- List common causes, risk factors, or underlying medical reasons contributing to this condition.

### 3. Treatment & Medication
- Provide common treatment options and medical solutions for this diagnosis.
- List commonly prescribed medicines, therapies, or procedures used to treat this condition.

### 4. When to Consult a Doctor
- Advise when the patient should definitely consult a medical specialist.
- Mention warning signs or conditions that require urgent medical attention.

⚠️ Important: Always include this disclaimer at the end:  
"This is an AI-generated analysis based on the image. Please consult a qualified medical professional for an accurate diagnosis and treatment plan."

Format the response in clear markdown headers and concise bullet points.

"""

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def analyze_image(image_path, groq_api_key):
    if not allowed_file(image_path):
        raise ValueError("Invalid file type. Allowed types: png, jpg, jpeg, dicom.")

    base64_image = encode_image(image_path)
    client = Groq(api_key=groq_api_key)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": MEDICAL_QUERY},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )

    markdown_result = chat_completion.choices[0].message.content
    html_result = Markup(markdown.markdown(markdown_result, extensions=['fenced_code', 'tables']))
    
    return markdown_result, html_result

if __name__ == "__main__":
    try:
        markdown_text, html_text = analyze_image(IMAGE_PATH, GROQ_API_KEY)
        print("\n=== Analysis Result (Markdown) ===\n")
        print(markdown_text)
        
        print("\n=== Analysis Result (HTML) ===\n")
        print(html_text)
    except Exception as e:
        print(f"Error: {e}")
