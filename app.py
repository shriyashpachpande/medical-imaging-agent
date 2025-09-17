
import os
import base64
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from groq import Groq
import markdown
from markupsafe import Markup

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "dicom"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Medical analysis prompt
MEDICAL_QUERY = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging.

Below is the uploaded medical image in base64 format:  
![Uploaded Image](data:image/jpeg;base64,{base64_image})

Your task is to analyze this image and respond with the following structured information:

### 1. Diagnosis
- Identify the most likely diagnosis based on the image.
- Explain briefly how you arrived at this diagnosis by observing the image.

### 2. Why This Happens
- Explain common causes, risk factors, or medical reasons that lead to this condition.

### 3. Treatment & Medicines
- Provide typical treatment options, medical solutions, and commonly prescribed medicines.

### 4. When to Consult a Doctor
- List warning signs or situations when the patient should urgently consult a medical professional.

⚠️ Disclaimer:  
"This is an AI-generated analysis based on the image. Please consult a qualified medical professional for an accurate diagnosis and treatment plan."
"""

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

@app.route("/", methods=["GET", "POST"])
def index():
    result_html = None
    error = None
    uploaded_image = None
    groq_api_key = ""
    
    if request.method == "POST":
        groq_api_key = request.form.get("groq_api_key", "").strip()
        if not groq_api_key:
            error = "Groq API Key is required."
            return render_template("index.html", result_html=result_html, error=error, groq_api_key=groq_api_key, uploaded_image=uploaded_image)

        if "image" not in request.files:
            error = "No image file part."
            return render_template("index.html", result_html=result_html, error=error, groq_api_key=groq_api_key, uploaded_image=uploaded_image)
        
        file = request.files["image"]
        if file.filename == "":
            error = "No selected file."
            return render_template("index.html", result_html=result_html, error=error, groq_api_key=groq_api_key, uploaded_image=uploaded_image)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            try:
                base64_image = encode_image(filepath)
                uploaded_image = f"data:image/jpeg;base64,{base64_image}"
                
                client = Groq(api_key=groq_api_key)
                
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": MEDICAL_QUERY.format(base64_image=base64_image)},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": uploaded_image,
                                    },
                                },
                            ],
                        }
                    ],
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                )
                markdown_result = chat_completion.choices[0].message.content
                
                # Convert markdown to HTML
                result_html = Markup(markdown.markdown(markdown_result, extensions=['fenced_code', 'tables']))
            
            except Exception as e:
                error = f"Error during analysis: {e}"
        else:
            error = "Allowed image types are png, jpg, jpeg, dicom."
    
    return render_template("index.html", result_html=result_html, error=error, groq_api_key=groq_api_key, uploaded_image=uploaded_image)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
