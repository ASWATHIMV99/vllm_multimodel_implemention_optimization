import os
import base64
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from flask_swagger import swagger
from flask_swagger_ui import get_swaggerui_blueprint
from google import genai
from google.genai import types
import tempfile
import mimetypes
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Gemini client
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
)

# Model to use
MODEL_NAME = "gemini-2.5-flash"

# Swagger UI configuration
SWAGGER_URL = '/api/docs'
API_URL = '/spec'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Multimodal Gemini API"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# HTML template for the main page with redirect to Swagger UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Multimodal Gemini API</title>
</head>
<body>
    <h1>Multimodal Gemini API</h1>
    <p>API is running successfully!</p>
    <p><a href="/api/docs">Visit Swagger UI Documentation</a></p>
</body>
</html>
"""

@app.route('/')
def index():
    """
    Health check endpoint
    ---
    responses:
      200:
        description: API is running
    """
    return render_template_string(HTML_TEMPLATE)

@app.route('/spec')
def spec():
    """
    Swagger specification endpoint
    ---
    responses:
      200:
        description: Swagger specification
    """
    swag = swagger(app)
    swag['info']['version'] = "1.0"
    swag['info']['title'] = "Multimodal Gemini API"
    return jsonify(swag)

@app.route('/text', methods=['POST'])
def generate_text():
    """
    Generate content from text prompt
    ---
    tags:
      - Text
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - prompt
          properties:
            prompt:
              type: string
              description: Text prompt for generation
            stream:
              type: boolean
              description: Whether to stream the response
              default: false
    consumes:
      - application/json
    responses:
      200:
        description: Generated text content
        schema:
          type: object
          properties:
            text:
              type: string
      400:
        description: Bad request
      500:
        description: Internal server error
    """
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        stream = data.get('stream', False)
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        
        if stream:
            # For streaming, we'll return the first chunk for simplicity in this example
            response_text = ""
            for chunk in client.models.generate_content_stream(
                model=MODEL_NAME,
                contents=contents,
            ):
                response_text += chunk.text
            return jsonify({"text": response_text})
        else:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
            )
            return jsonify({"text": response.text})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/image', methods=['POST'])
def generate_from_image():
    """
    Generate content from image
    ---
    tags:
      - Image
    parameters:
      - name: image
        in: formData
        type: file
        required: true
        description: Image file to analyze
      - name: prompt
        in: formData
        type: string
        required: false
        description: Prompt for image analysis
        default: "Describe this image."
    consumes:
      - multipart/form-data
    responses:
      200:
        description: Generated content based on image
        schema:
          type: object
          properties:
            text:
              type: string
      400:
        description: Bad request
      500:
        description: Internal server error
    """
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Image file is required"}), 400
            
        image_file = request.files['image']
        prompt = request.form.get('prompt', 'Describe this image.')
        
        if image_file.filename == '':
            return jsonify({"error": "No image selected"}), 400
            
        # Save image to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            image_file.save(tmp_file.name)
            tmp_filename = tmp_file.name
            
        try:
            # Read image bytes
            with open(tmp_filename, 'rb') as f:
                image_bytes = f.read()
                
            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(image_file.filename)
            if mime_type is None:
                mime_type = 'image/jpeg'  # Default to JPEG
                
            # Generate content
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type,
                    ),
                    prompt
                ]
            )
            
            return jsonify({"text": response.text})
        finally:
            # Clean up temporary file
            os.unlink(tmp_filename)
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/audio', methods=['POST'])
def generate_from_audio():
    """
    Generate content from audio
    ---
    tags:
      - Audio
    parameters:
      - name: audio
        in: formData
        type: file
        required: true
        description: Audio file to analyze (MP3 format recommended)
      - name: prompt
        in: formData
        type: string
        required: false
        description: Prompt for audio analysis
        default: "Describe this audio clip."
    consumes:
      - multipart/form-data
    responses:
      200:
        description: Generated content based on audio
        schema:
          type: object
          properties:
            text:
              type: string
      400:
        description: Bad request
      500:
        description: Internal server error
    """
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "Audio file is required"}), 400
            
        audio_file = request.files['audio']
        prompt = request.form.get('prompt', 'Describe this audio clip.')
        
        if audio_file.filename == '':
            return jsonify({"error": "No audio selected"}), 400
            
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            audio_file.save(tmp_file.name)
            tmp_filename = tmp_file.name
            
        try:
            # Upload file to Gemini with explicit MIME type in config
            uploaded_file = client.files.upload(
                file=tmp_filename,
                config={'mime_type': 'audio/mp3'}
            )
            
            # Generate content using the uploaded file directly
            response = client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=[prompt, uploaded_file]
            )
            
            return jsonify({"text": response.text})
        finally:
            # Clean up temporary file
            os.unlink(tmp_filename)
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/multimodal', methods=['POST'])
def generate_from_multimodal():
    """
    Generate content from multiple modalities (text, image, audio)
    ---
    tags:
      - Multimodal
    parameters:
      - name: text
        in: formData
        type: string
        required: false
        description: Text prompt
      - name: image
        in: formData
        type: file
        required: false
        description: Image file to analyze
      - name: audio
        in: formData
        type: file
        required: false
        description: Audio file to analyze
      - name: prompt
        in: formData
        type: string
        required: false
        description: General prompt for all modalities
        default: "Analyze these inputs."
    consumes:
      - multipart/form-data
    responses:
      200:
        description: Generated content based on all provided modalities
        schema:
          type: object
          properties:
            text:
              type: string
      400:
        description: Bad request
      500:
        description: Internal server error
    """
    try:
        contents = []
        
        # Add text if provided
        text = request.form.get('text')
        prompt = request.form.get('prompt', 'Analyze these inputs.')
        if text:
            contents.append(types.Part.from_text(text=text))
        else:
            contents.append(types.Part.from_text(text=prompt))
            
        # Handle image if provided
        if 'image' in request.files and request.files['image'].filename != '':
            image_file = request.files['image']
            
            # Save image to temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                image_file.save(tmp_file.name)
                tmp_filename = tmp_file.name
                
            try:
                # Read image bytes
                with open(tmp_filename, 'rb') as f:
                    image_bytes = f.read()
                    
                # Determine MIME type
                mime_type, _ = mimetypes.guess_type(image_file.filename)
                if mime_type is None:
                    mime_type = 'image/jpeg'  # Default to JPEG
                    
                # Add image to contents
                contents.append(types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                ))
            finally:
                # Clean up temporary file
                os.unlink(tmp_filename)
                
        # Handle audio if provided
        if 'audio' in request.files and request.files['audio'].filename != '':
            audio_file = request.files['audio']
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                audio_file.save(tmp_file.name)
                tmp_filename = tmp_file.name
                
            try:
                # Upload file to Gemini with explicit MIME type in config
                uploaded_file = client.files.upload(
                    file=tmp_filename,
                    config={'mime_type': 'audio/mp3'}
                )
                
                # Add audio to contents
                contents.append(uploaded_file)
            finally:
                # Clean up temporary file
                os.unlink(tmp_filename)
                
        if not contents:
            return jsonify({"error": "At least one modality (text, image, or audio) must be provided"}), 400
            
        # Generate content from all modalities
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents
        )
        
        return jsonify({"text": response.text})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)