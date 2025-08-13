# Multimodal Gemini API

A Flask-based API for interacting with Google's Gemini model using multiple modalities (text, image, audio).

## Features

- Text generation from text prompts
- Content analysis from images
- Content analysis from audio files (MP3 format)
- Combined multimodal analysis (text + image + audio)
- Swagger documentation for all endpoints

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your Gemini API key in the `.env` file:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

## Running the API

```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### Health Check
- `GET /` - Health check endpoint

### Swagger Documentation
- `GET /api/docs` - Swagger UI documentation
- `GET /spec` - Swagger specification

### Text Generation
- `POST /text` - Generate content from text prompt

### Image Analysis
- `POST /image` - Generate content from image file

### Audio Analysis
- `POST /audio` - Generate content from audio file (MP3 format)

### Multimodal Analysis
- `POST /multimodal` - Generate content from multiple modalities

## Supported File Formats

### Images
- JPEG - image/jpeg
- PNG - image/png
- WEBP - image/webp
- HEIC - image/heic
- HEIF - image/heif

### Audio
- MP3 - audio/mp3 (primary support)
- WAV - audio/wav
- AIFF - audio/aiff
- AAC - audio/aac
- OGG Vorbis - audio/ogg
- FLAC - audio/flac

## How It Works

### Audio Processing
The API uses the Gemini Files API to upload audio files before processing them. This approach:
- Supports larger audio files (up to 9.5 hours)
- Is more efficient for repeated use of the same audio file
- Explicitly sets the MIME type in the upload config to avoid errors

### Image Processing
Images are processed directly as inline data with the request, which is efficient for smaller files.

## Testing the API

You can test the API using the provided test scripts:

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. In another terminal, run the test script:
   ```bash
   python client_example.py
   ```

3. For testing with actual files, modify the client_example.py script to include paths to your image and audio files.

## Using Swagger UI

Once the application is running, you can access the Swagger UI at:
http://localhost:5000/api/docs

The Swagger UI provides an interactive interface to test all API endpoints, including file upload capabilities for image and audio endpoints.

## Implementation Details

For audio processing, the API explicitly sets the MIME type to 'audio/mp3' when sending the audio data to the Gemini model. This ensures proper handling of audio files by the model.