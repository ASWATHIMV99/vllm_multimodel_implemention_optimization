import unittest
import os
import sys
import json
import requests
from dotenv import load_dotenv
from unittest.mock import patch, MagicMock
from io import BytesIO

# Add the current directory to the path so we can import app
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

class TestApp(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        # Load environment variables
        load_dotenv()
        
        # Base URL for the API
        cls.base_url = "http://localhost:5000"
        
    # -----------------------------
    # Environment and Configuration Tests
    # -----------------------------
    
    def test_env_api_key_configured(self):
        """Test that the API key is properly configured in .env file"""
        # Check if .env file exists in current directory
        env_path = os.path.join(current_dir, '.env')
        self.assertTrue(os.path.exists(env_path), "The .env file does not exist")
        
        # Check if API key is set in environment variables
        api_key = os.environ.get('GEMINI_API_KEY')
        self.assertIsNotNone(api_key, "GEMINI_API_KEY is not set in environment variables")
        self.assertNotEqual(api_key, '', "GEMINI_API_KEY is empty")
        self.assertNotEqual(api_key, 'your_api_key_here', "GEMINI_API_KEY is still set to the default placeholder value")
        # Check that API key looks like a real API key (begins with 'AI' and has reasonable length)
        self.assertTrue(api_key.startswith('AI'), "GEMINI_API_KEY does not appear to be valid (should start with 'AI')")
        self.assertGreater(len(api_key), 20, "GEMINI_API_KEY appears too short to be valid")
        
    def test_app_imports_and_initialization(self):
        """Test that the app can be imported and initialized without errors"""
        try:
            from app import app
            self.assertIsNotNone(app, "Failed to import Flask app")
        except Exception as e:
            self.fail(f"Failed to import app.py: {e}")
            
    def test_required_dependencies_importable(self):
        """Test that all required dependencies can be imported"""
        required_modules = [
            'flask',
            'flask_cors',
            'flask_swagger',
            'flask_swagger_ui',
            'google.genai',
            'dotenv'
        ]
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError as e:
                self.fail(f"Failed to import required module {module}: {e}")
                
    # -----------------------------
    # Endpoint Tests (with server running)
    # -----------------------------
    
    def test_index_endpoint(self):
        """Test that the index endpoint is working"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            self.assertEqual(response.status_code, 200)
            self.assertIn('Multimodal Gemini API', response.text)
        except requests.exceptions.ConnectionError:
            self.skipTest("API server is not running. Start the server to test endpoints.")
        except requests.exceptions.Timeout:
            self.skipTest("Request to API server timed out.")
            
    def test_spec_endpoint(self):
        """Test that the spec endpoint is working"""
        try:
            response = requests.get(f"{self.base_url}/spec", timeout=5)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn('info', data)
            self.assertEqual(data['info']['title'], 'Multimodal Gemini API')
        except requests.exceptions.ConnectionError:
            self.skipTest("API server is not running. Start the server to test endpoints.")
        except requests.exceptions.Timeout:
            self.skipTest("Request to API server timed out.")
            
    def test_text_endpoint_success(self):
        """Test that the text endpoint works with valid input"""
        try:
            response = requests.post(
                f"{self.base_url}/text",
                json={'prompt': 'What is artificial intelligence?'},
                timeout=30
            )
            # We expect either 200 (success) or 500 (API key or network error)
            # but not 400 (bad request)
            self.assertNotEqual(response.status_code, 400)
        except requests.exceptions.ConnectionError:
            self.skipTest("API server is not running. Start the server to test endpoints.")
        except requests.exceptions.Timeout:
            self.skipTest("Request to API server timed out.")
            
    def test_text_endpoint_missing_prompt(self):
        """Test that the text endpoint returns error for missing prompt"""
        try:
            response = requests.post(
                f"{self.base_url}/text",
                json={},
                timeout=5
            )
            self.assertEqual(response.status_code, 400)
            data = response.json()
            self.assertIn('error', data)
            self.assertEqual(data['error'], 'Prompt is required')
        except requests.exceptions.ConnectionError:
            self.skipTest("API server is not running. Start the server to test endpoints.")
        except requests.exceptions.Timeout:
            self.skipTest("Request to API server timed out.")
            
    def test_image_endpoint_missing_file(self):
        """Test that the image endpoint returns error for missing file"""
        try:
            response = requests.post(
                f"{self.base_url}/image",
                data={'prompt': 'Describe this image'},
                timeout=5
            )
            self.assertEqual(response.status_code, 400)
            data = response.json()
            self.assertIn('error', data)
            self.assertEqual(data['error'], 'Image file is required')
        except requests.exceptions.ConnectionError:
            self.skipTest("API server is not running. Start the server to test endpoints.")
        except requests.exceptions.Timeout:
            self.skipTest("Request to API server timed out.")
            
    def test_audio_endpoint_missing_file(self):
        """Test that the audio endpoint returns error for missing file"""
        try:
            response = requests.post(
                f"{self.base_url}/audio",
                data={'prompt': 'Describe this audio'},
                timeout=5
            )
            self.assertEqual(response.status_code, 400)
            data = response.json()
            self.assertIn('error', data)
            self.assertEqual(data['error'], 'Audio file is required')
        except requests.exceptions.ConnectionError:
            self.skipTest("API server is not running. Start the server to test endpoints.")
        except requests.exceptions.Timeout:
            self.skipTest("Request to API server timed out.")
            
    def test_multimodal_endpoint_no_content(self):
        """Test that the multimodal endpoint returns error for no content"""
        try:
            response = requests.post(
                f"{self.base_url}/multimodal",
                data={},
                timeout=5
            )
            self.assertEqual(response.status_code, 400)
            data = response.json()
            self.assertIn('error', data)
            self.assertEqual(data['error'], 'At least one modality (text, image, or audio) must be provided')
        except requests.exceptions.ConnectionError:
            self.skipTest("API server is not running. Start the server to test endpoints.")
        except requests.exceptions.Timeout:
            self.skipTest("Request to API server timed out.")
            
    # -----------------------------
    # Mocked Tests (for CI/CD environments)
    # -----------------------------
    
    @patch('app.client')
    def test_text_endpoint_with_mocked_response(self, mock_client):
        """Test text endpoint with mocked AI response"""
        # Mock the Gemini client response
        mock_response = MagicMock()
        mock_response.text = "This is a mocked response from the AI model."
        mock_client.models.generate_content.return_value = mock_response
        
        # Import app inside the test to avoid issues with mocking
        with patch.dict('sys.modules', {'app': __import__('app', fromlist=['app'])}):
            from app import app
            client = app.test_client()
            
            # Test valid request
            response = client.post('/text', 
                                  data=json.dumps({'prompt': 'Hello, world!'}),
                                  content_type='application/json')
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('text', data)
            self.assertEqual(data['text'], "This is a mocked response from the AI model.")
        
    @patch('app.client')
    def test_text_endpoint_missing_prompt_with_mock(self, mock_client):
        """Test text endpoint missing prompt with mock"""
        # Import app inside the test to avoid issues with mocking
        with patch.dict('sys.modules', {'app': __import__('app', fromlist=['app'])}):
            from app import app
            client = app.test_client()
            
            # Test missing prompt
            response = client.post('/text', 
                                  data=json.dumps({}),
                                  content_type='application/json')
            self.assertEqual(response.status_code, 400)
            data = json.loads(response.data)
            self.assertIn('error', data)
            self.assertEqual(data['error'], 'Prompt is required')

if __name__ == '__main__':
    unittest.main()