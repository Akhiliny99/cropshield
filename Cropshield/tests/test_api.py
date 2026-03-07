
from fastapi.testclient import TestClient
import sys, os, json
import numpy as np
from PIL import Image
import io


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


import unittest.mock as mock

mock_session = mock.MagicMock()
mock_output  = np.zeros((1, 38), dtype=np.float32)
mock_output[0][0] = 5.0 
mock_session.run.return_value = [mock_output]
mock_session.get_inputs.return_value = [mock.MagicMock(name="input")]

with mock.patch("onnxruntime.InferenceSession", return_value=mock_session), \
     mock.patch("builtins.open", mock.mock_open(
         read_data=json.dumps([f"Class_{i}" for i in range(38)]))):
    from main import app

client = TestClient(app)

def test_root():
    """Test root endpoint returns API info"""
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "name" in data
    assert data["classes"] == 38
    print("✅ test_root passed")

def test_health():
    """Test health endpoint"""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    print("✅ test_health passed")

def test_classes():
    """Test classes endpoint returns 38 classes"""
    r = client.get("/classes")
    assert r.status_code == 200
    assert r.json()["total"] == 38
    print("✅ test_classes passed")

def test_predict_valid_image():
    """Test predict endpoint with a valid image"""
    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    r = client.post("/predict",
                    files={"file": ("test.jpg", buf, "image/jpeg")})
    assert r.status_code == 200
    data = r.json()
    assert "predicted_class" in data
    assert "confidence"      in data
    assert "treatment"       in data
    assert "inference_ms"    in data
    assert len(data["top5"]) == 5
    print("✅ test_predict_valid_image passed")

def test_predict_invalid_file():
    """Test predict rejects non-image files"""
    r = client.post("/predict",
                    files={"file": ("test.txt", b"not an image", "text/plain")})
    assert r.status_code == 400
    print("✅ test_predict_invalid_file passed")
