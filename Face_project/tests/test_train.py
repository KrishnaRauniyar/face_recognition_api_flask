import unittest
from app import app
import io
# endpoint testing of /train
class ImageLoadTest(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_successful_ImageUpload(self):
        data = {
            'file': (io.BytesIO(b"some random data"), "1_krishna.jpg")
        }
        response = self.app.post('/train', headers={"Content-Type": 'multipart/form-data'},
                                 data= data)
        self.assertEqual(200, response.status_code)