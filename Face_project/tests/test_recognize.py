import unittest
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Face_project/recognize')

from recognize import Recognize

### Testing of recognize

class TestStringMethods(unittest.TestCase):
    # test function to test equality of two value
    def test_successful_VideoUpload(self):
        obj = Recognize()
        result = obj.faceRecognize('krishna_video.mp4')
        # assertEqual() to check equality of first & second value
        self.assertEqual('1_krishna', result)

if __name__ == '__main__':
    unittest.main()
