import unittest
from .service_registry import ServiceRegistry

class TestServiceRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ServiceRegistry()

    def test_register_and_get_service(self):
        def mock_service():
            return "test"
        self.registry.register_service('test', mock_service)
        self.assertEqual(self.registry.get_service('test')(), "test")

    def test_get_nonexistent_service(self):
        self.assertIsNone(self.registry.get_service('nonexistent'))

if __name__ == '__main__':
    unittest.main()
