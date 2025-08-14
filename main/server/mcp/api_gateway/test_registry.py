# server/mcp/api_gateway/test_service_registry.py
import unittest
from .service_registry import ServiceRegistry

class TestServiceRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ServiceRegistry()

    def test_register_and_get_service(self):
        self.registry.register_service("agent1", "/api/agent1", {"type": "search"})
        service = self.registry.get_service("agent1")
        self.assertEqual(service["endpoint"], "/api/agent1")
        self.assertEqual(service["metadata"]["type"], "search")

    def test_deregister_service(self):
        self.registry.register_service("agent1", "/api/agent1")
        self.registry.deregister_service("agent1")
        self.assertIsNone(self.registry.get_service("agent1"))

    def test_list_services(self):
        self.registry.register_service("agent1", "/api/agent1")
        self.registry.register_service("agent2", "/api/agent2")
        services = self.registry.list_services()
        self.assertEqual(len(services), 2)
        self.assertIn("agent1", services)
        self.assertIn("agent2", services)

if __name__ == "__main__":
    unittest.main()
