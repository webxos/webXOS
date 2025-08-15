class ServiceRegistry:
    def __init__(self):
        self.services = {}

    def register_service(self, name, service):
        self.services[name] = service

    def get_service(self, name):
        return self.services.get(name, None)

# Example service implementations (to be moved to respective modules later)
def checklist_service():
    try:
        # Simulate checklist data
        return {"result": {"status": "OK", "details": "System check completed"}}
    except Exception as e:
        return {"error": {"code": -32603, "message": str(e), "traceback": str(e)}}

def oauth_service(provider, code):
    try:
        # Simulate authentication
        if provider == 'mock' and code == 'test_code':
            return {"access_token": "mock_token", "vials": ["vial1"]}
        raise ValueError("Invalid credentials")
    except Exception as e:
        return {"error": {"code": -32602, "message": str(e), "traceback": str(e)}}

# Register services
registry = ServiceRegistry()
registry.register_service('checklist', checklist_service)
registry.register_service('oauth', oauth_service)
