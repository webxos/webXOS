ROUTES = {
    "api": {
        "auth": {"oauth": "/api/auth/oauth"},
        "troubleshoot": "/api/troubleshoot",
        "health": "/api/health",
        "validate": "/api/validate",
        "404": "/api/404"
    },
    "vial2": {
        "api": {key: f"/vial2/api/{key}" for key in ["auth/oauth", "troubleshoot", "health"]}
    }
}

def get_route(service, action):
    return ROUTES.get(service, {}).get(action, f"/api/{action}")
