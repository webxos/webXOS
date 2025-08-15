class ErrorHandler:
    @staticmethod
    def handle_error(error, context=""):
        error_dict = {
            "code": -32000 if "network" in str(error).lower() or "json" in str(error).lower() else -32603,
            "message": str(error) if not isinstance(error, dict) else error.get("message", str(error)),
            "traceback": context or str(error)
        }
        return {"error": error_dict}

    @staticmethod
    def log_error(error, logger=None):
        if logger:
            logger.error(f"Error: {error.get('message', 'Unknown')}, Traceback: {error.get('traceback', 'None')}")
        else:
            print(f"Error: {error.get('message', 'Unknown')}, Traceback: {error.get('traceback', 'None')}")
