class ErrorHandler:
    @staticmethod
    def handle_error(error, context=""):
        error_dict = {
            "code": -32000 if "network" in str(error).lower() or "json" in str(error).lower() else -32603,
            "message": str(error),
            "traceback": context or str(error)
        }
        if "json" in str(error).lower():
            error_dict["message"] = f"JSON parse error: {str(error)}"
        return {"error": error_dict}

    @staticmethod
    def log_error(error, logger):
        logger.error(f"Error: {error['message']}, Traceback: {error['traceback']}")
