import ast
import logging

logger = logging.getLogger("mcp.code_validator")
logger.setLevel(logging.INFO)

class CodeValidator:
    def __init__(self):
        self.forbidden_modules = {'os', 'sys', 'subprocess', 'shutil', 'socket', 'requests'}
        self.forbidden_functions = {'eval', 'exec', 'open', '__import__'}

    def is_safe_code(self, code: str) -> bool:
        try:
            # Parse code to AST
            tree = ast.parse(code)
            
            # Check for forbidden imports and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    for name in node.names:
                        if name.name in self.forbidden_modules:
                            logger.warning(f"Detected forbidden module: {name.name}")
                            return False
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in self.forbidden_functions:
                        logger.warning(f"Detected forbidden function: {node.func.id}")
                        return False
                if isinstance(node, ast.Name) and node.id in self.forbidden_functions:
                    logger.warning(f"Detected forbidden name: {node.id}")
                    return False

            # Additional checks for unsafe patterns
            if "import " in code.lower() and any(mod in code.lower() for mod in self.forbidden_modules):
                logger.warning("Detected unsafe import pattern")
                return False

            logger.info("Code passed safety validation")
            return True
        except SyntaxError as e:
            logger.error(f"Code validation error: {str(e)}")
            return False
