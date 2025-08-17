from main.api.utils.logging import logger

class DataValidator:
    @staticmethod
    def validate_vial_id(vial_id: str):
        """Validate vial ID format."""
        valid_ids = [f"vial{i+1}" for i in range(4)]
        if vial_id not in valid_ids:
            logger.error(f"Invalid vial_id: {vial_id}")
            raise ValueError(f"Invalid vial_id: {vial_id}. Must be one of {valid_ids}")
        return vial_id

    @staticmethod
    def validate_dataset(dataset: dict):
        """Validate dataset structure."""
        try:
            if not isinstance(dataset, dict):
                raise ValueError("Dataset must be a dictionary")
            if not dataset:
                logger.warning("Empty dataset provided")
            return dataset
        except Exception as e:
            logger.error(f"Dataset validation failed: {str(e)}")
            raise
