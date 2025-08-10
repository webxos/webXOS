import uuid

class VectorContainer:
    def chunk_md(self, content):
        try:
            return content.split('\n\n')
        except Exception as e:
            raise ValueError(f"Chunking error: {str(e)}")

    def add_metadata(self, chunk, metadata):
        return f"<!-- Metadata: {json.dumps(metadata)} -->\n{chunk}"

    def vectorize(self, chunk):
        return f"vector_{uuid.uuid4()}"
