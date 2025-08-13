import re
import json

def enhance_query(query):
    try:
        # Basic query enhancement (expand as needed)
        query = query.lower().strip()
        keywords = re.findall(r'\w+', query)
        enhanced = ' '.join(keywords)
        return enhanced
    except Exception as e:
        return query
