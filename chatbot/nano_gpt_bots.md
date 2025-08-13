Agent 1
Role: Chatbot VialDescription: A conversational AI for natural language interactions, answering queries with context-aware responses.
def generate_response(query, wallet):
    wallet["transactions"].append({
        "type": "chatbot_query",
        "query": query,
        "timestamp": "2025-08-13T02:11:00Z"
    })
    wallet["webxos"] += 0.0001
    return f"Chatbot Vial: Responding to '{query}' with WebXOS balance {wallet['webxos']:.4f}"

Agent 2
Role: Data Analyst VialDescription: Analyzes data queries, providing insights or summaries based on input.
def generate_response(query, wallet):
    wallet["transactions"].append({
        "type": "data_query",
        "query": query,
        "timestamp": "2025-08-13T02:11:00Z"
    })
    wallet["webxos"] += 0.0001
    keywords = query.lower().split()
    return f"Data Analyst Vial: Analyzed '{query}' (keywords: {', '.join(keywords)}) with WebXOS balance {wallet['webxos']:.4f}"

Agent 3
Role: Code Generator VialDescription: Generates lightweight, standalone code snippets for web development tasks.
def generate_response(query, wallet):
    wallet["transactions"].append({
        "type": "code_query",
        "query": query,
        "timestamp": "2025-08-13T02:11:00Z"
    })
    wallet["webxos"] += 0.0001
    if "html" in query.lower():
        code = "<div>Hello, WebXOS!</div>"
    elif "javascript" in query.lower():
        code = "console.log('Hello, WebXOS!');"
    else:
        code = "# Sample Python code\nprint('Hello, WebXOS!')"
    return f"Code Generator Vial: Generated code for '{query}':\n```code\n{code}\n``` with WebXOS balance {wallet['webxos']:.4f}"

Agent 4
Role: Galaxy Web Crawler VialDescription: Simulates web crawling to fetch relevant information for search queries.
def generate_response(query, wallet):
    wallet["transactions"].append({
        "type": "web_crawl",
        "query": query,
        "timestamp": "2025-08-13T02:11:00Z"
    })
    wallet["webxos"] += 0.0001
    return f"Galaxy Web Crawler Vial: Crawled web for '{query}', found relevant data with WebXOS balance {wallet['webxos']:.4f}"
