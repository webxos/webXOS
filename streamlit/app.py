import streamlit as st
import sqlite3
import json

st.title("Vial MCP Visualization")

def fetch_resources(wallet_id):
    """Fetch resources from SQLite database."""
    try:
        with sqlite3.connect("/app/vial_mcp.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id,content,resource_id,timestamp FROM notes WHERE wallet_id=? ORDER BY timestamp DESC LIMIT 10", (wallet_id,))
            return cursor.fetchall()
    except Exception as e:
        st.error(f"Error fetching resources: {str(e)}")
        return []

def fetch_quantum_states(wallet_id):
    """Fetch quantum states from SQLite database."""
    try:
        with sqlite3.connect("/app/vial_mcp.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT vial_id,state,timestamp FROM quantum_states WHERE wallet_id=? ORDER BY timestamp DESC LIMIT 10", (wallet_id,))
            return cursor.fetchall()
    except Exception as e:
        st.error(f"Error fetching quantum states: {str(e)}")
        return []

wallet_id = st.text_input("Wallet ID", value="wallet_123")
if wallet_id:
    st.subheader("Recent Notes")
    notes = fetch_resources(wallet_id)
    for note in notes:
        st.write(f"ID: {note[0]}, Content: {note[1]}, Resource ID: {note[2]}, Timestamp: {note[3]}")
    
    st.subheader("Recent Quantum States")
    states = fetch_quantum_states(wallet_id)
    for state in states:
        st.write(f"Vial ID: {state[0]}, State: {json.loads(state[1])}, Timestamp: {state[2]}")
