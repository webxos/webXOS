// main/server/mcp/functions/notes.js
import { callTool } from './mcp.js';

export async function createNote(title, content, tags = []) {
  try {
    const response = await callTool('create_note', { title, content, tags });
    return response;
  } catch (error) {
    throw new Error(`Failed to create note: ${error.message}`);
  }
}

export async function getNote(noteId) {
  try {
    const response = await fetch('/mcp', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('apiKey')}`,
      },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'mcp.getNote',
        params: { note_id: noteId },
        id: Math.floor(Math.random() * 1000),
      }),
    });
    const data = await response.json();
    if (data.error) throw new Error(data.error.message);
    return data.result;
  } catch (error) {
    throw new Error(`Failed to get note: ${error.message}`);
  }
}

export async function searchNotes(tags = [], query = '') {
  try {
    const response = await fetch('/mcp', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('apiKey')}`,
      },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'mcp.searchNotes',
        params: { tags, query },
        id: Math.floor(Math.random() * 1000),
      }),
    });
    const data = await response.json();
    if (data.error) throw new Error(data.error.message);
    return data.result;
  } catch (error) {
    throw new Error(`Failed to search notes: ${error.message}`);
  }
}
