// main/server/mcp/functions/notes.js
import { callTool } from './mcp.js';

export async function createNote(userId, title, content, tags = []) {
  try {
    const response = await fetch('/mcp', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('apiKey')}`,
      },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'mcp.createNote',
        params: { user_id: userId, title, content, tags },
        id: Math.floor(Math.random() * 1000),
      }),
    });
    const data = await response.json();
    if (data.error) throw new Error(data.error.message);
    return data.result;
  } catch (error) {
    throw new Error(`Failed to create note: ${error.message}`);
  }
}

export async function addSubNote(noteId, content) {
  try {
    const response = await callTool('add_sub_note', { note_id: noteId, content });
    return response;
  } catch (error) {
    throw new Error(`Failed to add sub-note: ${error.message}`);
  }
}

export async function listSubNotes(noteId) {
  try {
    const response = await callTool('list_sub_notes', { note_id: noteId });
    return response;
  } catch (error) {
    throw new Error(`Failed to list sub-notes: ${error.message}`);
  }
}

export async function searchNotes(userId, query, tags = []) {
  try {
    const response = await callTool('search_notes', { user_id: userId, query, tags });
    return response;
  } catch (error) {
    throw new Error(`Failed to search notes: ${error.message}`);
  }
}
