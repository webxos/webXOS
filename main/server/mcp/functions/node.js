// main/server/mcp/functions/notes.js
async function createNote(title, content, tags = []) {
  const token = localStorage.getItem('apiKey');
  const userId = localStorage.getItem('userId');
  if (!token || !userId) throw new Error('Not authenticated');
  const response = await fetch(`${process.env.API_BASE || 'http://localhost:8000'}/notes`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify({ user_id: userId, title, content, tags })
  });
  if (!response.ok) throw new Error(`Failed to create note: ${await response.text()}`);
  return await response.json();
}

async function getNotes() {
  const token = localStorage.getItem('apiKey');
  const userId = localStorage.getItem('userId');
  if (!token || !userId) throw new Error('Not authenticated');
  const response = await fetch(`${process.env.API_BASE || 'http://localhost:8000'}/notes/${userId}`, {
    method: 'GET',
    headers: { 'Authorization': `Bearer ${token}` }
  });
  if (!response.ok) throw new Error(`Failed to fetch notes: ${await response.text()}`);
  return await response.json();
}

async function deleteNote(noteId) {
  const token = localStorage.getItem('apiKey');
  if (!token) throw new Error('Not authenticated');
  const response = await fetch(`${process.env.API_BASE || 'http://localhost:8000'}/notes/${noteId}`, {
    method: 'DELETE',
    headers: { 'Authorization': `Bearer ${token}` }
  });
  if (!response.ok) throw new Error(`Failed to delete note: ${await response.text()}`);
  return { status: 'success' };
}

export { createNote, getNotes, deleteNote };
