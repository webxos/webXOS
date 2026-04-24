// tools/markdown_kb.js
const path = require('path');
const fs = require('fs');
const Database = require('better-sqlite3');
const { Index } = require('flexsearch');
const matter = require('gray-matter');

const KB_DIR = process.env.MARKDOWN_KB_DIR || path.join(process.cwd(), 'knowledge');
const DB_PATH = path.join(KB_DIR, 'index.db');

// Ensure knowledge directory exists
if (!fs.existsSync(KB_DIR)) {
    fs.mkdirSync(KB_DIR, { recursive: true });
}

// Initialize SQLite database
const db = new Database(DB_PATH);
db.exec(`
    CREATE TABLE IF NOT EXISTS notes (
        id TEXT PRIMARY KEY,
        title TEXT,
        content TEXT,
        file_path TEXT,
        updated_at INTEGER
    );
`);

// Full-text search index
const index = new Index({ tokenize: 'forward', resolution: 9 });

// Helper: Sanitize title to a safe filename
function titleToFilename(title) {
    return title.replace(/[^a-zA-Z0-9_\-]/g, '_') + '.md';
}

// Rebuild index from all .md files in KB_DIR
function rebuildIndex() {
    const files = fs.readdirSync(KB_DIR).filter(f => f.endsWith('.md'));
    // Clear existing data
    db.prepare('DELETE FROM notes').run();
    // Re-index each file
    files.forEach(file => {
        const fullPath = path.join(KB_DIR, file);
        const raw = fs.readFileSync(fullPath, 'utf8');
        const { data, content } = matter(raw);
        const title = data.title || file.replace(/\.md$/, '');
        const id = file;

        db.prepare('INSERT OR REPLACE INTO notes VALUES (?, ?, ?, ?, ?)').run(
            id, title, content, fullPath, Date.now()
        );
        index.add(id, `${title} ${content}`);
    });
    return { success: true, indexed: files.length };
}

// Search the knowledge base
function search(query, limit = 5) {
    const results = index.search(query, limit);
    if (!results.length) return [];
    // Fetch full data for each result
    const stmt = db.prepare('SELECT title, content, file_path FROM notes WHERE id = ?');
    return results.map(id => {
        const row = stmt.get(id);
        return {
            title: row.title,
            snippet: row.content.slice(0, 300),
            path: row.file_path
        };
    });
}

// Write a new note (creates or overwrites)
function writeNote(title, content) {
    const filename = titleToFilename(title);
    const fullPath = path.join(KB_DIR, filename);
    // Add YAML front matter with title
    const fm = matter(content);
    fm.data.title = title;
    const finalContent = matter.stringify(fm.content, fm.data);
    fs.writeFileSync(fullPath, finalContent, 'utf8');
    // Re-index to include the new/updated note
    rebuildIndex();
    return { path: fullPath, title, success: true };
}

// Read a note by its file path (absolute or relative to KB_DIR)
function readNoteByPath(filePath) {
    const absolutePath = path.isAbsolute(filePath) ? filePath : path.join(KB_DIR, filePath);
    if (!fs.existsSync(absolutePath)) throw new Error(`File not found: ${absolutePath}`);
    const raw = fs.readFileSync(absolutePath, 'utf8');
    const { data, content } = matter(raw);
    return { title: data.title || path.basename(absolutePath, '.md'), content, path: absolutePath };
}

// Read a note by its title (converts title to filename)
function readNoteByTitle(title) {
    const filename = titleToFilename(title);
    const fullPath = path.join(KB_DIR, filename);
    return readNoteByPath(fullPath);
}

// List all notes with metadata
function listNotes() {
    const stmt = db.prepare('SELECT title, file_path, updated_at FROM notes ORDER BY title');
    return stmt.all();
}

module.exports = {
    search,
    writeNote,
    readNoteByPath,
    readNoteByTitle,
    rebuildIndex,
    listNotes
};
