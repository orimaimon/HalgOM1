import sqlite3
import uuid
from datetime import datetime

DB_PATH = "snippets.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    """מאתחל את בסיס הנתונים והטבלאות הנדרשות"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # טבלת קטעי הקוד הראשית
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS snippets (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT,
            code TEXT NOT NULL,
            language TEXT DEFAULT 'python',
            created_at TEXT NOT NULL
        )
    ''')
    
    # טבלה וירטואלית עבור Full-Text Search (FTS5)
    cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS snippets_fts USING fts5(
            title, description, code, tags
        )
    ''')
    
    # טבלת תגיות (לשימוש עתידי לסינון מתקדם)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS snippet_tags (
            snippet_id TEXT,
            tag TEXT,
            FOREIGN KEY(snippet_id) REFERENCES snippets(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def add_snippet(title, description, code, language="python", tags=""):
    """מוסיף קטע קוד חדש ומתנדקס אותו במנוע החיפוש"""
    conn = get_connection()
    cursor = conn.cursor()
    
    snippet_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
    # הוספה לטבלה הראשית
    cursor.execute('''
        INSERT INTO snippets (id, title, description, code, language, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (snippet_id, title, description, code, language, created_at))
    
    # עיבוד תגיות
    tags_str = ""
    if tags:
        tags_list = [t.strip() for t in tags.split(",")]
        tags_str = " ".join(tags_list)
        for tag in tags_list:
            cursor.execute('INSERT INTO snippet_tags (snippet_id, tag) VALUES (?, ?)', (snippet_id, tag))
    
    # הוספה למנוע החיפוש
    cursor.execute('''
        INSERT INTO snippets_fts (rowid, title, description, code, tags)
        VALUES (?, ?, ?, ?, ?)
    ''', (cursor.lastrowid, title, description, code, tags_str))
    
    conn.commit()
    conn.close()
    return snippet_id

def search_snippets(query):
    """מחפש קטעי קוד על בסיס מילות מפתח"""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # שימוש ב-MATCH לחיפוש מהיר
    cursor.execute('''
        SELECT s.id, s.title, s.description, s.language, s.created_at
        FROM snippets_fts f
        JOIN snippets s ON s.rowid = f.rowid
        WHERE snippets_fts MATCH ?
        ORDER BY rank
    ''', (query,))
    
    results = cursor.fetchall()
    conn.close()
    return results

def get_snippet_by_id(partial_id):
    """מחזיר קטע קוד מלא לפי תחילית של ה-ID"""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM snippets WHERE id LIKE ?', (f"{partial_id}%",))
    result = cursor.fetchone()
    conn.close()
    return result