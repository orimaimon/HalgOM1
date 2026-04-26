import argparse
import sys
import pyperclip
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from snippet_db import init_db, add_snippet, search_snippets, get_snippet_by_id

console = Console()

def main():
    parser = argparse.ArgumentParser(description="🚀 HalgOM Code Snippet Organizer")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # פקודת אתחול
    subparsers.add_parser("init", help="Initialize the database")

    # פקודת הוספה
    parser_add = subparsers.add_parser("add", help="Add a new snippet")
    parser_add.add_argument("-t", "--title", required=True, help="Snippet title")
    parser_add.add_argument("-d", "--desc", default="", help="Snippet description")
    parser_add.add_argument("-c", "--code", required=True, help="The code content")
    parser_add.add_argument("-l", "--lang", default="python", help="Programming language (default: python)")
    parser_add.add_argument("--tags", default="", help="Comma separated tags")

    # פקודת חיפוש
    parser_search = subparsers.add_parser("search", help="Search snippets")
    parser_search.add_argument("query", help="Search keywords (e.g., 'monte carlo')")

    # פקודת צפייה והעתקה
    parser_view = subparsers.add_parser("view", help="View a snippet and copy to clipboard")
    parser_view.add_argument("id", help="Full or partial Snippet ID")

    args = parser.parse_args()

    if args.command == "init":
        init_db()
        console.print("[bold green]✔ Database 'snippets.db' initialized successfully.[/bold green]")

    elif args.command == "add":
        try:
            snippet_id = add_snippet(args.title, args.desc, args.code, args.lang, args.tags)
            short_id = snippet_id[:8]
            console.print(f"[bold green]✔ Snippet added![/bold green] ID: [cyan]{short_id}[/cyan]")
        except Exception as e:
            console.print(f"[bold red]Error adding snippet: {e}[/bold red]")

    elif args.command == "search":
        results = search_snippets(args.query)
        if not results:
            console.print(f"[yellow]No results found for '{args.query}'.[/yellow]")
            return
        
        table = Table(title=f"Search Results: '{args.query}'")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Title", style="magenta")
        table.add_column("Description", style="white")
        table.add_column("Lang", style="green")

        for row in results:
            short_id = row['id'][:8]
            table.add_row(short_id, row['title'], row['description'], row['language'])
        
        console.print(table)
        console.print("\n[dim]Use 'python snippet_cli.py view <ID>' to view & copy code.[/dim]")

    elif args.command == "view":
        row = get_snippet_by_id(args.id)
        if not row:
            console.print(f"[bold red]✖ Snippet with ID starting with '{args.id}' not found.[/bold red]")
            return
        
        console.print(f"\n[bold magenta]Title:[/bold magenta] {row['title']}")
        if row['description']:
            console.print(f"[bold magenta]Description:[/bold magenta] {row['description']}")
        console.print("")
        
        # הדגשת קוד ויזואלית
        syntax = Syntax(row['code'], row['language'], theme="monokai", line_numbers=True)
        console.print(syntax)
        
        # העתקה אוטומטית ללוח
        try:
            pyperclip.copy(row['code'])
            console.print("\n[bold green]✔ Code copied to clipboard![/bold green]\n")
        except pyperclip.PyperclipException:
            console.print("\n[yellow]⚠ Could not access clipboard. Please copy manually.[/yellow]\n")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()