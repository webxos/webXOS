import re

# Global configuration dictionary
config = {
    'current_step': 'start',
    'format': 'blog',
    'fields': {},  # Format-specific fields (e.g., {'blog': {'title': '', 'subtitle': ''}})
    'bg_color': '#000000',
    'header_font_size': 'medium',
    'text_font_size': 'medium',
    'font': 'Courier New'
}

# Color palette
color_palette = {
    'red': '#FF0000',
    'blue': '#0000FF',
    'green': '#00FF00',
    'black': '#000000',
    'white': '#FFFFFF',
    'gray': '#808080',
    'yellow': '#FFFF00',
    'purple': '#800080',
    'orange': '#FFA500',
    'teal': '#008080'
}

# Font size options
font_sizes = {
    'small': '1.2em',
    'medium': '1.5em',
    'large': '2em'
}

# Supported fonts
font_options = ['Courier New', 'Arial', 'Georgia', 'Times New Roman', 'Verdana']

# Format-specific fields
format_fields = {
    'blog': ['title', 'subtitle', 'main_content', 'cta'],
    'essay': ['title', 'introduction', 'body', 'conclusion'],
    'resume': ['bio', 'job_history', 'skills', 'education'],
    'list': ['title', 'items'],
    'portfolio': ['title', 'projects', 'skills'],
    'journal': ['date', 'entry', 'mood'],
    'recipe': ['title', 'ingredients', 'instructions'],
    'review': ['title', 'rating', 'review_text'],
    'event': ['title', 'date', 'location', 'description'],
    'faq': ['question', 'answer'],
    'tutorial': ['title', 'steps'],
    'news': ['headline', 'byline', 'article'],
    'quote': ['quote_text', 'author'],
    'contact': ['name', 'email', 'phone', 'message']
}

# Format-specific CSS styles (unique, modern, minimal)
format_styles = {
    'blog': 'body { margin: 0; background: {bg_color}; color: #FFFFFF; font-family: "{font}", sans-serif; display: flex; flex-direction: column; align-items: center; padding: 30px; } .content { max-width: 800px; background: #222222; box-shadow: 0 4px 10px rgba(0,0,0,0.2); border-radius: 8px; padding: 20px; transition: box-shadow 0.3s; } .content:hover { box-shadow: 0 6px 14px rgba(0,0,0,0.3); } h1 { font-size: {header_font_size}; margin: 0 0 10px; border-bottom: 2px solid #00FF00; } h2 { font-size: 1.2em; color: #cccccc; } p, .cta { font-size: {text_font_size}; margin: 10px 0; } .cta { font-weight: bold; color: #00FF00; } footer { font-size: 0.9em; color: #aaaaaa; text-align: center; margin-top: 20px; }',
    'essay': 'body { margin: 0; background: {bg_color}; color: #333333; font-family: "{font}", serif; display: grid; grid-template-columns: 1fr 3fr; gap: 15px; padding: 25px; } .content { max-width: 900px; border-left: 1px solid #333333; padding-left: 15px; } h1 { font-size: {header_font_size}; grid-column: span 2; text-align: center; margin-bottom: 15px; } p { font-size: {text_font_size}; line-height: 1.6; } footer { grid-column: span 2; font-size: 0.9em; color: #666666; text-align: center; margin-top: 20px; }',
    'resume': 'body { margin: 0; background: {bg_color}; color: #000000; font-family: "{font}", sans-serif; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; padding: 20px; } .content { background: #f8f8f8; border: 1px solid #000000; border-radius: 5px; padding: 15px; } h1 { font-size: {header_font_size}; grid-column: span 3; text-align: center; margin-bottom: 10px; } section { font-size: {text_font_size}; margin: 10px 0; } footer { grid-column: span 3; font-size: 0.9em; color: #666666; text-align: center; margin-top: 20px; }',
    'list': 'body { margin: 0; background: {bg_color}; color: #333333; font-family: "{font}", monospace; display: flex; flex-direction: column; align-items: flex-start; padding: 25px; } .content { max-width: 700px; border: 1px dotted #333333; border-radius: 5px; padding: 15px; } ul { list-style: none; font-size: {text_font_size}; margin: 10px 0; } li:before { content: "• "; color: #00FF00; transition: color 0.3s; } li:hover:before { color: #FFFF00; } h1 { font-size: {header_font_size}; margin-bottom: 10px; } footer { font-size: 0.9em; color: #666666; text-align: center; margin-top: 20px; }',
    'portfolio': 'body { margin: 0; background: {bg_color}; color: #000000; font-family: "{font}", sans-serif; display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; padding: 20px; } .content { background: #ffffff; box-shadow: 0 2px 8px rgba(0,0,0,0.15); border-radius: 6px; padding: 15px; transition: transform 0.3s; } .content:hover { transform: scale(1.03); } h1 { font-size: {header_font_size}; text-align: center; margin-bottom: 10px; } p { font-size: {text_font_size}; } footer { grid-column: span 3; font-size: 0.9em; color: #666666; text-align: center; margin-top: 20px; }',
    'journal': 'body { margin: 0; background: {bg_color}; color: #333333; font-family: "{font}", serif; display: flex; flex-direction: column; align-items: center; padding: 30px; } .content { max-width: 700px; background: linear-gradient(#f5f5f5, #e8e8e8); border: 1px dashed #333333; border-radius: 6px; padding: 15px; font-style: italic; } h1 { font-size: {header_font_size}; margin-bottom: 10px; } p, .mood { font-size: {text_font_size}; margin: 10px 0; } .mood { color: #666666; } footer { font-size: 0.9em; color: #666666; text-align: center; margin-top: 20px; }',
    'recipe': 'body { margin: 0; background: {bg_color}; color: #000000; font-family: "{font}", monospace; display: grid; grid-template-columns: 1fr 1fr; gap: 15px; padding: 25px; } .content { border: 1px double #000000; border-radius: 5px; padding: 15px; } h1 { font-size: {header_font_size}; grid-column: span 2; text-align: center; margin-bottom: 10px; } p { font-size: {text_font_size}; margin: 10px 0; } footer { grid-column: span 2; font-size: 0.9em; color: #666666; text-align: center; margin-top: 20px; }',
    'review': 'body { margin: 0; background: {bg_color}; color: #333333; font-family: "{font}", sans-serif; display: flex; flex-direction: column; align-items: center; padding: 25px; } .content { max-width: 700px; border: 1px solid #333333; border-radius: 6px; padding: 15px; } h1 { font-size: {header_font_size}; margin-bottom: 10px; } p { font-size: {text_font_size}; margin: 10px 0; } .stars::before { content: attr(data-rating) "★"; color: #FFD700; display: block; text-align: center; transition: transform 0.3s; } .stars:hover::before { transform: scale(1.1); } footer { font-size: 0.9em; color: #666666; text-align: center; margin-top: 20px; }',
    'event': 'body { margin: 0; background: linear-gradient(to bottom, {bg_color}, #2a2a2a); color: #FFFFFF; font-family: "{font}", sans-serif; display: flex; flex-direction: column; align-items: center; padding: 30px; } .content { max-width: 600px; border: 2px solid #FFFFFF; border-radius: 8px; padding: 15px; background: rgba(0,0,0,0.4); } h1 { font-size: {header_font_size}; font-weight: bold; margin-bottom: 10px; } p { font-size: {text_font_size}; margin: 10px 0; } footer { font-size: 0.9em; color: #aaaaaa; text-align: center; margin-top: 20px; }',
    'faq': 'body { margin: 0; background: {bg_color}; color: #333333; font-family: "{font}", monospace; display: flex; flex-direction: column; align-items: center; padding: 25px; } .content { max-width: 700px; border: 1px dashed #333333; border-radius: 5px; padding: 15px; } h1 { font-size: {header_font_size}; margin-bottom: 10px; } p.question { font-size: {text_font_size}; font-weight: bold; margin: 10px 0; } p.answer { font-size: {text_font_size}; margin: 5px 0; } footer { font-size: 0.9em; color: #666666; text-align: center; margin-top: 20px; }',
    'tutorial': 'body { margin: 0; background: {bg_color}; color: #000000; font-family: "{font}", sans-serif; display: flex; flex-direction: column; align-items: center; padding: 25px; } .content { max-width: 700px; border: 1px solid #000000; border-radius: 5px; padding: 15px; } h1 { font-size: {header_font_size}; margin-bottom: 10px; } p { font-size: {text_font_size}; counter-increment: step; margin: 10px 0; } p:before { content: "Step " counter(step) ": "; font-weight: bold; } footer { font-size: 0.9em; color: #666666; text-align: center; margin-top: 20px; }',
    'news': 'body { margin: 0; background: {bg_color}; color: #333333; font-family: "{font}", serif; display: flex; flex-direction: column; align-items: center; padding: 25px; } .content { max-width: 800px; border: 1px solid #333333; border-radius: 5px; padding: 15px; background: #fafafa; } h1 { font-size: {header_font_size}; font-weight: bold; margin-bottom: 10px; } p { font-size: {text_font_size}; margin: 10px 0; } .byline { font-style: italic; text-align: center; } footer { font-size: 0.9em; color: #666666; text-align: center; margin-top: 20px; }',
    'quote': 'body { margin: 0; background: linear-gradient(to bottom, {bg_color}, #1a1a1a); color: #FFFFFF; font-family: "{font}", serif; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; padding: 20px; } .quote { max-width: 600px; text-align: center; border: 1px dashed #FFFFFF; border-radius: 10px; padding: 20px; font-size: {text_font_size}; } h1 { display: none; } .author { font-size: 1.1em; font-style: italic; margin-top: 10px; } footer { font-size: 0.9em; color: #aaaaaa; text-align: center; margin-top: 20px; }',
    'contact': 'body { margin: 0; background: {bg_color}; color: #000000; font-family: "{font}", sans-serif; display: flex; flex-direction: column; align-items: center; padding: 20px; } .content { max-width: 400px; border: 1px solid #000000; border-radius: 6px; padding: 15px; text-align: center; } h1 { font-size: {header_font_size}; margin-bottom: 10px; } p { font-size: {text_font_size}; margin: 10px 0; } footer { font-size: 0.9em; color: #666666; text-align: center; margin-top: 20px; }'
}

# Format-specific content HTML
format_content = {
    'blog': '<h1>{title}</h1><h2>{subtitle}</h2><p>{main_content}</p><p class="cta">{cta}</p>',
    'essay': '<h1>{title}</h1><p><strong>Introduction:</strong> {introduction}</p><p><strong>Body:</strong> {body}</p><p><strong>Conclusion:</strong> {conclusion}</p>',
    'resume': '<h1>Resume</h1><section><strong>Bio:</strong> {bio}</section><section><strong>Job History:</strong> {job_history}</section><section><strong>Skills:</strong> {skills}</section><section><strong>Education:</strong> {education}</section>',
    'list': '<h1>{title}</h1><ul>{items}</ul>',
    'portfolio': '<h1>{title}</h1><p><strong>Projects:</strong> {projects}</p><p><strong>Skills:</strong> {skills}</p>',
    'journal': '<h1>Journal Entry: {date}</h1><p>{entry}</p><p class="mood"><strong>Mood:</strong> {mood}</p>',
    'recipe': '<h1>{title}</h1><p><strong>Ingredients:</strong> {ingredients}</p><p><strong>Instructions:</strong> {instructions}</p>',
    'review': '<h1>{title}</h1><div class="stars" data-rating="{rating}"></div><p>{review_text}</p>',
    'event': '<h1>{title}</h1><p><strong>Date:</strong> {date}</p><p><strong>Location:</strong> {location}</p><p>{description}</p>',
    'faq': '<h1>FAQ</h1><p class="question">{question}</p><p class="answer">{answer}</p>',
    'tutorial': '<h1>{title}</h1>{steps}',
    'news': '<h1>{headline}</h1><p class="byline">{byline}</p><p>{article}</p>',
    'quote': '<p class="quote">{quote_text}</p><p class="author">{author}</p>',
    'contact': '<h1>Contact Info</h1><p><strong>Name:</strong> {name}</p><p><strong>Email:</strong> {email}</p><p><strong>Phone:</strong> {phone}</p><p>{message}</p>'
}

# Help messages and error tooltips
help_messages = {
    'start': "Type 'begin' to create a webpage, 'guide' for help, or 'quit' to exit.\nError Tip: Only these commands are valid.\nPowered by WebXOS: webxos.netlify.app | github.com/webxos/webxos",
    'format': f"Choose a format: {', '.join(format_fields.keys())}.\nError Tip: Enter a valid format name (e.g., 'blog'). Case-insensitive.\nPowered by WebXOS: webxos.netlify.app",
    'blog_title': "Enter blog title (max 200 chars, no <>\"'`\\).\nTip: Use a catchy title to attract readers (e.g., 'Top 10 Tips for 2025').",
    'blog_subtitle': "Enter blog subtitle (max 200 chars, no <>\"'`\\).\nTip: Provide a brief tagline (e.g., 'Your Guide to Success').",
    'blog_main_content': "Enter main content (max 200 chars, use \\n for newlines, no <>\"'`\\).\nTip: Aim for 150+ chars with engaging content.",
    'blog_cta': "Enter call-to-action (max 200 chars, no <>\"'`\\).\nTip: Encourage action (e.g., 'Read more blogs!').",
    'essay_title': "Enter essay title (max 200 chars, no <>\"'`\\).\nTip: Use a clear, specific title (e.g., 'The Future of AI').",
    'essay_introduction': "Enter introduction (max 200 chars, use \\n for newlines, no <>\"'`\\).\nTip: Introduce your topic clearly in 50-100 chars.",
    'essay_body': "Enter body (max 200 chars, use \\n for newlines, no <>\"'`\\).\nTip: Include 2-3 key points or arguments (100+ chars).",
    'essay_conclusion': "Enter conclusion (max 200 chars, use \\n for newlines, no <>\"'`\\).\nTip: Summarize your argument in 50-100 chars.",
    'resume_bio': "Enter bio (max 200 chars, no <>\"'`\\).\nTip: Write a 100-150 char summary of your background.",
    'resume_job_history': "Enter job history (max 200 chars, use \\n for new roles, no <>\"'`\\).\nTip: List 2-3 jobs with role, company, dates (e.g., 'Engineer, XYZ, 2023-2025').",
    'resume_skills': "Enter skills (max 200 chars, comma-separated, no <>\"'`\\).\nTip: List 3-5 key skills (e.g., 'Python, JavaScript, SQL').",
    'resume_education': "Enter education (max 200 chars, no <>\"'`\\).\nTip: Include degree, institution, year (e.g., 'BS Computer Science, MIT, 2023').",
    'list_title': "Enter list title (max 200 chars, no <>\"'`\\).\nTip: Make it descriptive (e.g., 'Top 5 Gadgets').",
    'list_items': "Enter list items (max 200 chars, comma-separated, no <>\"'`\\).\nTip: Provide 3-5 items (e.g., 'Phone, Laptop, Tablet').",
    'portfolio_title': "Enter portfolio title (max 200 chars, no <>\"'`\\).\nTip: Use your name or brand (e.g., 'Jane Doe Portfolio').",
    'portfolio_projects': "Enter projects (max 200 chars, use \\n for new projects, no <>\"'`\\).\nTip: List 2-3 projects with brief descriptions.",
    'portfolio_skills': "Enter skills (max 200 chars, comma-separated, no <>\"'`\\).\nTip: List 3-5 skills (e.g., 'Web Design, UX, Coding').",
    'journal_date': "Enter date (max 200 chars, no <>\"'`\\).\nTip: Use format like '2025-08-01'.",
    'journal_entry': "Enter journal entry (max 200 chars, use \\n for newlines, no <>\"'`\\).\nTip: Write 100-150 chars about your day or thoughts.",
    'journal_mood': "Enter mood (max 200 chars, no <>\"'`\\).\nTip: Describe your mood briefly (e.g., 'Happy', 'Reflective').",
    'recipe_title': "Enter recipe title (max 200 chars, no <>\"'`\\).\nTip: Be specific (e.g., 'Vegan Chocolate Cake').",
    'recipe_ingredients': "Enter ingredients (max 200 chars, comma-separated, no <>\"'`\\).\nTip: List 3-5 ingredients (e.g., 'Flour, Sugar, Cocoa').",
    'recipe_instructions': "Enter instructions (max 200 chars, use \\n for steps, no <>\"'`\\).\nTip: Provide 2-3 clear steps.",
    'review_title': "Enter review title (max 200 chars, no <>\"'`\\).\nTip: Include product or topic (e.g., 'iPhone 15 Review').",
    'review_rating': "Enter rating (1-5).\nTip: Choose a number between 1 and 5 (e.g., '4').",
    'review_review_text': "Enter review text (max 200 chars, use \\n for newlines, no <>\"'`\\).\nTip: Write 100-150 chars with pros and cons.",
    'event_title': "Enter event title (max 200 chars, no <>\"'`\\).\nTip: Be clear (e.g., 'Tech Conference 2025').",
    'event_date': "Enter date (max 200 chars, no <>\"'`\\).\nTip: Use format like '2025-08-01'.",
    'event_location': "Enter location (max 200 chars, no <>\"'`\\).\nTip: Include city or venue (e.g., 'New York, NY').",
    'event_description': "Enter description (max 200 chars, use \\n for newlines, no <>\"'`\\).\nTip: Write 100-150 chars about the event.",
    'faq_question': "Enter question (max 200 chars, no <>\"'`\\).\nTip: Ask a clear question (e.g., 'What is AI?').",
    'faq_answer': "Enter answer (max 200 chars, use \\n for newlines, no <>\"'`\\).\nTip: Provide a concise answer (100-150 chars).",
    'tutorial_title': "Enter tutorial title (max 200 chars, no <>\"'`\\).\nTip: Be specific (e.g., 'Learn Python in 5 Steps').",
    'tutorial_steps': "Enter steps (max 200 chars, comma-separated, no <>\"'`\\).\nTip: List 3-5 steps (e.g., 'Install Python, Write code, Test').",
    'news_headline': "Enter headline (max 200 chars, no <>\"'`\\).\nTip: Make it attention-grabbing (e.g., 'AI Breakthrough in 2025').",
    'news_byline': "Enter byline (max 200 chars, no <>\"'`\\).\nTip: Include author name (e.g., 'By Jane Doe').",
    'news_article': "Enter article (max 200 chars, use \\n for newlines, no <>\"'`\\).\nTip: Write 150+ chars with key details.",
    'quote_quote_text': "Enter quote (max 200 chars, no <>\"'`\\).\nTip: Choose an inspiring quote (100-150 chars).",
    'quote_author': "Enter author (max 200 chars, no <>\"'`\\).\nTip: Include the quote's author (e.g., 'Albert Einstein').",
    'contact_name': "Enter name (max 200 chars, no <>\"'`\\).\nTip: Use full name (e.g., 'Jane Doe').",
    'contact_email': "Enter email (max 200 chars, no <>\"'`\\).\nTip: Use a valid email format (e.g., 'jane@example.com').",
    'contact_phone': "Enter phone (max 200 chars, no <>\"'`\\).\nTip: Use format like '123-456-7890'.",
    'contact_message': "Enter message (max 200 chars, use \\n for newlines, no <>\"'`\\).\nTip: Write a brief message (100-150 chars).",
    'bg_color': f"Choose a background color: {', '.join(color_palette.keys())} or a hex code (e.g., #000000).\nTip: Choose a color that contrasts with text (e.g., dark background with light text).",
    'header_font_size': f"Choose header font size: {', '.join(font_sizes.keys())}.\nTip: Use 'large' for emphasis (e.g., events) or 'medium' for readability.",
    'text_font_size': f"Choose text font size: {', '.join(font_sizes.keys())}.\nTip: Use 'medium' for most formats, 'small' for compact layouts.",
    'font': f"Choose a font: {', '.join(font_options)}.\nTip: Use 'Arial' for professional, 'Georgia' for elegant, or 'Courier New' for code-like text.",
    'export': "Review your inputs. The HTML will be generated next.\nError Tip: If HTML fails, check earlier steps for invalid inputs.\nPowered by WebXOS: webxos.netlify.app"
}

# ML-inspired design recommendations
def recommend_design(format, field, value):
    recommendations = {
        'blog': {'header_font_size': 'large', 'text_font_size': 'medium', 'bg_color': 'dark'},
        'essay': {'header_font_size': 'medium', 'text_font_size': 'medium', 'bg_color': 'light'},
        'resume': {'header_font_size': 'large', 'text_font_size': 'medium', 'bg_color': 'light'},
        'list': {'header_font_size': 'medium', 'text_font_size': 'medium', 'bg_color': 'light'},
        'portfolio': {'header_font_size': 'large', 'text_font_size': 'medium', 'bg_color': 'light'},
        'journal': {'header_font_size': 'medium', 'text_font_size': 'medium', 'bg_color': 'light'},
        'recipe': {'header_font_size': 'large', 'text_font_size': 'medium', 'bg_color': 'light'},
        'review': {'header_font_size': 'medium', 'text_font_size': 'medium', 'bg_color': 'light'},
        'event': {'header_font_size': 'large', 'text_font_size': 'medium', 'bg_color': 'dark'},
        'faq': {'header_font_size': 'medium', 'text_font_size': 'medium', 'bg_color': 'light'},
        'tutorial': {'header_font_size': 'medium', 'text_font_size': 'medium', 'bg_color': 'light'},
        'news': {'header_font_size': 'large', 'text_font_size': 'medium', 'bg_color': 'light'},
        'quote': {'header_font_size': 'medium', 'text_font_size': 'large', 'bg_color': 'dark'},
        'contact': {'header_font_size': 'medium', 'text_font_size': 'medium', 'bg_color': 'light'}
    }
    if field in ['header_font_size', 'text_font_size']:
        rec = recommendations[format][field]
        print(f"Recommended {field}: {rec} for {format}")
    elif field == 'bg_color':
        rec = recommendations[format]['bg_color']
        color_suggestion = 'black or blue' if rec == 'dark' else 'white or gray'
        print(f"Recommended background: {color_suggestion} for {format} (high contrast with text)")
    elif field in format_fields[format]:
        min_length = 100 if field in ['bio', 'main_content', 'body', 'entry', 'review_text', 'description', 'article', 'quote_text', 'message'] else 50
        if len(value) < min_length:
            print(f"Tip: For {field}, aim for {min_length}+ chars to provide sufficient detail.")

# Input validation
def validate_input(text, field):
    try:
        if field == 'bg_color':
            if text.lower() in color_palette:
                return True
            if not re.match(r'^#[0-9A-Fa-f]{6}$', text):
                print(f"Invalid {field}: Choose from {', '.join(color_palette.keys())} or use a hex code (e.g., #000000)")
                return False
        elif field in ['header_font_size', 'text_font_size']:
            if text.lower() not in font_sizes:
                print(f"Invalid {field}: Choose from {', '.join(font_sizes.keys())}")
                return False
        elif field == 'font':
            if text.lower() not in [f.lower() for f in font_options]:
                print(f"Invalid font: Choose from {', '.join(font_options)}")
                return False
        elif field == 'review_rating':
            try:
                rating = int(text)
                if rating < 1 or rating > 5:
                    print("Invalid rating: Choose a number between 1 and 5")
                    return False
            except ValueError:
                print("Invalid rating: Enter a number (e.g., '4')")
                return False
        elif field in [f for fmt in format_fields.values() for f in fmt]:
            if len(text) > 200:
                print(f"Invalid {field}: Must be 200 characters or less")
                return False
            if re.search(r'[<>\'"`\\]', text):
                print(f"Invalid {field}: No HTML tags, quotes, or backticks allowed")
                return False
        elif field == 'format':
            valid_formats = format_fields.keys()
            if text.lower() not in valid_formats:
                print(f"Invalid format: Choose from {', '.join(valid_formats)}")
                return False
        return True
    except Exception as e:
        print(f"Validation error: {str(e)}\nError Tip: Follow the instructions for this step.")
        return False

# HTML template (text-only with footer)
html_template = [
    '<!DOCTYPE html>',
    '<html lang="en">',
    '<head>',
    '  <meta charset="UTF-8">',
    '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
    '  <title>{title}</title>',
    '  <style>',
    '    {format_style}',
    '  </style>',
    '</head>',
    '<body>',
    '  <div class="{content_class}">{content_html}</div>',
    '  <footer>© 2025 WebXOS | Powered by <a href="https://webxos.netlify.app">webxos.netlify.app</a> | Source: <a href="https://github.com/webxos/webxos">github.com/webxos/webxos</a></footer>',
    '</body>',
    '</html>'
]

def print_intro():
    print("=== WebXOS Webpage Creator ===")
    print("Create modern, text-only webpages with unique, minimal designs.")
    print("© 2025 WebXOS | Powered by webxos.netlify.app | Source: github.com/webxos/webxos")
    print("Type 'begin' to start, 'guide' for help, or 'quit' to exit.")
    print("=============================")

def print_guide():
    print("=== WebXOS Webpage Creator Guide ===")
    print("© 2025 WebXOS | Powered by webxos.netlify.app | Source: github.com/webxos/webxos")
    print("Commands:")
    print("- begin: Start the step-by-step webpage creation process.")
    print("- guide: Show this help message.")
    print("- quit: Exit the program.")
    print("Steps (when you type 'begin'):")
    print("1. Choose format (e.g., blog, essay, resume, etc.)")
    print("2. Enter format-specific fields (e.g., for resume: bio, job history, skills, education)")
    print("3. Set background color (palette name or hex, e.g., red or #000000)")
    print("4. Set header font size (small, medium, large)")
    print("5. Set text font size (small, medium, large)")
    print("6. Choose font (e.g., Arial, Georgia)")
    print("7. View and copy the final HTML")
    print("Copy the HTML output, paste into a .html file, and open in a browser.")
    print("If errors occur, use 'Eject' in the console and type 'begin' to restart.")
    print("===================================")

def export_html():
    try:
        # Validate inputs
        if not all(validate_input(config[field], field) for field in ['format', 'bg_color', 'header_font_size', 'text_font_size', 'font']):
            print("Export aborted: Invalid inputs detected.\nError Tip: Go back to 'begin' and check each step.")
            return None
        for field in format_fields[config['format']]:
            if field not in config['fields'] or not validate_input(config['fields'][field], field):
                print(f"Invalid {field}: Ensure all fields are valid.\nError Tip: Go back to 'begin' and check each step.")
                return None
        # Safer string escaping
        fields = {k: v.replace("'", "\\'").replace('"', '\\"').replace('\n', '<br>') for k, v in config['fields'].items()}
        if config['format'] == 'list':
            items = fields.get('items', '').split(',')
            fields['items'] = ''.join(f'<li>{item.strip()}</li>' for item in items if item.strip())
        elif config['format'] == 'tutorial':
            steps = fields.get('steps', '').split(',')
            fields['steps'] = ''.join(f'<p>{step.strip()}</p>' for step in steps if step.strip())
        bg_color = color_palette.get(config['bg_color'].lower(), config['bg_color'])
        header_font_size = font_sizes[config['header_font_size'].lower()]
        text_font_size = font_sizes[config['text_font_size'].lower()]
        format_style = format_styles[config['format']].replace('{bg_color}', bg_color).replace('{font}', config['font']).replace('{header_font_size}', header_font_size).replace('{text_font_size}', text_font_size)
        content_html = format_content[config['format']].format(**fields)
        content_class = 'content' if config['format'] != 'quote' else 'quote'
        title = fields.get('title', fields.get('headline', fields.get('name', 'WebXOS Webpage')))
        # Build HTML incrementally
        html_lines = []
        for line in html_template:
            try:
                html_lines.append(line.replace('{format_style}', format_style)
                                     .replace('{title}', title)
                                     .replace('{content_html}', content_html)
                                     .replace('{content_class}', content_class))
            except Exception as e:
                print(f"Error processing HTML line: {str(e)}\nError Tip: Ensure all inputs are valid (e.g., no special characters).")
                return None
        return '\n'.join(html_lines)
    except Exception as e:
        print(f"Export error: {str(e)}\nError Tip: Use simple inputs (e.g., 'Test', 'Hello'), avoid special characters.")
        return None

def main():
    try:
        # Ensure current_step is initialized
        if 'current_step' not in config:
            config['current_step'] = 'start'
            print("Initialized current_step to 'start'")
        print_intro()
        current_field_index = 0
        while True:
            print(f"\n{help_messages[config['current_step']]}")
            if config['current_step'] == 'start':
                print("Enter command: ")
                command = input().lower()
                if command == 'begin':
                    config['current_step'] = 'format'
                    config['fields'] = {}
                    current_field_index = 0
                    print(f"Step changed to: {config['current_step']}")
                elif command == 'guide':
                    print_guide()
                elif command == 'quit':
                    print("Exiting WebXOS Webpage Creator.")
                    break
                else:
                    print("Invalid command: Use 'begin', 'guide', or 'quit'.\nError Tip: Only these commands are valid.")
            elif config['current_step'] == 'format':
                print("Enter format: ")
                value = input().lower()
                if validate_input(value, 'format'):
                    config['format'] = value
                    config['current_step'] = f"{value}_{format_fields[value][0]}"
                    current_field_index = 0
                    print(f"Step changed to: {config['current_step']}")
                    recommend_design(value, 'format', value)
                else:
                    print("Try again.")
            elif config['current_step'].startswith(tuple(format_fields.keys())):
                current_format = config['current_step'].split('_')[0]
                field = format_fields[current_format][current_field_index]
                print(f"Enter {field}: ")
                value = input()
                if validate_input(value, field):
                    config['fields'][field] = value
                    recommend_design(current_format, field, value)
                    current_field_index += 1
                    if current_field_index < len(format_fields[current_format]):
                        config['current_step'] = f"{current_format}_{format_fields[current_format][current_field_index]}"
                        print(f"Step changed to: {config['current_step']}")
                    else:
                        config['current_step'] = 'bg_color'
                        print(f"Step changed to: {config['current_step']}")
                else:
                    print("Try again.")
            elif config['current_step'] == 'bg_color':
                print("Enter background color: ")
                value = input().lower()
                if validate_input(value, 'bg_color'):
                    config['bg_color'] = value
                    recommend_design(config['format'], 'bg_color', value)
                    config['current_step'] = 'header_font_size'
                    print(f"Step changed to: {config['current_step']}")
                else:
                    print("Try again.")
            elif config['current_step'] == 'header_font_size':
                print("Enter header font size: ")
                value = input().lower()
                if validate_input(value, 'header_font_size'):
                    config['header_font_size'] = value
                    recommend_design(config['format'], 'header_font_size', value)
                    config['current_step'] = 'text_font_size'
                    print(f"Step changed to: {config['current_step']}")
                else:
                    print("Try again.")
            elif config['current_step'] == 'text_font_size':
                print("Enter text font size: ")
                value = input().lower()
                if validate_input(value, 'text_font_size'):
                    config['text_font_size'] = value
                    recommend_design(config['format'], 'text_font_size', value)
                    config['current_step'] = 'font'
                    print(f"Step changed to: {config['current_step']}")
                else:
                    print("Try again.")
            elif config['current_step'] == 'font':
                print("Enter font: ")
                value = input()
                if validate_input(value, 'font'):
                    config['font'] = value
                    config['current_step'] = 'export'
                    print(f"Step changed to: {config['current_step']}")
                else:
                    print("Try again.")
            elif config['current_step'] == 'export':
                html = export_html()
                if html:
                    print('=== Generated HTML (copy and paste into a .html file) ===')
                    print(html)
                    print('=== End of HTML ===')
                    print("Webpage creation complete. Start over with 'begin' or 'quit'.")
                    config['current_step'] = 'start'
                    print(f"Step changed to: {config['current_step']}")
                else:
                    print("HTML generation failed. Start over with 'begin' or 'quit'.\nError Tip: Check inputs for errors (e.g., invalid colors or special characters).")
                    config['current_step'] = 'start'
                    print(f"Step changed to: {config['current_step']}")
            else:
                print(f"Invalid step: {config['current_step']}. Resetting to 'start'.\nError Tip: Restart with 'begin'.")
                config['current_step'] = 'start'
                print(f"Step changed to: {config['current_step']}")
    except Exception as e:
        print(f"Main loop error: {str(e)}\nError Tip: Reset the console with 'Eject' and type 'begin' to start again.")
        config['current_step'] = 'start'
        print(f"Step reset to: {config['current_step']}")

# Run the script
main()
