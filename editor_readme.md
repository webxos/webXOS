WebXOS Editor IDE
WebXOS Editor is a lightweight, distraction-free web-based Integrated Development Environment (IDE) that supports Markdown editing with real-time preview and basic drawing capabilities. It provides a clean interface for writing, formatting, and sketching, with features like document statistics, export options, and customizable text formatting.
Features

Markdown Editing: Write in Markdown with real-time preview of formatted content.
Drawing Mode: Create simple sketches with tools like pencil, spray paint, shapes, and eraser.
Document Statistics: Track word count, characters, lines, paragraphs, sentences, pages, and estimated reading time.
Export Options: Save your work as Markdown (.md), PDF (.pdf), or JSON (.json).
Text Formatting: Customize font, font size, text color, and alignment.
Drawing Tools: Includes rulers, grid overlay, and templates for quick sketches (e.g., house plans, room layouts, flowcharts).
Autosave: Automatically saves your Markdown content to localStorage.
Responsive Design: Adapts to various screen sizes, with a mobile-friendly interface.
Fullscreen Mode: Toggle fullscreen for a distraction-free experience.
Cheat Sheets: Quick reference guides for Markdown syntax and drawing tools.

Getting Started

Open the Editor: Load editor.html in a modern web browser.
Write Mode:
Use the left panel to write Markdown.
See real-time formatted output in the right panel.
Use the toolbar to adjust font, size, color, and alignment.
Enable spell check or clear the editor as needed.


Draw Mode:
Switch to Draw mode using the header button.
Select tools (pencil, spray, shapes, eraser) and adjust brush size/color.
Use rulers and grid overlay for precise drawings.
Choose from templates like house plans or flowcharts.


Export:
Click the "Export" button to open the export modal.
Choose Markdown, PDF, or JSON format and confirm to download.


Statistics: View document or drawing stats in the sidebar.
Cheat Sheets:
Access Markdown syntax guide via the question mark button.
Access drawing templates and tool guide via the shapes button.



Usage
Write Mode

Markdown Syntax: Supports headers, bold/italic text, lists, links, code blocks, blockquotes, and more (see cheat sheet).
Formatting:
Select from fonts: Default, Times New Roman, Arial, Arcade 80s.
Adjust font sizes: 12px to 24px.
Change text color using the color picker.
Align text left, center, or right.


Statistics: Automatically updates word count, characters, lines, etc.
Autosave: Content is saved to localStorage on every change.
Export:
Markdown: Downloads raw Markdown text.
PDF: Generates a PDF with Helvetica font and standard margins.
JSON: Exports content with metadata (word count, export date, etc.).



Draw Mode

Tools:
Pencil: Freehand drawing with adjustable brush size.
Spray Paint: Airbrush effect with customizable density.
Shapes: Draw lines, rectangles, circles, or triangles.
Eraser: Remove parts of your drawing.


Rulers: Drag horizontal/vertical rulers for alignment; lock them in place.
Grid: Toggle a 20x20px grid overlay for precision.
Templates: Predefined layouts for house plans, room layouts, floor plans, furniture, landscapes, mechanical parts, circuit diagrams, and flowcharts.
Clear Canvas: Reset the canvas (with confirmation).

Shortcuts

Escape: Exit fullscreen mode.
Export: Click the export button or use the header control.

Dependencies
The editor uses the following external libraries (loaded via CDN):

Font Awesome 6.4.0: For icons.
Marked.js: For Markdown parsing.
jsPDF 2.5.1: For PDF export.

No additional setup is required as these are included in the HTML.
Installation

Download or clone the repository.
Open editor.html in a web browser.
Ensure an internet connection for CDN-loaded dependencies.

File Structure

editor.html: The main HTML file containing the IDE's structure, styles, and JavaScript.

Customization

Styles: Modify the CSS in the <style> section to change colors, fonts, or layout.
Templates: Add new drawing templates in the loadTemplate function.
Export Formats: Extend the exportDocument function to support additional formats.
Statistics: Customize the updateStats function to track additional metrics.

Limitations

Drawing: Limited to basic shapes and freehand drawing; no advanced vector editing.
PDF Export: Uses jsPDF with basic formatting; complex Markdown may not render perfectly.
Local Storage: Autosave relies on browser localStorage, which has size limits.
Offline Use: Requires internet for CDN dependencies unless hosted locally.

Troubleshooting

Preview Not Updating: Click the "Refresh" button or ensure valid Markdown syntax.
Export Fails: Check browser compatibility (modern browsers recommended).
Drawing Issues: Ensure canvas is not obstructed by rulers or modal dialogs.
Mobile: Use touch events for drawing; some features may be less precise.

License
This project is open-source and available under the MIT License.
