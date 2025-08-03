require('dotenv').config();
const fs = require('fs').promises;
const path = require('path');

async function generateSiteIndex() {
    const repoDir = process.cwd();
    const outputFile = path.join(repoDir, 'site_index.json');
    let index = [];
    let errors = [];

    // Scan local repository for HTML, .py, .md files
    async function scanLocalDir(dir, prefix = '') {
        try {
            const entries = await fs.readdir(dir, { withFileTypes: true });
            for (const entry of entries) {
                const fullPath = path.join(dir, entry.name);
                const relativePath = path.join(prefix, entry.name).replace(/\\/g, '/');
                if (entry.isDirectory() && !/node_modules|\.git/.test(relativePath)) {
                    await scanLocalDir(fullPath, relativePath);
                } else if (/\.(html|py|md)$/i.test(entry.name)) {
                    try {
                        const text = await fs.readFile(fullPath, 'utf8');
                        if (text.trim()) {
                            index.push({
                                path: `/${relativePath}`,
                                source: `Site: /${relativePath}`,
                                text
                            });
                        } else {
                            errors.push(`Empty content for ${relativePath}`);
                        }
                    } catch (error) {
                        errors.push(`Failed to read ${relativePath}: ${error.message}`);
                    }
                }
            }
        } catch (error) {
            errors.push(`Failed to scan directory ${dir}: ${error.message}`);
        }
    }

    try {
        // Scan local files
        await scanLocalDir(repoDir);
        // Write index
        await fs.writeFile(outputFile, JSON.stringify(index, null, 2));
        console.log(`Generated ${outputFile} with ${index.length} files`);
        if (errors.length > 0) {
            console.error(`Errors during generation: ${errors.join('; ')}`);
        }
    } catch (error) {
        console.error(`Error generating site_index.json: ${error.message}`);
    }
}

generateSiteIndex();
