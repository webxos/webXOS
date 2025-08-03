const fs = require('fs').promises;
const path = require('path');
const fetch = require('node-fetch');

async function generateSiteIndex() {
    const repoDir = process.cwd();
    const outputFile = path.join(repoDir, 'site_index.json');
    const githubApiUrl = 'https://api.github.com/repos/webxos/webxos/contents';
    const token = process.env.GITHUB_TOKEN || '';
    const headers = token ? { Authorization: `token ${token}` } : {};

    let index = [];

    // Scan local repository for HTML, .py, .md files
    async function scanLocalDir(dir, prefix = '') {
        const entries = await fs.readdir(dir, { withFileTypes: true });
        for (const entry of entries) {
            const fullPath = path.join(dir, entry.name);
            const relativePath = path.join(prefix, entry.name).replace(/\\/g, '/');
            if (entry.isDirectory()) {
                await scanLocalDir(fullPath, relativePath);
            } else if (/\.(html|py|md)$/i.test(entry.name)) {
                const text = await fs.readFile(fullPath, 'utf8');
                if (text.trim()) {
                    index.push({
                        path: `/${relativePath}`,
                        source: `Site: /${relativePath}`,
                        text
                    });
                }
            }
        }
    }

    // Fetch GitHub repo files
    async function fetchGitHubFiles(url, pathPrefix = '') {
        try {
            const response = await fetch(url, { headers });
            if (!response.ok) {
                throw new Error(`GitHub API fetch failed: HTTP ${response.status}`);
            }
            const data = await response.json();
            for (const item of data) {
                if (item.type === 'file' && /\.(html|py|md)$/i.test(item.name)) {
                    const fileResponse = await fetch(item.download_url, { headers });
                    if (fileResponse.ok) {
                        const text = await fileResponse.text();
                        if (text.trim()) {
                            index.push({
                                path: item.path,
                                source: `GitHub: ${item.path}`,
                                text
                            });
                        }
                    }
                } else if (item.type === 'dir') {
                    await fetchGitHubFiles(item.url, `${pathPrefix}/${item.name}`);
                }
            }
        } catch (error) {
            console.error(`GitHub fetch error: ${error.message}`);
        }
    }

    try {
        // Scan local files
        await scanLocalDir(repoDir);
        // Fetch GitHub files
        await fetchGitHubFiles(githubApiUrl);
        // Write index
        await fs.writeFile(outputFile, JSON.stringify(index, null, 2));
        console.log(`Generated ${outputFile} with ${index.length} files`);
    } catch (error) {
        console.error(`Error generating site_index.json: ${error.message}`);
    }
}

generateSiteIndex();
