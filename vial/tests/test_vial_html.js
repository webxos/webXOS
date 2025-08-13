import { JSDOM } from 'jsdom';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const html = fs.readFileSync(path.resolve(__dirname, '../../pages/vial.js'), 'utf8');

describe('Vial Frontend Tests', () => {
  let dom;
  let container;

  beforeEach(() => {
    dom = new JSDOM(html, { runScripts: 'dangerously', resources: 'usable' });
    container = dom.window.document.body;
  });

  test('renders vial container', () => {
    const vialContainer = container.querySelector('#vial-container');
    expect(vialContainer).not.toBeNull();
  });

  test('displays search results', async () => {
    const mockResults = {
      data: { matches: [{ id: 'doc1', data: 'Test data', score: 0.9 }] }
    };
    dom.window.axios = {
      post: jest.fn().mockResolvedValue({ data: mockResults })
    };
    const nomicAgent = await import('../../static/agent1.js');
    await nomicAgent.default.search('test', 'user123', 'key123');
    await nomicAgent.default.displayResults(mockResults, 'search-results');
    const resultsDiv = container.querySelector('#search-results');
    expect(resultsDiv.textContent).toContain('doc1: Test data (Score: 0.9)');
  });

  test('handles search error', async () => {
    dom.window.axios = {
      post: jest.fn().mockRejectedValue(new Error('Invalid query'))
    };
    const nomicAgent = await import('../../static/agent1.js');
    await expect(nomicAgent.default.search('test', 'user123', 'key123')).rejects.toThrow('Nomic search failed: Invalid query');
  });

  test('executes git command', async () => {
    dom.window.axios = {
      post: jest.fn().mockResolvedValue({ data: { status: 'success', output: 'git status output' } })
    };
    const response = await dom.window.axios.post('/v1/api/git', {
      user_id: 'user123',
      command: 'git status',
      repo_url: 'https://github.com/webxos/webxos.git'
    }, { headers: { Authorization: 'Bearer key123' } });
    expect(response.data.output).toBe('git status output');
  });
});
