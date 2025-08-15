// main/app/chatbot/test_chatbot.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import Chatbot from './page';
import { login } from '../../server/mcp/functions/auth';
import { createNote } from '../../server/mcp/functions/notes';
import '@testing-library/jest-dom';

jest.mock('../../server/mcp/functions/auth');
jest.mock('../../server/mcp/functions/notes');

describe('Chatbot Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.clear();
  });

  test('renders chatbot UI correctly', () => {
    render(<Chatbot />);
    expect(screen.getByText('WebXOS Searchbot with Vial')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Search...')).toBeInTheDocument();
    expect(screen.getByText('Search')).toBeInTheDocument();
    expect(screen.getByText('Clear')).toBeInTheDocument();
    expect(screen.getByText('Authenticate')).toBeInTheDocument();
    expect(screen.getByText('Import')).toBeInTheDocument();
    expect(screen.getByText('Copyright webXOS 2025')).toBeInTheDocument();
  });

  test('handles search without authentication', async () => {
    render(<Chatbot />);
    const searchInput = screen.getByPlaceholderText('Search...');
    const searchButton = screen.getByText('Search');
    fireEvent.change(searchInput, { target: { value: 'test query' } });
    fireEvent.click(searchButton);
    expect(await screen.findByText('Please authenticate first.')).toBeInTheDocument();
  });

  test('handles authentication', async () => {
    (login as jest.Mock).mockResolvedValue({ token: 'mock_token', userId: 'test_user' });
    window.prompt = jest.fn()
      .mockReturnValueOnce('test_user')
      .mockReturnValueOnce('password');
    render(<Chatbot />);
    const authButton = screen.getByText('Authenticate');
    fireEvent.click(authButton);
    expect(login).toHaveBeenCalledWith('test_user', 'password');
    expect(await screen.findByText('Authentication successful!')).toBeInTheDocument();
  });

  test('handles search with authentication', async () => {
    localStorage.setItem('apiKey', 'mock_token');
    localStorage.setItem('userId', 'test_user');
    (createNote as jest.Mock).mockResolvedValue({ note_id: 'note123' });
    render(<Chatbot />);
    const searchInput = screen.getByPlaceholderText('Search...');
    const searchButton = screen.getByText('Search');
    fireEvent.change(searchInput, { target: { value: 'test query' } });
    fireEvent.click(searchButton);
    expect(createNote).toHaveBeenCalledWith('Search Query', 'test query', ['search']);
    expect(await screen.findByText('Search query saved as note: note123')).toBeInTheDocument();
  });
});
