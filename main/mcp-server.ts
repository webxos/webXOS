import { createServer } from 'http';
import { MCPTransport } from './transport/http-transport';
import { AuthenticationTool } from './tools/authentication';
import { VialManagementTool } from './tools/vial-management';
import { logger } from './lib/logger';
import { DatabaseConfig } from './config/database-config';

class MCPServer {
  private transport: MCPTransport;
  private tools: Map<string, any>;
  private db: DatabaseConfig;

  constructor() {
    this.db = new DatabaseConfig();
    this.tools = new Map([
      ['authentication', new AuthenticationTool(this.db)],
      ['vial-management', new VialManagementTool(this.db)]
    ]);
    this.transport = new MCPTransport(this.handleRequest.bind(this));
  }

  async start() {
    try {
      await this.db.connect();
      const server = createServer(this.transport.handle.bind(this.transport));
      server.listen(8080, () => {
        logger.info('MCP Server started on port 8080');
      });
    } catch (error) {
      logger.error(`Server startup error: ${error.message}`);
      process.exit(1);
    }
  }

  async handleRequest(request: any): Promise<any> {
    try {
      const { tool, input } = request;
      if (!this.tools.has(tool)) {
        throw new Error(`Unknown tool: ${tool}`);
      }
      const toolInstance = this.tools.get(tool);
      const result = await toolInstance.execute(input);
      return {
        status: 'success',
        data: result,
        capabilities: Array.from(this.tools.keys())
      };
    } catch (error) {
      logger.error(`Request error: ${error.message}`);
      return {
        status: 'error',
        error: error.message,
        capabilities: Array.from(this.tools.keys())
      };
    }
  }
}

const server = new MCPServer();
server.start();
