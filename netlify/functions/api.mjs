import { createServer } from 'fastify';
import { handler as handlerFn } from '../../main/api/mcp/main.py';

const fastify = createServer();

fastify.post('/mcp/execute', async (request, reply) => {
  const response = await handlerFn(request.body);
  return response;
});

export default async (request, context) => {
  const response = await fastify.inject({
    method: request.method,
    url: request.url,
    headers: request.headers,
    body: request.body
  });
  return new Response(response.body, {
    status: response.statusCode,
    headers: response.headers
  });
};
