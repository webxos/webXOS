const net = new brain.NeuralNetwork();

function vectorizeQuery(query) {
  return query.split('').map(c => c.charCodeAt(0) / 255).slice(0, 10).padEnd(10, 0);
}

async function trainNetwork(query, score) {
  const input = vectorizeQuery(query);
  await net.trainAsync([{ input, output: [score] }]);
}

async function predictRelevance(query) {
  const input = vectorizeQuery(query);
  return net.run(input)[0];
}

export { trainNetwork, predictRelevance };
