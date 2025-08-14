// main/server/mcp/wallet/vial_abi.ts
export const vialAbi = [
  {
    "inputs": [
      { "internalType": "string", "name": "vialId", "type": "string" },
      { "internalType": "string", "name": "status", "type": "string" }
    ],
    "name": "updateVialStatus",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      { "internalType": "string", "name": "vialId", "type": "string" }
    ],
    "name": "getVialStatus",
    "outputs": [
      { "internalType": "string", "name": "", "type": "string" }
    ],
    "stateMutability": "view",
    "type": "function"
  }
] as const;
