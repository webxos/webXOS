export interface JSONRPCRequest {
  jsonrpc: string;
  method: string;
  params: {
    user_id?: string;
    vial_id?: string;
    code?: string;
    commit_message?: string;
    redirect_uri?: string;
    [key: string]: any;
  };
  id: number;
}

export interface JSONRPCResponse {
  jsonrpc: string;
  result?: {
    vial_id?: string;
    balance?: number;
    commit_hash?: string;
    access_token?: string;
    user_id?: string;
    [key: string]: any;
  };
  error?: {
    code: number;
    message: string;
  };
  id: number;
}

export interface WalletBalanceOutput {
  vial_id: string;
  balance: number;
}

export interface VialGitPushOutput {
  commit_hash: string;
  balance: number;
}

export interface AuthTokenOutput {
  access_token: string;
  user_id: string;
}
