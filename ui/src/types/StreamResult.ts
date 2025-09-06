export interface StatusUpdate {
    type: 'status' | 'content' | 'done';
    message?: string;
    timestamp?: number;
    isDone?: boolean;
}

export interface StreamResult {
    content: string;
    status: StatusUpdate | null;
}
