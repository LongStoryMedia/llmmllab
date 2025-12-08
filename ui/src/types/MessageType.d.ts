/**
 * Type of message
 */
export type MessageType = 'connect' | 'connected' | 'pause' | 'paused' | 'resume' | 'resumed' | 'cancel' | 'cancelled' | 'complete' | 'error' | 'warning' | 'info' | 'failed';
/**
 * Constant values for MessageType
 */
export declare const MessageTypeValues: {
    /** connect */
    readonly CONNECT: "connect";
    /** connected */
    readonly CONNECTED: "connected";
    /** pause */
    readonly PAUSE: "pause";
    /** paused */
    readonly PAUSED: "paused";
    /** resume */
    readonly RESUME: "resume";
    /** resumed */
    readonly RESUMED: "resumed";
    /** cancel */
    readonly CANCEL: "cancel";
    /** cancelled */
    readonly CANCELLED: "cancelled";
    /** complete */
    readonly COMPLETE: "complete";
    /** error */
    readonly ERROR: "error";
    /** warning */
    readonly WARNING: "warning";
    /** info */
    readonly INFO: "info";
    /** failed */
    readonly FAILED: "failed";
};
