import { ChatResponse } from '../../types/ChatResponse';
import { ResponseSection } from '../../types/ResponseSection';
interface StreamingState {
    sections: ResponseSection[];
    currentSection?: ResponseSection;
    observerMessages: string[];
}
export declare const useStreamHandler: () => {
    sections: ResponseSection[];
    currentSection: ResponseSection | undefined;
    observerMessages: string[];
    processChunk: (chunk: ChatResponse) => StreamingState;
    resetStreaming: () => void;
    getFinalContent: () => string;
};
export {};
