declare function useScrollToBottom(): {
    containerRef: import("react").RefObject<HTMLDivElement | null>;
    scrollToBottom: () => void;
};
export default useScrollToBottom;
