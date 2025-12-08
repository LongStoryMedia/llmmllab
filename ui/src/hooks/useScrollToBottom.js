import { useRef } from "react";
function useScrollToBottom() {
    const containerRef = useRef(null);
    const scrollToBottom = () => {
        if (containerRef.current) {
            containerRef.current.scrollTop = containerRef.current.scrollHeight;
        }
    };
    return { containerRef, scrollToBottom };
}
export default useScrollToBottom;
