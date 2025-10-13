import { useEffect, useRef } from "react";

function useScrollContainerRef() {
  const ref = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, []);

  // Return both ref and scroll function for external use
  return ref;
}

export default useScrollContainerRef;