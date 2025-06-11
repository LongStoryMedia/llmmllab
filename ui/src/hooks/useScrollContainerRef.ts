import { useEffect, useRef } from "react";


function useScrollContainerRef() {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight;
    }
  }, []);

  return ref;
}

export default useScrollContainerRef;