import { useState, useEffect, useRef } from 'react';

export const useContainerDimensions = () => {
  const ref = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const getCurrentDimensions = () => {
      if (ref.current) {
        return {
          width: ref.current.offsetWidth,
          height: ref.current.offsetHeight
        };
      }
      return { width: 0, height: 0 };
    };

    const handleResize = () => {
      setDimensions(getCurrentDimensions());
    };

    // Set initial dimensions
    handleResize();

    // Add resize listener
    window.addEventListener('resize', handleResize);

    // Use ResizeObserver for more accurate container resizing
    const resizeObserver = new ResizeObserver(() => {
      handleResize();
    });

    if (ref.current) {
      resizeObserver.observe(ref.current);
    }

    return () => {
      window.removeEventListener('resize', handleResize);
      resizeObserver.disconnect();
    };
  }, []);

  return { ref, ...dimensions };
};