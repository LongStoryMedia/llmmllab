import { useEffect, useState } from 'react';
const useMediaQuery = (query) => {
    const [matches, setMatches] = useState(false);
    useEffect(() => {
        const mediaQueryList = window.matchMedia(query);
        const handleChange = (event) => {
            setMatches(event.matches);
        };
        setMatches(mediaQueryList.matches);
        mediaQueryList.addEventListener('change', handleChange);
        return () => {
            mediaQueryList.removeEventListener('change', handleChange);
        };
    }, [query]);
    return matches;
};
export default useMediaQuery;
