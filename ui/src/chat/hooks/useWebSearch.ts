import { useState } from 'react';
import { useConfigContext } from '../../context/ConfigContext';
import { useAuth } from '../../auth';

interface WebSearchResult {
  query: string;
  results?: string[];
  contents?: string[];
  error?: string;
}

interface UseWebSearchOptions {
  autoSearch?: boolean;
}

export const useWebSearch = (options: UseWebSearchOptions = {}) => {
  const { config } = useConfigContext();
  const { user } = useAuth();
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<WebSearchResult | null>(null);

  // Check if web search is enabled in user config
  const isWebSearchEnabled = config?.web_search?.enabled ?? false;

  // Should we auto-detect and search?
  const shouldAutoDetect = options.autoSearch ?? (config?.web_search?.auto_detect ?? false);

  const detectWebSearchIntent = async (query: string): Promise<boolean> => {
    if (!isWebSearchEnabled) {
      return false;
    }
    if (!shouldAutoDetect) {
      return false;
    }

    try {
      const response = await fetch('/api/websearch/detect', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query,
          user_id: user?.profile.sub
        })
      });

      if (!response.ok) {
        return false;
      }

      const data = await response.json();
      return data.needs_web_search ?? false;
    } catch (error) {
      console.error('Error detecting web search intent:', error);
      return false;
    }
  };

  const performWebSearch = async (query: string): Promise<WebSearchResult | null> => {
    if (!isWebSearchEnabled) {
      return null;
    }

    try {
      setIsSearching(true);
      const response = await fetch('/api/websearch', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query,
          user_id: user?.profile.sub
        })
      });

      if (!response.ok) {
        throw new Error(`Search failed with status ${response.status}`);
      }

      const data = await response.json();
      setSearchResults(data);
      return data;
    } catch (error) {
      console.error('Web search error:', error);
      const errorResult = {
        query,
        error: error instanceof Error ? error.message : 'Unknown error during web search'
      };
      setSearchResults(errorResult);
      return errorResult;
    } finally {
      setIsSearching(false);
    }
  };

  // Utility function to format search results for LLM consumption
  const formatSearchResultsForLLM = (results: WebSearchResult | null): string => {
    if (!results) {
      return '';
    }
    if (results.error) {
      return `Search error: ${results.error}`;
    }

    let formatted = `Web search results for "${results.query}":\n\n`;

    if (results.contents && results.contents.length > 0) {
      results.contents.forEach((content, index) => {
        // Add source URL if available
        const sourceUrl = results.results && results.results[index]
          ? `Source: ${results.results[index]}\n`
          : '';

        // Truncate content if it's too long
        const truncatedContent = content.length > 1000
          ? content.substring(0, 1000) + '... (content truncated)'
          : content;

        formatted += `[Result ${index + 1}]\n${sourceUrl}${truncatedContent}\n\n`;
      });
    } else if (results.results && results.results.length > 0) {
      formatted += 'Sources found:\n';
      results.results.forEach((url, index) => {
        formatted += `[${index + 1}] ${url}\n`;
      });
    } else {
      formatted += 'No relevant results found.';
    }

    return formatted;
  };

  return {
    isWebSearchEnabled,
    isSearching,
    searchResults,
    detectWebSearchIntent,
    performWebSearch,
    formatSearchResultsForLLM
  };
};