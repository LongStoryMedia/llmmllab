import React, { useCallback, useEffect, useRef, useMemo, useState } from 'react';
import { VariableSizeList as List } from 'react-window';
import { styled } from '@mui/material';
import ChatBubble from './ChatBubble';
import { Message } from '../../types/Message';

interface VirtualizedChatListProps {
  messages: Message[];
  streamingMessage?: Message;
  containerHeight: number;
}

const ListContainer = styled('div')(({ theme }) => ({
  flex: 1,
  width: '100%',
  paddingBottom: '120px', // Space for fixed input
  paddingLeft: theme.spacing(2),
  paddingRight: theme.spacing(2),
  paddingTop: theme.spacing(1)
}));

const MessageWrapper = styled('div')(({ theme }) => ({
  padding: `${theme.spacing(0.5)} 0`,
  display: 'flex',
  flexDirection: 'column'
}));

// Item component for react-window
const MessageItem: React.FC<{
  index: number;
  style: React.CSSProperties;
  data: {
    messages: Message[];
    streamingMessage?: Message;
    onHeightChange: (index: number, height: number) => void;
  };
}> = ({ index, style, data }) => {
  const { messages, streamingMessage, onHeightChange } = data;
  const itemRef = useRef<HTMLDivElement>(null);

  // Determine if this is the streaming message (always last item if present)
  const isStreamingItem = streamingMessage && index === messages.length;
  const message = isStreamingItem ? streamingMessage : messages[index];

  // Measure height and report changes
  useEffect(() => {
    if (itemRef.current) {
      const resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const height = entry.contentRect.height;
          onHeightChange(index, height);
        }
      });

      resizeObserver.observe(itemRef.current);
      return () => resizeObserver.disconnect();
    }
  }, [index, onHeightChange, message]);

  if (!message) {
    return null;
  }

  return (
    <div style={style}>
      <MessageWrapper ref={itemRef}>
        <ChatBubble message={message} />
      </MessageWrapper>
    </div>
  );
};

const VirtualizedChatList: React.FC<VirtualizedChatListProps> = ({
  messages,
  streamingMessage,
  containerHeight
}) => {
  const listRef = useRef<List>(null);
  const [itemHeights, setItemHeights] = useState<Map<number, number>>(new Map());
  const [shouldScrollToBottom, setShouldScrollToBottom] = useState(true);
  // Track the raw scroll offset to decide "near bottom" precisely
  const scrollOffsetRef = useRef<number>(0);
  // Prevent scheduling many scroll calls while streaming updates continuously
  const scrollScheduledRef = useRef<boolean>(false);

  // Total items = messages + streaming message (if present)
  const totalItems = messages.length + (streamingMessage ? 1 : 0);

  // Height cache for each item
  const getItemHeight = useCallback((index: number) => {
    return itemHeights.get(index) || 100; // Default height estimate
  }, [itemHeights]);

  // Handle height changes from individual items
  const handleHeightChange = useCallback((index: number, height: number) => {
    setItemHeights(prev => {
      const newMap = new Map(prev);
      if (newMap.get(index) !== height) {
        newMap.set(index, height);
        // Reset cache for this item in react-window
        if (listRef.current) {
          listRef.current.resetAfterIndex(index, false);

          // If this is the last item (streaming message) consider scrolling
          // only when the user is near the bottom. This allows the user to
          // scroll up while new content is streaming without being forcibly
          // scrolled to the bottom. Use a small margin in pixels.
          if (index === totalItems - 1) {
            const marginPx = 30;
            const totalHeight = Array.from(newMap.values()).reduce((s, h) => s + h, 0) || totalItems * 100;
            const isNearBottom = scrollOffsetRef.current + containerHeight >= totalHeight - marginPx;
            if (shouldScrollToBottom && isNearBottom) {
              if (!scrollScheduledRef.current) {
                scrollScheduledRef.current = true;
                // schedule one scroll on next animation frame to avoid constant resets
                window.requestAnimationFrame(() => {
                  if (listRef.current) {
                    listRef.current.scrollToItem(totalItems - 1, 'end');
                  }
                  scrollScheduledRef.current = false;
                });
              }
            }
          }
        }
      }
      return newMap;
    });
  }, [shouldScrollToBottom, totalItems, containerHeight]);

  // Data to pass to each item
  const itemData = useMemo(() => ({
    messages,
    streamingMessage,
    onHeightChange: handleHeightChange
  }), [messages, streamingMessage, handleHeightChange]);

  // Auto-scroll to bottom when new messages arrive or streaming content updates
  useEffect(() => {
    if (shouldScrollToBottom && listRef.current && totalItems > 0) {
      if (!scrollScheduledRef.current) {
        scrollScheduledRef.current = true;
        window.requestAnimationFrame(() => {
          if (listRef.current) {
            listRef.current.scrollToItem(totalItems - 1, 'end');
          }
          scrollScheduledRef.current = false;
        });
      }
    }
  }, [totalItems, shouldScrollToBottom, streamingMessage]);

  // Check if user is at bottom to determine auto-scroll behavior
  const handleScroll = useCallback(({ scrollOffset, scrollUpdateWasRequested }: { scrollOffset: number; scrollUpdateWasRequested: boolean }) => {
    // Track the raw scroll offset for near-bottom calculations
    scrollOffsetRef.current = scrollOffset;
    if (!scrollUpdateWasRequested) {
      const totalHeight = Array.from(itemHeights.values()).reduce((sum, height) => sum + height, 0) || totalItems * 100;
      const marginPx = 60; // small margin to decide "near bottom"
      const isNearBottom = scrollOffset + containerHeight >= totalHeight - marginPx;
      setShouldScrollToBottom(isNearBottom);
    }
  }, [containerHeight, totalItems, itemHeights]);

  if (totalItems === 0) {
    return <ListContainer />;
  }

  return (
    <ListContainer onPointerDown={() => setShouldScrollToBottom(false)}>
      <List
        ref={listRef}
        height={containerHeight - 140} // Account for padding and input
        width="100%"
        itemCount={totalItems}
        itemSize={getItemHeight}
        itemData={itemData}
        onScroll={handleScroll}
        overscanCount={5} // Render a few extra items for smooth scrolling
      >
        {MessageItem}
      </List>
    </ListContainer>
  );
};

export default VirtualizedChatList;