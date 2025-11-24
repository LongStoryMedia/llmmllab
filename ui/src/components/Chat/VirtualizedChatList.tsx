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
    contentCount: number;
    spacerHeight: number;
  };
}> = ({ index, style, data }) => {
  const { messages, streamingMessage, onHeightChange, contentCount, spacerHeight } = data;
  const itemRef = useRef<HTMLDivElement>(null);

  // If this index is the extra spacer at the end, render empty spacer
  const isSpacer = index === contentCount;
  // Determine if this is the streaming message (always last content index if present)
  const isStreamingItem = streamingMessage && index === messages.length;
  const message = isSpacer ? null : isStreamingItem ? streamingMessage : messages[index];

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

  if (isSpacer) {
    return (
      <div style={style}>
        <div ref={itemRef} style={{ height: spacerHeight }} />
      </div>
    );
  }

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

  // Extra spacer to allow overscroll past the last message
  const EXTRA_SPACER_HEIGHT = 400;

  // contentCount = messages + streaming message (if present)
  const contentCount = messages.length + (streamingMessage ? 1 : 0);
  // itemCount includes an extra spacer item at the end to allow overscroll
  const totalItems = contentCount + 1;

  // Height cache for each item
  const getItemHeight = useCallback((index: number) => {
    if (index === contentCount) {
      return EXTRA_SPACER_HEIGHT;
    }
    return itemHeights.get(index) || 100; // Default height estimate
  }, [itemHeights, contentCount]);

  // Handle height changes from individual items
  const handleHeightChange = useCallback((index: number, height: number) => {
    setItemHeights(prev => {
      const newMap = new Map(prev);
      if (newMap.get(index) !== height) {
        newMap.set(index, height);
        // Reset cache for this item in react-window
        if (listRef.current) {
          listRef.current.resetAfterIndex(index, false);

          // If this is the last content item (not the spacer) and auto-scroll
          // is enabled, schedule a single RAF scroll to the last content item.
          // This avoids relying on potentially inaccurate "near bottom"
          // heuristics driven by partially-measured heights on initial load.
          if (index === contentCount - 1 && shouldScrollToBottom) {
            if (!scrollScheduledRef.current) {
              scrollScheduledRef.current = true;
              window.requestAnimationFrame(() => {
                if (listRef.current) {
                  listRef.current.scrollToItem(Math.max(0, contentCount - 1), 'end');
                }
                scrollScheduledRef.current = false;
              });
            }
          }
        }
      }
      return newMap;
    });
  }, [shouldScrollToBottom, contentCount]);

  // Data to pass to each item
  const itemData = useMemo(() => ({
    messages,
    streamingMessage,
    onHeightChange: handleHeightChange,
    contentCount,
    spacerHeight: EXTRA_SPACER_HEIGHT
  }), [messages, streamingMessage, handleHeightChange, contentCount]);

  // Auto-scroll to bottom when new messages arrive or streaming content updates
  // Scroll to bottom when contentCount changes (including on mount).
  // Reset react-window cache and use two RAFs so measurements settle before scrolling.
  useEffect(() => {
    if (!listRef.current || contentCount === 0) {
      return;
    }

    const tryScroll = () => {
      try {
        // Force react-window to recompute sizes before scrolling
        listRef.current!.resetAfterIndex(0, true);
      } catch (_err) {
        // ignore
      }

      // Double RAF to ensure DOM/layout settled. First scroll to the spacer
      // (true bottom) then repeatedly align to the last content item across
      // a few RAFs to let react-window measure actual sizes and avoid landing
      // mid-list when many items use estimated heights.
      window.requestAnimationFrame(() => {
        window.requestAnimationFrame(() => {
          if (listRef.current) {
            const lastContentIndex = Math.max(0, contentCount - 1);
            // Scroll to spacer to ensure true bottom first
            listRef.current.scrollToItem(totalItems - 1, 'end');

            // Try aligning to last content item across several RAFs.
            let attempts = 6;
            const attemptAlign = () => {
              if (!listRef.current) {
                return;
              }
              if (attempts <= 0) {
                return;
              }
              listRef.current.scrollToItem(lastContentIndex, 'end');
              attempts -= 1;
              // schedule another try to cope with measurements settling
              window.requestAnimationFrame(attemptAlign);
            };
            window.requestAnimationFrame(attemptAlign);

            setShouldScrollToBottom(true);
          }
        });
      });
    };

    tryScroll();
  }, [contentCount, totalItems, containerHeight]);

  // Check if user is at bottom to determine auto-scroll behavior
  const handleScroll = useCallback(({ scrollOffset, scrollUpdateWasRequested }: { scrollOffset: number; scrollUpdateWasRequested: boolean }) => {
    // Track the raw scroll offset for near-bottom calculations
    scrollOffsetRef.current = scrollOffset;
    if (!scrollUpdateWasRequested) {
      // Compute total measured height; fall back to estimated height for unknowns
      const measuredTotal = Array.from(itemHeights.values()).reduce((sum, height) => sum + height, 0);
      const estimatedUnknowns = Math.max(0, totalItems - itemHeights.size) * 100;
      const totalHeight = measuredTotal + estimatedUnknowns;

      const marginPx = 10; // small margin to decide "near bottom"
      const isNearBottom = scrollOffset + containerHeight >= totalHeight - marginPx;

      // If user scrolls into the spacer region (last item), treat as bottom and stick.
      const spacerTop = totalHeight - EXTRA_SPACER_HEIGHT;
      const isInSpacer = scrollOffset + containerHeight >= spacerTop;

      setShouldScrollToBottom(isNearBottom || isInSpacer);
    }
  }, [containerHeight, totalItems, itemHeights]);

  if (totalItems === 0) {
    return <ListContainer />;
  }

  return (
    <ListContainer
      onPointerDown={() => setShouldScrollToBottom(false)}
      onPointerUp={() => {
        // If the user released the pointer while in the spacer area, enable auto-scroll
        const totalMeasured = Array.from(itemHeights.values()).reduce((s, h) => s + h, 0);
        const estimatedUnknowns = Math.max(0, totalItems - itemHeights.size) * 100;
        const totalHeight = totalMeasured + estimatedUnknowns;
        const viewportBottom = scrollOffsetRef.current + containerHeight - 5;
        if (viewportBottom >= totalHeight - EXTRA_SPACER_HEIGHT) {
          setShouldScrollToBottom(true);
        }
      }}
    >
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