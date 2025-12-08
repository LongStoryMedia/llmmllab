"use client";
import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useControllableState } from "@radix-ui/react-use-controllable-state";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "../ui/collapsible";
import { cn } from "../../lib/utils";
import { BrainIcon, ChevronDownIcon } from "lucide-react";
import { createContext, memo, useContext, useEffect, useState } from "react";
import { Streamdown } from "streamdown";
import { Shimmer } from "./shimmer";
const ReasoningContext = createContext(null);
const useReasoning = () => {
    const context = useContext(ReasoningContext);
    if (!context) {
        throw new Error("Reasoning components must be used within Reasoning");
    }
    return context;
};
const AUTO_CLOSE_DELAY = 1000;
const MS_IN_S = 1000;
export const Reasoning = memo(({ className, isStreaming = false, open, defaultOpen = true, onOpenChange, duration: durationProp, children, ...props }) => {
    const [isOpen, setIsOpen] = useControllableState({
        prop: open,
        defaultProp: defaultOpen,
        onChange: onOpenChange
    });
    const [duration, setDuration] = useControllableState({
        prop: durationProp,
        defaultProp: undefined
    });
    const [hasAutoClosed, setHasAutoClosed] = useState(false);
    const [startTime, setStartTime] = useState(null);
    // Track duration when streaming starts and ends
    useEffect(() => {
        if (isStreaming) {
            if (startTime === null) {
                setStartTime(Date.now());
            }
        }
        else if (startTime !== null) {
            setDuration(Math.ceil((Date.now() - startTime) / MS_IN_S));
            setStartTime(null);
        }
    }, [isStreaming, startTime, setDuration]);
    // Auto-open when streaming starts, auto-close when streaming ends (once only)
    useEffect(() => {
        if (defaultOpen && !isStreaming && isOpen && !hasAutoClosed) {
            // Add a small delay before closing to allow user to see the content
            const timer = setTimeout(() => {
                setIsOpen(false);
                setHasAutoClosed(true);
            }, AUTO_CLOSE_DELAY);
            return () => clearTimeout(timer);
        }
    }, [isStreaming, isOpen, defaultOpen, setIsOpen, hasAutoClosed]);
    const handleOpenChange = (newOpen) => {
        setIsOpen(newOpen);
    };
    return (_jsx(ReasoningContext.Provider, { value: { isStreaming, isOpen, setIsOpen, duration }, children: _jsx(Collapsible, { className: cn("not-prose mb-4", className), onOpenChange: handleOpenChange, open: isOpen, ...props, children: children }) }));
});
const getThinkingMessage = (isStreaming, duration) => {
    if (isStreaming || duration === 0) {
        return _jsx(Shimmer, { duration: 1, children: "Thinking..." });
    }
    if (duration === undefined) {
        return _jsx("p", { children: "Thought for a few seconds" });
    }
    return _jsxs("p", { children: ["Thought for ", duration, " seconds"] });
};
export const ReasoningTrigger = memo(({ className, children, ...props }) => {
    const { isStreaming, isOpen, duration } = useReasoning();
    return (_jsx(CollapsibleTrigger, { className: cn("flex w-full items-center gap-2 text-muted-foreground text-sm transition-colors hover:text-foreground", className), ...props, children: children ?? (_jsxs(_Fragment, { children: [_jsx(BrainIcon, { className: "size-4" }), getThinkingMessage(isStreaming, duration), _jsx(ChevronDownIcon, { className: cn("size-4 transition-transform", isOpen ? "rotate-180" : "rotate-0") })] })) }));
});
export const ReasoningContent = memo(({ className, children, ...props }) => (_jsx(CollapsibleContent, { className: cn("mt-4 text-sm", "data-[state=closed]:fade-out-0 data-[state=closed]:slide-out-to-top-2 data-[state=open]:slide-in-from-top-2 text-muted-foreground outline-none data-[state=closed]:animate-out data-[state=open]:animate-in", className), ...props, children: _jsx(Streamdown, { ...props, children: children }) })));
Reasoning.displayName = "Reasoning";
ReasoningTrigger.displayName = "ReasoningTrigger";
ReasoningContent.displayName = "ReasoningContent";
