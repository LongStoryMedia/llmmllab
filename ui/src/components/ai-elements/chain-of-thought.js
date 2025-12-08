"use client";
import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useControllableState } from "@radix-ui/react-use-controllable-state";
import { Badge } from "../ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "../ui/collapsible";
import { cn } from "../../lib/utils";
import { BrainIcon, ChevronDownIcon, DotIcon } from "lucide-react";
import { createContext, memo, useContext, useMemo } from "react";
const ChainOfThoughtContext = createContext(null);
const useChainOfThought = () => {
    const context = useContext(ChainOfThoughtContext);
    if (!context) {
        throw new Error("ChainOfThought components must be used within ChainOfThought");
    }
    return context;
};
export const ChainOfThought = memo(({ className, open, defaultOpen = false, onOpenChange, children, ...props }) => {
    const [isOpen, setIsOpen] = useControllableState({
        prop: open,
        defaultProp: defaultOpen,
        onChange: onOpenChange
    });
    const chainOfThoughtContext = useMemo(() => ({ isOpen, setIsOpen }), [isOpen, setIsOpen]);
    return (_jsx(ChainOfThoughtContext.Provider, { value: chainOfThoughtContext, children: _jsx("div", { className: cn("not-prose max-w-prose space-y-4", className), ...props, children: children }) }));
});
export const ChainOfThoughtHeader = memo(({ className, children, ...props }) => {
    const { isOpen, setIsOpen } = useChainOfThought();
    return (_jsx(Collapsible, { onOpenChange: setIsOpen, open: isOpen, children: _jsxs(CollapsibleTrigger, { className: cn("flex w-full items-center gap-2 text-muted-foreground text-sm transition-colors hover:text-foreground", className), ...props, children: [_jsx(BrainIcon, { className: "size-4" }), _jsx("span", { className: "flex-1 text-left", children: children ?? "Chain of Thought" }), _jsx(ChevronDownIcon, { className: cn("size-4 transition-transform", isOpen ? "rotate-180" : "rotate-0") })] }) }));
});
export const ChainOfThoughtStep = memo(({ className, icon: Icon = DotIcon, label, description, status = "complete", children, ...props }) => {
    const statusStyles = {
        complete: "text-muted-foreground",
        active: "text-foreground",
        pending: "text-muted-foreground/50"
    };
    return (_jsxs("div", { className: cn("flex gap-2 text-sm", statusStyles[status], "fade-in-0 slide-in-from-top-2 animate-in", className), ...props, children: [_jsxs("div", { className: "relative mt-0.5", children: [_jsx(Icon, { className: "size-4" }), _jsx("div", { className: "-mx-px absolute top-7 bottom-0 left-1/2 w-px bg-border" })] }), _jsxs("div", { className: "flex-1 space-y-2", children: [_jsx("div", { children: label }), description && (_jsx("div", { className: "text-muted-foreground text-xs", children: description })), children] })] }));
});
export const ChainOfThoughtSearchResults = memo(({ className, ...props }) => (_jsx("div", { className: cn("flex items-center gap-2", className), ...props })));
export const ChainOfThoughtSearchResult = memo(({ className, children, ...props }) => (_jsx(Badge, { className: cn("gap-1 px-2 py-0.5 font-normal text-xs", className), variant: "secondary", ...props, children: children })));
export const ChainOfThoughtContent = memo(({ className, children, ...props }) => {
    const { isOpen } = useChainOfThought();
    return (_jsx(Collapsible, { open: isOpen, children: _jsx(CollapsibleContent, { className: cn("mt-2 space-y-3", "data-[state=closed]:fade-out-0 data-[state=closed]:slide-out-to-top-2 data-[state=open]:slide-in-from-top-2 text-popover-foreground outline-none data-[state=closed]:animate-out data-[state=open]:animate-in", className), ...props, children: children }) }));
});
export const ChainOfThoughtImage = memo(({ className, children, caption, ...props }) => (_jsxs("div", { className: cn("mt-2 space-y-2", className), ...props, children: [_jsx("div", { className: "relative flex max-h-[22rem] items-center justify-center overflow-hidden rounded-lg bg-muted p-3", children: children }), caption && _jsx("p", { className: "text-muted-foreground text-xs", children: caption })] })));
ChainOfThought.displayName = "ChainOfThought";
ChainOfThoughtHeader.displayName = "ChainOfThoughtHeader";
ChainOfThoughtStep.displayName = "ChainOfThoughtStep";
ChainOfThoughtSearchResults.displayName = "ChainOfThoughtSearchResults";
ChainOfThoughtSearchResult.displayName = "ChainOfThoughtSearchResult";
ChainOfThoughtContent.displayName = "ChainOfThoughtContent";
ChainOfThoughtImage.displayName = "ChainOfThoughtImage";
