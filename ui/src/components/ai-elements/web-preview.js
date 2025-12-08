"use client";
import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Button } from "../ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "../ui/collapsible";
import { Input } from "../ui/input";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "../ui/tooltip";
import { cn } from "../../lib/utils";
import { ChevronDownIcon } from "lucide-react";
import { createContext, useContext, useEffect, useState } from "react";
const WebPreviewContext = createContext(null);
const useWebPreview = () => {
    const context = useContext(WebPreviewContext);
    if (!context) {
        throw new Error("WebPreview components must be used within a WebPreview");
    }
    return context;
};
export const WebPreview = ({ className, children, defaultUrl = "", onUrlChange, ...props }) => {
    const [url, setUrl] = useState(defaultUrl);
    const [consoleOpen, setConsoleOpen] = useState(false);
    const handleUrlChange = (newUrl) => {
        setUrl(newUrl);
        onUrlChange?.(newUrl);
    };
    const contextValue = {
        url,
        setUrl: handleUrlChange,
        consoleOpen,
        setConsoleOpen
    };
    return (_jsx(WebPreviewContext.Provider, { value: contextValue, children: _jsx("div", { className: cn("flex size-full flex-col rounded-lg border bg-card", className), ...props, children: children }) }));
};
export const WebPreviewNavigation = ({ className, children, ...props }) => (_jsx("div", { className: cn("flex items-center gap-1 border-b p-2", className), ...props, children: children }));
export const WebPreviewNavigationButton = ({ onClick, disabled, tooltip, children, ...props }) => (_jsx(TooltipProvider, { children: _jsxs(Tooltip, { children: [_jsx(TooltipTrigger, { asChild: true, children: _jsx(Button, { className: "h-8 w-8 p-0 hover:text-foreground", disabled: disabled, onClick: onClick, size: "sm", variant: "ghost", ...props, children: children }) }), _jsx(TooltipContent, { children: _jsx("p", { children: tooltip }) })] }) }));
export const WebPreviewUrl = ({ value, onChange, onKeyDown, ...props }) => {
    const { url, setUrl } = useWebPreview();
    const [inputValue, setInputValue] = useState(url);
    // Sync input value with context URL when it changes externally
    useEffect(() => {
        setInputValue(url);
    }, [url]);
    const handleChange = (event) => {
        setInputValue(event.target.value);
        onChange?.(event);
    };
    const handleKeyDown = (event) => {
        if (event.key === "Enter") {
            const target = event.target;
            setUrl(target.value);
        }
        onKeyDown?.(event);
    };
    return (_jsx(Input, { className: "h-8 flex-1 text-sm", onChange: onChange ?? handleChange, onKeyDown: handleKeyDown, placeholder: "Enter URL...", value: value ?? inputValue, ...props }));
};
export const WebPreviewBody = ({ className, loading, src, ...props }) => {
    const { url } = useWebPreview();
    return (_jsxs("div", { className: "flex-1", children: [_jsx("iframe", { className: cn("size-full", className), sandbox: "allow-scripts allow-same-origin allow-forms allow-popups allow-presentation", src: (src ?? url) || undefined, title: "Preview", ...props }), loading] }));
};
export const WebPreviewConsole = ({ className, logs = [], children, ...props }) => {
    const { consoleOpen, setConsoleOpen } = useWebPreview();
    return (_jsxs(Collapsible, { className: cn("border-t bg-muted/50 font-mono text-sm", className), onOpenChange: setConsoleOpen, open: consoleOpen, ...props, children: [_jsx(CollapsibleTrigger, { asChild: true, children: _jsxs(Button, { className: "flex w-full items-center justify-between p-4 text-left font-medium hover:bg-muted/50", variant: "ghost", children: ["Console", _jsx(ChevronDownIcon, { className: cn("h-4 w-4 transition-transform duration-200", consoleOpen && "rotate-180") })] }) }), _jsx(CollapsibleContent, { className: cn("px-4 pb-4", "data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 outline-none data-[state=closed]:animate-out data-[state=open]:animate-in"), children: _jsxs("div", { className: "max-h-48 space-y-1 overflow-y-auto", children: [logs.length === 0 ? (_jsx("p", { className: "text-muted-foreground", children: "No console output" })) : (logs.map((log, index) => (_jsxs("div", { className: cn("text-xs", log.level === "error" && "text-destructive", log.level === "warn" && "text-yellow-600", log.level === "log" && "text-foreground"), children: [_jsx("span", { className: "text-muted-foreground", children: log.timestamp.toLocaleTimeString() }), " ", log.message] }, `${log.timestamp.getTime()}-${index}`)))), children] }) })] }));
};
