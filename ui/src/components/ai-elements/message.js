"use client";
import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { Button } from "../ui/button";
import { ButtonGroup, ButtonGroupText } from "../ui/button-group";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "../ui/tooltip";
import { cn } from "../../lib/utils";
// import { Box, Typography, Link, Table, TableBody, TableCell, TableHead, TableRow, useTheme } from '@mui/material';
// import { Prism as SyntaxHighlighter, SyntaxHighlighterProps } from 'react-syntax-highlighter';
// import { vscDarkPlus, vs } from 'react-syntax-highlighter/dist/esm/styles/prism';
// import CopyButton from '../Shared/CopyButton';
// import LazyImage from '../Shared/LazyImage';
import { ChevronLeftIcon, ChevronRightIcon, PaperclipIcon, XIcon } from "lucide-react";
import { createContext, memo, useContext, useEffect, useState } from "react";
import { Streamdown } from "streamdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
export const Message = ({ className, from, ...props }) => (_jsx("div", { className: cn("group flex w-full max-w-[80%] flex-col gap-2", from === "user" ? "is-user ml-auto justify-end" : "is-assistant", className), ...props }));
export const MessageContent = ({ children, className, ...props }) => (_jsx("div", { className: cn("flex flex-col gap-2 overflow-hidden text-sm", "group-[.is-user]:ml-auto group-[.is-user]:rounded-lg group-[.is-user]:bg-secondary group-[.is-user]:px-4 group-[.is-user]:py-3 group-[.is-user]:text-foreground", "group-[.is-assistant]:text-foreground", className), ...props, children: children }));
export const MessageActions = ({ className, children, ...props }) => (_jsx("div", { className: cn("flex items-center gap-1", className), ...props, children: children }));
export const MessageAction = ({ tooltip, children, label, variant = "ghost", size = "icon-sm", ...props }) => {
    const button = (_jsxs(Button, { size: size, type: "button", variant: variant, ...props, children: [children, _jsx("span", { className: "sr-only", children: label || tooltip })] }));
    if (tooltip) {
        return (_jsx(TooltipProvider, { children: _jsxs(Tooltip, { children: [_jsx(TooltipTrigger, { asChild: true, children: button }), _jsx(TooltipContent, { children: _jsx("p", { children: tooltip }) })] }) }));
    }
    return button;
};
const MessageBranchContext = createContext(null);
const useMessageBranch = () => {
    const context = useContext(MessageBranchContext);
    if (!context) {
        throw new Error("MessageBranch components must be used within MessageBranch");
    }
    return context;
};
export const MessageBranch = ({ defaultBranch = 0, onBranchChange, className, ...props }) => {
    const [currentBranch, setCurrentBranch] = useState(defaultBranch);
    const [branches, setBranches] = useState([]);
    const handleBranchChange = (newBranch) => {
        setCurrentBranch(newBranch);
        onBranchChange?.(newBranch);
    };
    const goToPrevious = () => {
        const newBranch = currentBranch > 0 ? currentBranch - 1 : branches.length - 1;
        handleBranchChange(newBranch);
    };
    const goToNext = () => {
        const newBranch = currentBranch < branches.length - 1 ? currentBranch + 1 : 0;
        handleBranchChange(newBranch);
    };
    const contextValue = {
        currentBranch,
        totalBranches: branches.length,
        goToPrevious,
        goToNext,
        branches,
        setBranches
    };
    return (_jsx(MessageBranchContext.Provider, { value: contextValue, children: _jsx("div", { className: cn("grid w-full gap-2 [&>div]:pb-0", className), ...props }) }));
};
export const MessageBranchContent = ({ children, ...props }) => {
    const { currentBranch, setBranches, branches } = useMessageBranch();
    const childrenArray = Array.isArray(children) ? children : [children];
    // Use useEffect to update branches when they change
    useEffect(() => {
        if (branches.length !== childrenArray.length) {
            setBranches(childrenArray);
        }
    }, [childrenArray, branches, setBranches]);
    return childrenArray.map((branch, index) => (_jsx("div", { className: cn("grid gap-2 overflow-hidden [&>div]:pb-0", index === currentBranch ? "block" : "hidden"), ...props, children: branch }, branch.key)));
};
export const MessageBranchSelector = ({ className, from, ...props }) => {
    const { totalBranches } = useMessageBranch();
    // Don't render if there's only one branch
    if (totalBranches <= 1) {
        return null;
    }
    return (_jsx(ButtonGroup, { className: "[&>*:not(:first-child)]:rounded-l-md [&>*:not(:last-child)]:rounded-r-md", orientation: "horizontal", ...props }));
};
export const MessageBranchPrevious = ({ children, ...props }) => {
    const { goToPrevious, totalBranches } = useMessageBranch();
    return (_jsx(Button, { "aria-label": "Previous branch", disabled: totalBranches <= 1, onClick: goToPrevious, size: "icon-sm", type: "button", variant: "ghost", ...props, children: children ?? _jsx(ChevronLeftIcon, { size: 14 }) }));
};
export const MessageBranchNext = ({ children, className, ...props }) => {
    const { goToNext, totalBranches } = useMessageBranch();
    return (_jsx(Button, { "aria-label": "Next branch", disabled: totalBranches <= 1, onClick: goToNext, size: "icon-sm", type: "button", variant: "ghost", ...props, children: children ?? _jsx(ChevronRightIcon, { size: 14 }) }));
};
export const MessageBranchPage = ({ className, ...props }) => {
    const { currentBranch, totalBranches } = useMessageBranch();
    return (_jsxs(ButtonGroupText, { className: cn("border-none bg-transparent text-muted-foreground shadow-none", className), ...props, children: [currentBranch + 1, " of ", totalBranches] }));
};
export const MessageResponse = memo(({ className, ...props }) => {
    // const theme = useTheme();
    // const [mode] = useColorMode()
    return (_jsx(Streamdown, { controls: true, className: cn("size-full [&>*:first-child]:mt-0 [&>*:last-child]:mb-0", className), shikiTheme: ["github-light", "github-dark"], rehypePlugins: [
            rehypeKatex
        ], remarkPlugins: [
            remarkGfm,
            remarkMath
        ], ...props }));
}, (prevProps, nextProps) => prevProps.children === nextProps.children);
MessageResponse.displayName = "MessageResponse";
export function MessageAttachment({ data, className, onRemove, ...props }) {
    const filename = data.filename || "";
    const mediaType = data.mediaType?.startsWith("image/") && data.url ? "image" : "file";
    const isImage = mediaType === "image";
    const attachmentLabel = filename || (isImage ? "Image" : "Attachment");
    return (_jsx("div", { className: cn("group relative size-24 overflow-hidden rounded-lg", className), ...props, children: isImage ? (_jsxs(_Fragment, { children: [_jsx("img", { alt: filename || "attachment", className: "size-full object-cover", height: 100, src: data.url, width: 100 }), onRemove && (_jsxs(Button, { "aria-label": "Remove attachment", className: "absolute top-2 right-2 size-6 rounded-full bg-background/80 p-0 opacity-0 backdrop-blur-sm transition-opacity hover:bg-background group-hover:opacity-100 [&>svg]:size-3", onClick: (e) => {
                        e.stopPropagation();
                        onRemove();
                    }, type: "button", variant: "ghost", children: [_jsx(XIcon, {}), _jsx("span", { className: "sr-only", children: "Remove" })] }))] })) : (_jsxs(_Fragment, { children: [_jsxs(Tooltip, { children: [_jsx(TooltipTrigger, { asChild: true, children: _jsx("div", { className: "flex size-full shrink-0 items-center justify-center rounded-lg bg-muted text-muted-foreground", children: _jsx(PaperclipIcon, { className: "size-4" }) }) }), _jsx(TooltipContent, { children: _jsx("p", { children: attachmentLabel }) })] }), onRemove && (_jsxs(Button, { "aria-label": "Remove attachment", className: "size-6 shrink-0 rounded-full p-0 opacity-0 transition-opacity hover:bg-accent group-hover:opacity-100 [&>svg]:size-3", onClick: (e) => {
                        e.stopPropagation();
                        onRemove();
                    }, type: "button", variant: "ghost", children: [_jsx(XIcon, {}), _jsx("span", { className: "sr-only", children: "Remove" })] }))] })) }));
}
export function MessageAttachments({ children, className, ...props }) {
    if (!children) {
        return null;
    }
    return (_jsx("div", { className: cn("ml-auto flex w-fit flex-wrap items-start gap-2", className), ...props, children: children }));
}
export const MessageToolbar = ({ className, children, ...props }) => (_jsx("div", { className: cn("mt-4 flex w-full items-center justify-between gap-4", className), ...props, children: children }));
