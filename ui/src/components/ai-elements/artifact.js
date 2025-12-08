"use client";
import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Button } from "../ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "../ui/tooltip";
import { cn } from "../../lib/utils";
import { XIcon } from "lucide-react";
export const Artifact = ({ className, ...props }) => (_jsx("div", { className: cn("flex flex-col overflow-hidden rounded-lg border bg-background shadow-sm", className), ...props }));
export const ArtifactHeader = ({ className, ...props }) => (_jsx("div", { className: cn("flex items-center justify-between border-b bg-muted/50 px-4 py-3", className), ...props }));
export const ArtifactClose = ({ className, children, size = "sm", variant = "ghost", ...props }) => (_jsxs(Button, { className: cn("size-8 p-0 text-muted-foreground hover:text-foreground", className), size: size, type: "button", variant: variant, ...props, children: [children ?? _jsx(XIcon, { className: "size-4" }), _jsx("span", { className: "sr-only", children: "Close" })] }));
export const ArtifactTitle = ({ className, ...props }) => (_jsx("p", { className: cn("font-medium text-foreground text-sm", className), ...props }));
export const ArtifactDescription = ({ className, ...props }) => (_jsx("p", { className: cn("text-muted-foreground text-sm", className), ...props }));
export const ArtifactActions = ({ className, ...props }) => (_jsx("div", { className: cn("flex items-center gap-1", className), ...props }));
export const ArtifactAction = ({ tooltip, label, icon: Icon, children, className, size = "sm", variant = "ghost", ...props }) => {
    const button = (_jsxs(Button, { className: cn("size-8 p-0 text-muted-foreground hover:text-foreground", className), size: size, type: "button", variant: variant, ...props, children: [Icon ? _jsx(Icon, { className: "size-4" }) : children, _jsx("span", { className: "sr-only", children: label || tooltip })] }));
    if (tooltip) {
        return (_jsx(TooltipProvider, { children: _jsxs(Tooltip, { children: [_jsx(TooltipTrigger, { asChild: true, children: button }), _jsx(TooltipContent, { children: _jsx("p", { children: tooltip }) })] }) }));
    }
    return button;
};
export const ArtifactContent = ({ className, ...props }) => (_jsx("div", { className: cn("flex-1 overflow-auto p-4", className), ...props }));
