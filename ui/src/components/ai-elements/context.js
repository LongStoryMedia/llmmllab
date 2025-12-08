"use client";
import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { Button } from "../ui/button";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "../ui/hover-card";
import { Progress } from "../ui/progress";
import { cn } from "../../lib/utils";
import { createContext, useContext } from "react";
import { getUsage } from "tokenlens";
const PERCENT_MAX = 100;
const ICON_RADIUS = 10;
const ICON_VIEWBOX = 24;
const ICON_CENTER = 12;
const ICON_STROKE_WIDTH = 2;
const ContextContext = createContext(null);
const useContextValue = () => {
    const context = useContext(ContextContext);
    if (!context) {
        throw new Error("Context components must be used within Context");
    }
    return context;
};
export const Context = ({ usedTokens, maxTokens, usage, modelId, ...props }) => (_jsx(ContextContext.Provider, { value: {
        usedTokens,
        maxTokens,
        usage,
        modelId
    }, children: _jsx(HoverCard, { closeDelay: 0, openDelay: 0, ...props }) }));
const ContextIcon = () => {
    const { usedTokens, maxTokens } = useContextValue();
    const circumference = 2 * Math.PI * ICON_RADIUS;
    const usedPercent = usedTokens / maxTokens;
    const dashOffset = circumference * (1 - usedPercent);
    return (_jsxs("svg", { "aria-label": "Model context usage", height: "20", role: "img", style: { color: "currentcolor" }, viewBox: `0 0 ${ICON_VIEWBOX} ${ICON_VIEWBOX}`, width: "20", children: [_jsx("circle", { cx: ICON_CENTER, cy: ICON_CENTER, fill: "none", opacity: "0.25", r: ICON_RADIUS, stroke: "currentColor", strokeWidth: ICON_STROKE_WIDTH }), _jsx("circle", { cx: ICON_CENTER, cy: ICON_CENTER, fill: "none", opacity: "0.7", r: ICON_RADIUS, stroke: "currentColor", strokeDasharray: `${circumference} ${circumference}`, strokeDashoffset: dashOffset, strokeLinecap: "round", strokeWidth: ICON_STROKE_WIDTH, style: { transformOrigin: "center", transform: "rotate(-90deg)" } })] }));
};
export const ContextTrigger = ({ children, ...props }) => {
    const { usedTokens, maxTokens } = useContextValue();
    const usedPercent = usedTokens / maxTokens;
    const renderedPercent = new Intl.NumberFormat("en-US", {
        style: "percent",
        maximumFractionDigits: 1
    }).format(usedPercent);
    return (_jsx(HoverCardTrigger, { asChild: true, children: children ?? (_jsxs(Button, { type: "button", variant: "ghost", ...props, children: [_jsx("span", { className: "font-medium text-muted-foreground", children: renderedPercent }), _jsx(ContextIcon, {})] })) }));
};
export const ContextContent = ({ className, ...props }) => (_jsx(HoverCardContent, { className: cn("min-w-60 divide-y overflow-hidden p-0", className), ...props }));
export const ContextContentHeader = ({ children, className, ...props }) => {
    const { usedTokens, maxTokens } = useContextValue();
    const usedPercent = usedTokens / maxTokens;
    const displayPct = new Intl.NumberFormat("en-US", {
        style: "percent",
        maximumFractionDigits: 1
    }).format(usedPercent);
    const used = new Intl.NumberFormat("en-US", {
        notation: "compact"
    }).format(usedTokens);
    const total = new Intl.NumberFormat("en-US", {
        notation: "compact"
    }).format(maxTokens);
    return (_jsx("div", { className: cn("w-full space-y-2 p-3", className), ...props, children: children ?? (_jsxs(_Fragment, { children: [_jsxs("div", { className: "flex items-center justify-between gap-3 text-xs", children: [_jsx("p", { children: displayPct }), _jsxs("p", { className: "font-mono text-muted-foreground", children: [used, " / ", total] })] }), _jsx("div", { className: "space-y-2", children: _jsx(Progress, { className: "bg-muted", value: usedPercent * PERCENT_MAX }) })] })) }));
};
export const ContextContentBody = ({ children, className, ...props }) => (_jsx("div", { className: cn("w-full p-3", className), ...props, children: children }));
export const ContextContentFooter = ({ children, className, ...props }) => {
    const { modelId, usage } = useContextValue();
    const costUSD = modelId
        ? getUsage({
            modelId,
            usage: {
                input: usage?.inputTokens ?? 0,
                output: usage?.outputTokens ?? 0
            }
        }).costUSD?.totalUSD
        : undefined;
    const totalCost = new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD"
    }).format(costUSD ?? 0);
    return (_jsx("div", { className: cn("flex w-full items-center justify-between gap-3 bg-secondary p-3 text-xs", className), ...props, children: children ?? (_jsxs(_Fragment, { children: [_jsx("span", { className: "text-muted-foreground", children: "Total cost" }), _jsx("span", { children: totalCost })] })) }));
};
export const ContextInputUsage = ({ className, children, ...props }) => {
    const { usage, modelId } = useContextValue();
    const inputTokens = usage?.inputTokens ?? 0;
    if (children) {
        return children;
    }
    if (!inputTokens) {
        return null;
    }
    const inputCost = modelId
        ? getUsage({
            modelId,
            usage: { input: inputTokens, output: 0 }
        }).costUSD?.totalUSD
        : undefined;
    const inputCostText = new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD"
    }).format(inputCost ?? 0);
    return (_jsxs("div", { className: cn("flex items-center justify-between text-xs", className), ...props, children: [_jsx("span", { className: "text-muted-foreground", children: "Input" }), _jsx(TokensWithCost, { costText: inputCostText, tokens: inputTokens })] }));
};
export const ContextOutputUsage = ({ className, children, ...props }) => {
    const { usage, modelId } = useContextValue();
    const outputTokens = usage?.outputTokens ?? 0;
    if (children) {
        return children;
    }
    if (!outputTokens) {
        return null;
    }
    const outputCost = modelId
        ? getUsage({
            modelId,
            usage: { input: 0, output: outputTokens }
        }).costUSD?.totalUSD
        : undefined;
    const outputCostText = new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD"
    }).format(outputCost ?? 0);
    return (_jsxs("div", { className: cn("flex items-center justify-between text-xs", className), ...props, children: [_jsx("span", { className: "text-muted-foreground", children: "Output" }), _jsx(TokensWithCost, { costText: outputCostText, tokens: outputTokens })] }));
};
export const ContextReasoningUsage = ({ className, children, ...props }) => {
    const { usage, modelId } = useContextValue();
    const reasoningTokens = usage?.reasoningTokens ?? 0;
    if (children) {
        return children;
    }
    if (!reasoningTokens) {
        return null;
    }
    const reasoningCost = modelId
        ? getUsage({
            modelId,
            usage: { reasoningTokens }
        }).costUSD?.totalUSD
        : undefined;
    const reasoningCostText = new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD"
    }).format(reasoningCost ?? 0);
    return (_jsxs("div", { className: cn("flex items-center justify-between text-xs", className), ...props, children: [_jsx("span", { className: "text-muted-foreground", children: "Reasoning" }), _jsx(TokensWithCost, { costText: reasoningCostText, tokens: reasoningTokens })] }));
};
export const ContextCacheUsage = ({ className, children, ...props }) => {
    const { usage, modelId } = useContextValue();
    const cacheTokens = usage?.cachedInputTokens ?? 0;
    if (children) {
        return children;
    }
    if (!cacheTokens) {
        return null;
    }
    const cacheCost = modelId
        ? getUsage({
            modelId,
            usage: { cacheReads: cacheTokens, input: 0, output: 0 }
        }).costUSD?.totalUSD
        : undefined;
    const cacheCostText = new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD"
    }).format(cacheCost ?? 0);
    return (_jsxs("div", { className: cn("flex items-center justify-between text-xs", className), ...props, children: [_jsx("span", { className: "text-muted-foreground", children: "Cache" }), _jsx(TokensWithCost, { costText: cacheCostText, tokens: cacheTokens })] }));
};
const TokensWithCost = ({ tokens, costText }) => (_jsxs("span", { children: [tokens === undefined
            ? "—"
            : new Intl.NumberFormat("en-US", {
                notation: "compact"
            }).format(tokens), costText ? (_jsxs("span", { className: "ml-2 text-muted-foreground", children: ["\u2022 ", costText] })) : null] }));
