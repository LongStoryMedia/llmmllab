"use client";
import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Button } from "../ui/button";
import { Card, CardAction, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "../ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "../ui/collapsible";
import { cn } from "../../lib/utils";
import { ChevronsUpDownIcon } from "lucide-react";
import { createContext, useContext } from "react";
import { Shimmer } from "./shimmer";
const PlanContext = createContext(null);
const usePlan = () => {
    const context = useContext(PlanContext);
    if (!context) {
        throw new Error("Plan components must be used within Plan");
    }
    return context;
};
export const Plan = ({ className, isStreaming = false, children, ...props }) => (_jsx(PlanContext.Provider, { value: { isStreaming }, children: _jsx(Collapsible, { asChild: true, "data-slot": "plan", ...props, children: _jsx(Card, { className: cn("shadow-none", className), children: children }) }) }));
export const PlanHeader = ({ className, ...props }) => (_jsx(CardHeader, { className: cn("flex items-start justify-between", className), "data-slot": "plan-header", ...props }));
export const PlanTitle = ({ children, ...props }) => {
    const { isStreaming } = usePlan();
    return (_jsx(CardTitle, { "data-slot": "plan-title", ...props, children: isStreaming ? _jsx(Shimmer, { children: children }) : children }));
};
export const PlanDescription = ({ className, children, ...props }) => {
    const { isStreaming } = usePlan();
    return (_jsx(CardDescription, { className: cn("text-balance", className), "data-slot": "plan-description", ...props, children: isStreaming ? _jsx(Shimmer, { children: children }) : children }));
};
export const PlanAction = (props) => (_jsx(CardAction, { "data-slot": "plan-action", ...props }));
export const PlanContent = (props) => (_jsx(CollapsibleContent, { asChild: true, children: _jsx(CardContent, { "data-slot": "plan-content", ...props }) }));
export const PlanFooter = (props) => (_jsx(CardFooter, { "data-slot": "plan-footer", ...props }));
export const PlanTrigger = ({ className, ...props }) => (_jsx(CollapsibleTrigger, { asChild: true, children: _jsxs(Button, { className: cn("size-8", className), "data-slot": "plan-trigger", size: "icon", variant: "ghost", ...props, children: [_jsx(ChevronsUpDownIcon, { className: "size-4" }), _jsx("span", { className: "sr-only", children: "Toggle plan" })] }) }));
