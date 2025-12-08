"use client";
import { jsx as _jsx } from "react/jsx-runtime";
import { Alert, AlertDescription } from "../ui/alert";
import { Button } from "../ui/button";
import { cn } from "../../lib/utils";
import { createContext, useContext } from "react";
const ConfirmationContext = createContext(null);
const useConfirmation = () => {
    const context = useContext(ConfirmationContext);
    if (!context) {
        throw new Error("Confirmation components must be used within Confirmation");
    }
    return context;
};
export const Confirmation = ({ className, approval, state, ...props }) => {
    if (!approval || state === "input-streaming" || state === "input-available") {
        return null;
    }
    return (_jsx(ConfirmationContext.Provider, { value: { approval, state }, children: _jsx(Alert, { className: cn("flex flex-col gap-2", className), ...props }) }));
};
export const ConfirmationTitle = ({ className, ...props }) => (_jsx(AlertDescription, { className: cn("inline", className), ...props }));
export const ConfirmationRequest = ({ children }) => {
    const { state } = useConfirmation();
    // Only show when tool is available for input (equivalent to approval-requested)
    if (state !== "input-available") {
        return null;
    }
    return children;
};
export const ConfirmationAccepted = ({ children }) => {
    const { approval, state } = useConfirmation();
    // Only show when approved and in output states
    if (!approval?.approved ||
        (state !== "output-available" &&
            state !== "output-error")) {
        return null;
    }
    return children;
};
export const ConfirmationRejected = ({ children }) => {
    const { approval, state } = useConfirmation();
    // Only show when rejected and in error state
    if (approval?.approved !== false ||
        state !== "output-error") {
        return null;
    }
    return children;
};
export const ConfirmationActions = ({ className, ...props }) => {
    const { state } = useConfirmation();
    // Only show when tool is available for input
    if (state !== "input-available") {
        return null;
    }
    return (_jsx("div", { className: cn("flex items-center justify-end gap-2 self-end", className), ...props }));
};
export const ConfirmationAction = (props) => (_jsx(Button, { className: "h-8 px-3 text-sm", type: "button", ...props }));
