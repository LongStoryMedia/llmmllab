import { Alert, AlertDescription } from "../ui/alert";
import { Button } from "../ui/button";
import type { ToolUIPart } from "ai";
import { type ComponentProps, type ReactNode } from "react";
type ToolUIPartApproval = {
    id: string;
    approved?: never;
    reason?: never;
} | {
    id: string;
    approved: boolean;
    reason?: string;
} | {
    id: string;
    approved: true;
    reason?: string;
} | {
    id: string;
    approved: true;
    reason?: string;
} | {
    id: string;
    approved: false;
    reason?: string;
} | undefined;
export type ConfirmationProps = ComponentProps<typeof Alert> & {
    approval?: ToolUIPartApproval;
    state: ToolUIPart["state"];
};
export declare const Confirmation: ({ className, approval, state, ...props }: ConfirmationProps) => import("react/jsx-runtime").JSX.Element | null;
export type ConfirmationTitleProps = ComponentProps<typeof AlertDescription>;
export declare const ConfirmationTitle: ({ className, ...props }: ConfirmationTitleProps) => import("react/jsx-runtime").JSX.Element;
export type ConfirmationRequestProps = {
    children?: ReactNode;
};
export declare const ConfirmationRequest: ({ children }: ConfirmationRequestProps) => ReactNode;
export type ConfirmationAcceptedProps = {
    children?: ReactNode;
};
export declare const ConfirmationAccepted: ({ children }: ConfirmationAcceptedProps) => ReactNode;
export type ConfirmationRejectedProps = {
    children?: ReactNode;
};
export declare const ConfirmationRejected: ({ children }: ConfirmationRejectedProps) => ReactNode;
export type ConfirmationActionsProps = ComponentProps<"div">;
export declare const ConfirmationActions: ({ className, ...props }: ConfirmationActionsProps) => import("react/jsx-runtime").JSX.Element | null;
export type ConfirmationActionProps = ComponentProps<typeof Button>;
export declare const ConfirmationAction: (props: ConfirmationActionProps) => import("react/jsx-runtime").JSX.Element;
export {};
