import { Button } from "../ui/button";
import type { FileUIPart, UIMessage } from "ai";
import type { ComponentProps, HTMLAttributes } from "react";
import { Streamdown } from "streamdown";
export type MessageProps = HTMLAttributes<HTMLDivElement> & {
    from: UIMessage["role"];
};
export declare const Message: ({ className, from, ...props }: MessageProps) => import("react/jsx-runtime").JSX.Element;
export type MessageContentProps = HTMLAttributes<HTMLDivElement>;
export declare const MessageContent: ({ children, className, ...props }: MessageContentProps) => import("react/jsx-runtime").JSX.Element;
export type MessageActionsProps = ComponentProps<"div">;
export declare const MessageActions: ({ className, children, ...props }: MessageActionsProps) => import("react/jsx-runtime").JSX.Element;
export type MessageActionProps = ComponentProps<typeof Button> & {
    tooltip?: string;
    label?: string;
};
export declare const MessageAction: ({ tooltip, children, label, variant, size, ...props }: MessageActionProps) => import("react/jsx-runtime").JSX.Element;
export type MessageBranchProps = HTMLAttributes<HTMLDivElement> & {
    defaultBranch?: number;
    onBranchChange?: (branchIndex: number) => void;
};
export declare const MessageBranch: ({ defaultBranch, onBranchChange, className, ...props }: MessageBranchProps) => import("react/jsx-runtime").JSX.Element;
export type MessageBranchContentProps = HTMLAttributes<HTMLDivElement>;
export declare const MessageBranchContent: ({ children, ...props }: MessageBranchContentProps) => import("react/jsx-runtime").JSX.Element[];
export type MessageBranchSelectorProps = HTMLAttributes<HTMLDivElement> & {
    from: UIMessage["role"];
};
export declare const MessageBranchSelector: ({ className, from, ...props }: MessageBranchSelectorProps) => import("react/jsx-runtime").JSX.Element | null;
export type MessageBranchPreviousProps = ComponentProps<typeof Button>;
export declare const MessageBranchPrevious: ({ children, ...props }: MessageBranchPreviousProps) => import("react/jsx-runtime").JSX.Element;
export type MessageBranchNextProps = ComponentProps<typeof Button>;
export declare const MessageBranchNext: ({ children, className, ...props }: MessageBranchNextProps) => import("react/jsx-runtime").JSX.Element;
export type MessageBranchPageProps = HTMLAttributes<HTMLSpanElement>;
export declare const MessageBranchPage: ({ className, ...props }: MessageBranchPageProps) => import("react/jsx-runtime").JSX.Element;
export type MessageResponseProps = ComponentProps<typeof Streamdown>;
export declare const MessageResponse: import("react").MemoExoticComponent<({ className, ...props }: MessageResponseProps) => import("react/jsx-runtime").JSX.Element>;
export type MessageAttachmentProps = HTMLAttributes<HTMLDivElement> & {
    data: FileUIPart;
    className?: string;
    onRemove?: () => void;
};
export declare function MessageAttachment({ data, className, onRemove, ...props }: MessageAttachmentProps): import("react/jsx-runtime").JSX.Element;
export type MessageAttachmentsProps = ComponentProps<"div">;
export declare function MessageAttachments({ children, className, ...props }: MessageAttachmentsProps): import("react/jsx-runtime").JSX.Element | null;
export type MessageToolbarProps = ComponentProps<"div">;
export declare const MessageToolbar: ({ className, children, ...props }: MessageToolbarProps) => import("react/jsx-runtime").JSX.Element;
