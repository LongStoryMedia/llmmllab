import { Button } from "../ui/button";
import { Collapsible, CollapsibleContent } from "../ui/collapsible";
import { ScrollArea } from "../ui/scroll-area";
import type { ComponentProps } from "react";
export type QueueMessagePart = {
    type: string;
    text?: string;
    url?: string;
    filename?: string;
    mediaType?: string;
};
export type QueueMessage = {
    id: string;
    parts: QueueMessagePart[];
};
export type QueueTodo = {
    id: string;
    title: string;
    description?: string;
    status?: "pending" | "completed";
};
export type QueueItemProps = ComponentProps<"li">;
export declare const QueueItem: ({ className, ...props }: QueueItemProps) => import("react/jsx-runtime").JSX.Element;
export type QueueItemIndicatorProps = ComponentProps<"span"> & {
    completed?: boolean;
};
export declare const QueueItemIndicator: ({ completed, className, ...props }: QueueItemIndicatorProps) => import("react/jsx-runtime").JSX.Element;
export type QueueItemContentProps = ComponentProps<"span"> & {
    completed?: boolean;
};
export declare const QueueItemContent: ({ completed, className, ...props }: QueueItemContentProps) => import("react/jsx-runtime").JSX.Element;
export type QueueItemDescriptionProps = ComponentProps<"div"> & {
    completed?: boolean;
};
export declare const QueueItemDescription: ({ completed, className, ...props }: QueueItemDescriptionProps) => import("react/jsx-runtime").JSX.Element;
export type QueueItemActionsProps = ComponentProps<"div">;
export declare const QueueItemActions: ({ className, ...props }: QueueItemActionsProps) => import("react/jsx-runtime").JSX.Element;
export type QueueItemActionProps = Omit<ComponentProps<typeof Button>, "variant" | "size">;
export declare const QueueItemAction: ({ className, ...props }: QueueItemActionProps) => import("react/jsx-runtime").JSX.Element;
export type QueueItemAttachmentProps = ComponentProps<"div">;
export declare const QueueItemAttachment: ({ className, ...props }: QueueItemAttachmentProps) => import("react/jsx-runtime").JSX.Element;
export type QueueItemImageProps = ComponentProps<"img">;
export declare const QueueItemImage: ({ className, ...props }: QueueItemImageProps) => import("react/jsx-runtime").JSX.Element;
export type QueueItemFileProps = ComponentProps<"span">;
export declare const QueueItemFile: ({ children, className, ...props }: QueueItemFileProps) => import("react/jsx-runtime").JSX.Element;
export type QueueListProps = ComponentProps<typeof ScrollArea>;
export declare const QueueList: ({ children, className, ...props }: QueueListProps) => import("react/jsx-runtime").JSX.Element;
export type QueueSectionProps = ComponentProps<typeof Collapsible>;
export declare const QueueSection: ({ className, defaultOpen, ...props }: QueueSectionProps) => import("react/jsx-runtime").JSX.Element;
export type QueueSectionTriggerProps = ComponentProps<"button">;
export declare const QueueSectionTrigger: ({ children, className, ...props }: QueueSectionTriggerProps) => import("react/jsx-runtime").JSX.Element;
export type QueueSectionLabelProps = ComponentProps<"span"> & {
    count?: number;
    label: string;
    icon?: React.ReactNode;
};
export declare const QueueSectionLabel: ({ count, label, icon, className, ...props }: QueueSectionLabelProps) => import("react/jsx-runtime").JSX.Element;
export type QueueSectionContentProps = ComponentProps<typeof CollapsibleContent>;
export declare const QueueSectionContent: ({ className, ...props }: QueueSectionContentProps) => import("react/jsx-runtime").JSX.Element;
export type QueueProps = ComponentProps<"div">;
export declare const Queue: ({ className, ...props }: QueueProps) => import("react/jsx-runtime").JSX.Element;
