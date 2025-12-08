import { CollapsibleContent, CollapsibleTrigger } from "../ui/collapsible";
import type { ComponentProps } from "react";
export type SourcesProps = ComponentProps<"div">;
export declare const Sources: ({ className, ...props }: SourcesProps) => import("react/jsx-runtime").JSX.Element;
export type SourcesTriggerProps = ComponentProps<typeof CollapsibleTrigger> & {
    count: number;
};
export declare const SourcesTrigger: ({ className, count, children, ...props }: SourcesTriggerProps) => import("react/jsx-runtime").JSX.Element;
export type SourcesContentProps = ComponentProps<typeof CollapsibleContent>;
export declare const SourcesContent: ({ className, ...props }: SourcesContentProps) => import("react/jsx-runtime").JSX.Element;
export type SourceProps = ComponentProps<"a">;
export declare const Source: ({ href, title, children, ...props }: SourceProps) => import("react/jsx-runtime").JSX.Element;
