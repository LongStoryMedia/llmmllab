import { CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card";
import { Collapsible, CollapsibleTrigger } from "../ui/collapsible";
import type { ComponentProps } from "react";
export type PlanProps = ComponentProps<typeof Collapsible> & {
    isStreaming?: boolean;
};
export declare const Plan: ({ className, isStreaming, children, ...props }: PlanProps) => import("react/jsx-runtime").JSX.Element;
export type PlanHeaderProps = ComponentProps<typeof CardHeader>;
export declare const PlanHeader: ({ className, ...props }: PlanHeaderProps) => import("react/jsx-runtime").JSX.Element;
export type PlanTitleProps = Omit<ComponentProps<typeof CardTitle>, "children"> & {
    children: string;
};
export declare const PlanTitle: ({ children, ...props }: PlanTitleProps) => import("react/jsx-runtime").JSX.Element;
export type PlanDescriptionProps = Omit<ComponentProps<typeof CardDescription>, "children"> & {
    children: string;
};
export declare const PlanDescription: ({ className, children, ...props }: PlanDescriptionProps) => import("react/jsx-runtime").JSX.Element;
export type PlanActionProps = ComponentProps<typeof CardAction>;
export declare const PlanAction: (props: PlanActionProps) => import("react/jsx-runtime").JSX.Element;
export type PlanContentProps = ComponentProps<typeof CardContent>;
export declare const PlanContent: (props: PlanContentProps) => import("react/jsx-runtime").JSX.Element;
export type PlanFooterProps = ComponentProps<"div">;
export declare const PlanFooter: (props: PlanFooterProps) => import("react/jsx-runtime").JSX.Element;
export type PlanTriggerProps = ComponentProps<typeof CollapsibleTrigger>;
export declare const PlanTrigger: ({ className, ...props }: PlanTriggerProps) => import("react/jsx-runtime").JSX.Element;
