import { Button } from "../ui/button";
import { ScrollArea } from "../ui/scroll-area";
import type { ComponentProps } from "react";
export type SuggestionsProps = ComponentProps<typeof ScrollArea>;
export declare const Suggestions: ({ className, children, ...props }: SuggestionsProps) => import("react/jsx-runtime").JSX.Element;
export type SuggestionProps = Omit<ComponentProps<typeof Button>, "onClick"> & {
    suggestion: string;
    onClick?: (suggestion: string) => void;
};
export declare const Suggestion: ({ suggestion, onClick, className, variant, size, children, ...props }: SuggestionProps) => import("react/jsx-runtime").JSX.Element;
