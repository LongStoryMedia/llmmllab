import { Button } from "../ui/button";
import { type ComponentProps, type HTMLAttributes } from "react";
import { type BundledLanguage } from "shiki";
type CodeBlockProps = HTMLAttributes<HTMLDivElement> & {
    code: string;
    language: BundledLanguage;
    showLineNumbers?: boolean;
};
export declare function highlightCode(code: string, language: BundledLanguage, showLineNumbers?: boolean): Promise<[string, string]>;
export declare const CodeBlock: ({ code, language, showLineNumbers, className, children, ...props }: CodeBlockProps) => import("react/jsx-runtime").JSX.Element;
export type CodeBlockCopyButtonProps = ComponentProps<typeof Button> & {
    onCopy?: () => void;
    onError?: (error: Error) => void;
    timeout?: number;
};
export declare const CodeBlockCopyButton: ({ onCopy, onError, timeout, children, className, ...props }: CodeBlockCopyButtonProps) => import("react/jsx-runtime").JSX.Element;
export {};
