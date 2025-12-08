"use client";
import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Button } from "../ui/button";
import { cn } from "../../lib/utils";
import { CheckIcon, CopyIcon } from "lucide-react";
import { createContext, useContext, useEffect, useRef, useState } from "react";
import { codeToHtml } from "shiki";
const CodeBlockContext = createContext({
    code: ""
});
const lineNumberTransformer = {
    name: "line-numbers",
    line(node, line) {
        node.children.unshift({
            type: "element",
            tagName: "span",
            properties: {
                className: [
                    "inline-block",
                    "min-w-10",
                    "mr-4",
                    "text-right",
                    "select-none",
                    "text-muted-foreground"
                ]
            },
            children: [{ type: "text", value: String(line) }]
        });
    }
};
export async function highlightCode(code, language, showLineNumbers = false) {
    const transformers = showLineNumbers
        ? [lineNumberTransformer]
        : [];
    return await Promise.all([
        codeToHtml(code, {
            lang: language,
            theme: "one-light",
            transformers
        }),
        codeToHtml(code, {
            lang: language,
            theme: "one-dark-pro",
            transformers
        })
    ]);
}
export const CodeBlock = ({ code, language, showLineNumbers = false, className, children, ...props }) => {
    const [html, setHtml] = useState("");
    const [darkHtml, setDarkHtml] = useState("");
    const mounted = useRef(false);
    useEffect(() => {
        highlightCode(code, language, showLineNumbers).then(([light, dark]) => {
            if (!mounted.current) {
                setHtml(light);
                setDarkHtml(dark);
                mounted.current = true;
            }
        });
        return () => {
            mounted.current = false;
        };
    }, [code, language, showLineNumbers]);
    return (_jsx(CodeBlockContext.Provider, { value: { code }, children: _jsx("div", { className: cn("group relative w-full overflow-hidden rounded-md border bg-background text-foreground", className), ...props, children: _jsxs("div", { className: "relative", children: [_jsx("div", { className: "overflow-hidden dark:hidden [&>pre]:m-0 [&>pre]:bg-background! [&>pre]:p-4 [&>pre]:text-foreground! [&>pre]:text-sm [&_code]:font-mono [&_code]:text-sm", 
                        // biome-ignore lint/security/noDangerouslySetInnerHtml: "this is needed."
                        dangerouslySetInnerHTML: { __html: html } }), _jsx("div", { className: "hidden overflow-hidden dark:block [&>pre]:m-0 [&>pre]:bg-background! [&>pre]:p-4 [&>pre]:text-foreground! [&>pre]:text-sm [&_code]:font-mono [&_code]:text-sm", 
                        // biome-ignore lint/security/noDangerouslySetInnerHtml: "this is needed."
                        dangerouslySetInnerHTML: { __html: darkHtml } }), children && (_jsx("div", { className: "absolute top-2 right-2 flex items-center gap-2", children: children }))] }) }) }));
};
export const CodeBlockCopyButton = ({ onCopy, onError, timeout = 2000, children, className, ...props }) => {
    const [isCopied, setIsCopied] = useState(false);
    const { code } = useContext(CodeBlockContext);
    const copyToClipboard = async () => {
        if (typeof window === "undefined" || !navigator?.clipboard?.writeText) {
            onError?.(new Error("Clipboard API not available"));
            return;
        }
        try {
            await navigator.clipboard.writeText(code);
            setIsCopied(true);
            onCopy?.();
            setTimeout(() => setIsCopied(false), timeout);
        }
        catch (error) {
            onError?.(error);
        }
    };
    const Icon = isCopied ? CheckIcon : CopyIcon;
    return (_jsx(Button, { className: cn("shrink-0", className), onClick: copyToClipboard, size: "icon", variant: "ghost", ...props, children: children ?? _jsx(Icon, { size: 14 }) }));
};
