import type { HTMLAttributes } from "react";
export type LoaderProps = HTMLAttributes<HTMLDivElement> & {
    size?: number;
};
export declare const Loader: ({ className, size, ...props }: LoaderProps) => import("react/jsx-runtime").JSX.Element;
