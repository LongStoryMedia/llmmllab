import { type ElementType } from "react";
export type TextShimmerProps = {
    children: string;
    as?: ElementType;
    className?: string;
    duration?: number;
    spread?: number;
};
export declare const Shimmer: import("react").MemoExoticComponent<({ children, as: Component, className, duration, spread }: TextShimmerProps) => import("react/jsx-runtime").JSX.Element>;
