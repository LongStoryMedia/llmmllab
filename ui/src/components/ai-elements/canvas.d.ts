import { type ReactFlowProps } from "@xyflow/react";
import type { ReactNode } from "react";
import "@xyflow/react/dist/style.css";
type CanvasProps = ReactFlowProps & {
    children?: ReactNode;
};
export declare const Canvas: ({ children, ...props }: CanvasProps) => import("react/jsx-runtime").JSX.Element;
export {};
