import { type EdgeProps } from "@xyflow/react";
export declare const Edge: {
    Temporary: ({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition }: EdgeProps) => import("react/jsx-runtime").JSX.Element;
    Animated: ({ id, source, target, markerEnd, style }: EdgeProps) => import("react/jsx-runtime").JSX.Element | null;
};
