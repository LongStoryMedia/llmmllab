"use client";
import { jsx as _jsx } from "react/jsx-runtime";
import { cn } from "../../lib/utils";
import { Controls as ControlsPrimitive } from "@xyflow/react";
export const Controls = ({ className, ...props }) => (_jsx(ControlsPrimitive, { className: cn("gap-px overflow-hidden rounded-md border bg-card p-1 shadow-none!", "[&>button]:rounded-md [&>button]:border-none! [&>button]:bg-transparent! [&>button]:hover:bg-secondary!", className), ...props }));
