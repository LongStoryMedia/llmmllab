import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Background, ReactFlow } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
export const Canvas = ({ children, ...props }) => (_jsxs(ReactFlow, { deleteKeyCode: ["Backspace", "Delete"], fitView: true, panOnDrag: false, panOnScroll: true, selectionOnDrag: true, zoomOnDoubleClick: false, ...props, children: [_jsx(Background, { bgColor: "var(--sidebar)" }), children] }));
