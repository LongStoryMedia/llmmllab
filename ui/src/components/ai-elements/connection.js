import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
const HALF = 0.5;
export const Connection = ({ fromX, fromY, toX, toY }) => (_jsxs("g", { children: [_jsx("path", { className: "animated", d: `M${fromX},${fromY} C ${fromX + (toX - fromX) * HALF},${fromY} ${fromX + (toX - fromX) * HALF},${toY} ${toX},${toY}`, fill: "none", stroke: "var(--color-ring)", strokeWidth: 1 }), _jsx("circle", { cx: toX, cy: toY, fill: "#fff", r: 3, stroke: "var(--color-ring)", strokeWidth: 1 })] }));
