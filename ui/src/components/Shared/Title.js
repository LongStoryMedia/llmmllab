import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTheme } from "@mui/material";
const Title = ({ size = 150, speed = 5 }) => {
    const theme = useTheme();
    return (_jsxs("svg", { viewBox: "0 0 940 180", preserveAspectRatio: "none", width: size * 1.5, height: size / 1.25, xmlns: "http://www.w3.org/2000/svg", children: [_jsx("defs", { children: _jsxs("filter", { id: "glow", children: [_jsx("feGaussianBlur", { in: "SourceGraphic", stdDeviation: "1.5", result: "blur" }), _jsxs("feMerge", { children: [_jsx("feMergeNode", { in: "blur" }), _jsx("feMergeNode", { in: "SourceGraphic" })] })] }) }), _jsxs("g", { className: "title", children: [_jsx("path", { className: "trace", d: "M0 40 V140 H36" }), _jsx("path", { className: "leaf", d: "M6 70 Q-18 50 30 45 Q-6 80 42 65" }), _jsx("path", { className: "leaf", d: "M18 110 Q-12 100 30 90 Q0 130 47 110" }), _jsx("path", { className: "trace", d: "M59 40 V140 H95" }), _jsx("path", { className: "leaf", d: "M65 70 Q41 50 90 45 Q54 80 101 65" }), _jsx("path", { className: "leaf", d: "M77 110 Q47 100 90 90 Q59 130 107 110" }), _jsx("path", { className: "trace", d: "M142 40 V140 H178" }), _jsx("path", { className: "leaf", d: "M148 80 Q119 60 172 55 Q136 100 191 80" }), _jsx("path", { className: "leaf", d: "M160 120 Q130 110 184 100 Q153 140 201 120" }), _jsx("path", { className: "trace", d: "M201 140 L209 100 L218 60 L227 100 L235 140 M207 120 H232" }), _jsx("path", { className: "leaf", d: "M218 80 Q188 50 235 45 Q201 100 249 80" }), _jsx("path", { className: "leaf", d: "M225 120 Q195 110 243 100 Q212 140 257 120" }), _jsx("path", { className: "trace", d: "M260 40 V140 H289 Q307 125 289 80 Q307 65 289 40 H260" }), _jsx("path", { className: "leaf", d: "M284 60 Q321 30 303 65 Q289 60 326 70" }), _jsx("path", { className: "leaf", d: "M297 110 Q334 90 316 125 Q303 110 340 120" })] }), _jsx("style", { children: `
    .trace {
      stroke: ${theme.palette.secondary.main};
      stroke-width: 5;
      fill: none;
      stroke-dasharray: 1000;
      stroke-dashoffset: 1000;
      animation: draw var(--trace-speed, ${speed}s) ease-in-out infinite;
      filter: url(#glow);
    }
    .leaf {
      stroke: ${theme.palette.primary.main};
      stroke-width: 3.5;
      fill: none;
      stroke-linecap: round;
      stroke-linejoin: round;
      opacity: 0.95;
      filter: url(#glow);
      stroke-dasharray: 60;
      stroke-dashoffset: 60;
      animation: draw var(--trace-speed, ${speed}s) ease-in-out infinite;
      animation-delay: 0s;
    }
    .title {
      opacity: 0;
      animation: glowPulse ${speed / 8}s ease-in infinite alternate;
    }
    @keyframes draw {
      from {
        stroke-dashoffset: 1000;
      }
      to {
        stroke-dashoffset: 0;
      }
    }
    @keyframes glowPulse {
      to {
        opacity: 1;
      }
      from {
        opacity: 0.25;
      }
    }
  ` })] }));
};
export default Title;
