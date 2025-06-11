import { useTheme } from "@mui/material";


interface TitleProps {
  size?: number;
  speed?: number;
}


const Title: React.FC<TitleProps> = ({size = 150, speed = 5 }) => {
  const theme = useTheme();
  return (
    <svg viewBox="0 0 940 180" preserveAspectRatio="none" width={size*1.5} height={size/1.25} xmlns="http://www.w3.org/2000/svg">
      <defs>
        <filter id="glow">
          <feGaussianBlur in="SourceGraphic" stdDeviation="1.5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <g className="title">
        {/* L (extra, shifted to x=0) */}
        <path className="trace" d="M0 40 V140 H36" />
        {/* L leaves (extra, shifted to x=0) */}
        <path className="leaf" d="M6 70 Q-18 50 30 45 Q-6 80 42 65" />
        <path className="leaf" d="M18 110 Q-12 100 30 90 Q0 130 47 110" />
        {/* L (was at 53, now stretched) */}
        <path className="trace" d="M59 40 V140 H95" />
        {/* L leaves (stretched) */}
        <path className="leaf" d="M65 70 Q41 50 90 45 Q54 80 101 65" />
        <path className="leaf" d="M77 110 Q47 100 90 90 Q59 130 107 110" />
        {/* L (was at 127, now stretched) */}
        <path className="trace" d="M142 40 V140 H178" />
        {/* L leaves */}
        <path className="leaf" d="M148 80 Q119 60 172 55 Q136 100 191 80" />
        <path className="leaf" d="M160 120 Q130 110 184 100 Q153 140 201 120" />
        {/* A (was at 180, now stretched) */}
        <path className="trace" d="M201 140 L209 100 L218 60 L227 100 L235 140 M207 120 H232" />
        {/* A leaves */}
        <path className="leaf" d="M218 80 Q188 50 235 45 Q201 100 249 80" />
        <path className="leaf" d="M225 120 Q195 110 243 100 Q212 140 257 120" />
        {/* B (now with 2 bumps) */}
        <path className="trace" d="M260 40 V140 H289 Q307 125 289 80 Q307 65 289 40 H260" />
        {/* B leaves */}
        <path className="leaf" d="M284 60 Q321 30 303 65 Q289 60 326 70" />
        <path className="leaf" d="M297 110 Q334 90 316 125 Q303 110 340 120" />
 
      </g>
      <style>{`
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
      animation: glowPulse ${speed/8}s ease-in infinite alternate;
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
  `}</style>
    </svg>
  );
}
export default Title;

//  {/* O (was at 302, now stretched) */}
//  <path className="trace" d="M338 110 L345 60 L369 50 L393 60 L400 110 L369 130 L338 110 Z" />
//  {/* O leaves */}
//  <path className="leaf" d="M369 55 Q345 20 381 50 Q369 55 393 30" />
//  <path className="leaf" d="M369 120 Q345 160 381 130 Q369 120 393 150" />
//  {/* R (was at 382, now stretched) */}
//  <path className="trace" d="M429 140 V40 H465 L490 60 L465 90 L490 140" />
//  {/* R leaves */}
//  <path className="leaf" d="M465 70 Q429 30 478 65 Q465 70 490 40" />
//  <path className="leaf" d="M472 110 Q436 150 484 125 Q472 110 496 140" />
//  {/* A (was at 456, now stretched) */}
//  <path className="trace" d="M511 140 L520 100 L529 60 L538 100 L547 140 M517 120 H543" />
//  {/* A leaves */}
//  <path className="leaf" d="M529 80 Q499 50 547 45 Q511 100 555 80" />
//  <path className="leaf" d="M536 120 Q506 110 554 100 Q523 140 561 120" />
//  {/* T (top bar much wider) */}
//  <path className="trace" d="M540 40 H634 M587 40 V140" />
//  {/* T leaves */}
//  <path className="leaf" d="M587 60 Q558 30 605 50 Q587 60 609 40" />
//  <path className="leaf" d="M587 110 Q558 150 605 130 Q587 110 609 140" />
//  {/* O (was at 567, now stretched) */}
//  <path className="trace" d="M634 110 L641 60 L665 50 L689 60 L696 110 L665 130 L634 110 Z" />
//  {/* O leaves */}
//  <path className="leaf" d="M665 55 Q641 20 689 50 Q665 55 689 30" />
//  <path className="leaf" d="M665 120 Q641 160 689 130 Q665 120 689 150" />
//  {/* R (was at 646, now stretched) */}
//  <path className="trace" d="M721 140 V40 H757 L782 60 L757 90 L782 140" />
//  {/* R leaves */}
//  <path className="leaf" d="M757 70 Q721 30 770 65 Q757 70 782 40" />
//  <path className="leaf" d="M764 110 Q728 150 777 125 Q764 110 789 140" />
//  {/* Y (was at 721, now stretched) */}
//  <path className="trace" d="M805 40 L823 90 L841 40 M823 90 V140" />
//  {/* Y leaves */}
//  <path className="leaf" d="M823 110 Q794 80 841 100 Q823 110 845 90" />
//  <path className="leaf" d="M823 150 Q794 170 841 160 Q823 150 845 170" />