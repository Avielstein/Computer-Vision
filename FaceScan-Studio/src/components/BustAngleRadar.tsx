import type { BustAngleCoverage, BustAngleLabel } from '../types';

interface Props {
  coverage:     BustAngleCoverage;
  currentAngle: BustAngleLabel | null;
  pendingPct:   number; // 0-1
  size?:        number;
}

// 8 horizontal angles + up/down — layout on a circle
const RING_ANGLES: { label: BustAngleLabel; deg: number; symbol: string }[] = [
  { label: 'front',      deg: 270, symbol: '●' },
  { label: 'frontRight', deg: 315, symbol: '↗' },
  { label: 'right',      deg:   0, symbol: '→' },
  { label: 'backRight',  deg:  45, symbol: '↘' },
  { label: 'back',       deg:  90, symbol: '●' },
  { label: 'backLeft',   deg: 135, symbol: '↙' },
  { label: 'left',       deg: 180, symbol: '←' },
  { label: 'frontLeft',  deg: 225, symbol: '↖' },
];

const TOP_BOTTOM: { label: BustAngleLabel; symbol: string; yOffset: number }[] = [
  { label: 'up',   symbol: '↑', yOffset: -1 },
  { label: 'down', symbol: '↓', yOffset:  1 },
];

export function BustAngleRadar({ coverage, currentAngle, pendingPct, size = 120 }: Props) {
  const cx = size / 2;
  const cy = size / 2;
  const r  = size * 0.36;

  return (
    <svg width={size} height={size} style={{ display: 'block' }}>
      {/* Background circle */}
      <circle cx={cx} cy={cy} r={r + 8} fill="#0d0d15" stroke="#1a1a2a" strokeWidth={1} />
      <circle cx={cx} cy={cy} r={4} fill="#333" />

      {/* Ring segments */}
      {RING_ANGLES.map(({ label, deg, symbol }) => {
        const rad     = (deg * Math.PI) / 180;
        const x       = cx + r * Math.cos(rad);
        const y       = cy + r * Math.sin(rad);
        const conf    = coverage[label];
        const active  = currentAngle === label && !conf;
        const color   = conf ? '#4ade80' : active ? '#fbbf24' : '#333';
        const stroke  = conf ? '#4ade80' : active ? '#fbbf24' : '#222';
        const dotR    = 10;

        return (
          <g key={label}>
            {/* Pending arc for active angle */}
            {active && pendingPct > 0 && (
              <circle
                cx={x} cy={y} r={dotR + 3}
                fill="none"
                stroke="#fbbf2466"
                strokeWidth={2}
                strokeDasharray={`${pendingPct * 2 * Math.PI * (dotR + 3)} ${2 * Math.PI * (dotR + 3)}`}
                strokeLinecap="round"
                transform={`rotate(-90 ${x} ${y})`}
              />
            )}
            <circle cx={x} cy={y} r={dotR} fill={color + '22'} stroke={stroke} strokeWidth={1.5} />
            <text
              x={x} y={y + 1}
              textAnchor="middle"
              dominantBaseline="middle"
              fontSize={9}
              fontWeight={700}
              fill={color}
            >
              {symbol}
            </text>
          </g>
        );
      })}

      {/* Up / down indicators */}
      {TOP_BOTTOM.map(({ label, symbol, yOffset }) => {
        const x    = cx;
        const y    = cy + yOffset * (r * 0.45);
        const conf = coverage[label];
        const act  = currentAngle === label && !conf;
        const col  = conf ? '#4ade80' : act ? '#fbbf24' : '#333';
        return (
          <text
            key={label} x={x} y={y}
            textAnchor="middle" dominantBaseline="middle"
            fontSize={10} fontWeight={700} fill={col}
          >
            {symbol}
          </text>
        );
      })}

      {/* Centre label */}
      <text x={cx} y={cy} textAnchor="middle" dominantBaseline="middle" fontSize={7} fill="#444">
        HEAD
      </text>
    </svg>
  );
}
