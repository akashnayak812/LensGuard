import { gaugeColor } from '../utils/helpers'

/**
 * Animated SVG circular gauge showing a score 0–100.
 * Uses a stroke-dashoffset animation to draw the arc.
 */
export default function ScoreGauge({ score = 0, label = '', size = 180 }) {
  const radius = 70
  const circumference = 2 * Math.PI * radius
  // Only fill 270° of the circle (leave a gap at the bottom)
  const arcLength = circumference * 0.75
  const offset = arcLength - (score / 100) * arcLength
  const color = gaugeColor(score)
  const cx = size / 2
  const cy = size / 2

  return (
    <div
      className="flex flex-col items-center gap-2"
      role="img"
      aria-label={`Quality score: ${score} out of 100, label: ${label}`}
    >
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        className="drop-shadow-lg"
      >
        {/* Background arc */}
        <circle
          cx={cx}
          cy={cy}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.05)"
          strokeWidth={12}
          strokeDasharray={`${arcLength} ${circumference - arcLength}`}
          strokeDashoffset={0}
          strokeLinecap="round"
          transform={`rotate(135 ${cx} ${cy})`}
        />
        {/* Foreground arc — animated */}
        <circle
          cx={cx}
          cy={cy}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={12}
          strokeDasharray={`${arcLength} ${circumference - arcLength}`}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform={`rotate(135 ${cx} ${cy})`}
          className="transition-all duration-1000 ease-out"
          style={{ filter: `drop-shadow(0 0 8px ${color}88)` }}
        />
        {/* Score text */}
        <text
          x={cx}
          y={cy - 6}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize={36}
          fontWeight={700}
          fill={color}
          fontFamily="Inter, sans-serif"
        >
          {score}
        </text>
        <text
          x={cx}
          y={cy + 22}
          textAnchor="middle"
          fontSize={11}
          fill="rgba(148,163,184,0.8)"
          fontFamily="Inter, sans-serif"
          letterSpacing="0.08em"
        >
          / 100
        </text>
      </svg>

      {/* Label pill */}
      {label && (
        <span
          className={`rounded-full px-3 py-0.5 text-xs font-semibold tracking-wider border ${
            label === 'ACCEPTABLE'
              ? 'bg-success-500/15 text-success-400 border-success-500/30'
              : label === 'DEGRADED'
              ? 'bg-warning-500/15 text-warning-400 border-warning-500/30'
              : 'bg-danger-500/15 text-danger-400 border-danger-500/30'
          }`}
        >
          {label}
        </span>
      )}
    </div>
  )
}
