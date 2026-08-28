import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import ScoreGauge from '../../src/components/ScoreGauge'
import IssueCard from '../../src/components/IssueCard'

// ── ScoreGauge tests ─────────────────────────────────────────────────────────

describe('ScoreGauge', () => {
  it('renders the score number', () => {
    render(<ScoreGauge score={82} label="ACCEPTABLE" />)
    expect(screen.getByText('82')).toBeInTheDocument()
  })

  it('renders the label', () => {
    render(<ScoreGauge score={82} label="ACCEPTABLE" />)
    expect(screen.getByText('ACCEPTABLE')).toBeInTheDocument()
  })

  it('renders DEGRADED label correctly', () => {
    render(<ScoreGauge score={55} label="DEGRADED" />)
    expect(screen.getByText('DEGRADED')).toBeInTheDocument()
  })

  it('renders DEFECTIVE label correctly', () => {
    render(<ScoreGauge score={20} label="DEFECTIVE" />)
    expect(screen.getByText('DEFECTIVE')).toBeInTheDocument()
  })

  it('has accessible role and aria-label', () => {
    render(<ScoreGauge score={75} label="ACCEPTABLE" />)
    expect(screen.getByRole('img')).toHaveAttribute('aria-label', expect.stringContaining('75'))
  })

  it('renders without label', () => {
    render(<ScoreGauge score={50} />)
    expect(screen.getByText('50')).toBeInTheDocument()
  })
})

// ── IssueCard tests ──────────────────────────────────────────────────────────

describe('IssueCard', () => {
  const mockIssue = {
    type: 'blur',
    severity: 'medium',
    confidence: 0.75,
  }

  it('renders the issue type', () => {
    render(<IssueCard issue={mockIssue} />)
    expect(screen.getByText(/blur/i)).toBeInTheDocument()
  })

  it('renders the severity badge', () => {
    render(<IssueCard issue={mockIssue} />)
    expect(screen.getByText('medium')).toBeInTheDocument()
  })

  it('renders confidence as percentage', () => {
    render(<IssueCard issue={mockIssue} />)
    expect(screen.getByText('75%')).toBeInTheDocument()
  })

  it('has a progressbar with correct aria attributes', () => {
    render(<IssueCard issue={mockIssue} />)
    const bar = screen.getByRole('progressbar')
    expect(bar).toHaveAttribute('aria-valuenow', '75')
    expect(bar).toHaveAttribute('aria-valuemin', '0')
    expect(bar).toHaveAttribute('aria-valuemax', '100')
  })

  it('renders noise issue correctly', () => {
    render(<IssueCard issue={{ type: 'noise', severity: 'high', confidence: 0.9 }} />)
    expect(screen.getByText(/noise/i)).toBeInTheDocument()
    expect(screen.getByText('high')).toBeInTheDocument()
    expect(screen.getByText('90%')).toBeInTheDocument()
  })

  it('renders defect issue correctly', () => {
    render(<IssueCard issue={{ type: 'defect', severity: 'low', confidence: 0.3 }} />)
    const elements = screen.getAllByText(/defect/i)
    expect(elements.length).toBeGreaterThan(0)
  })
})
