import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import UploadPage from '../../src/pages/UploadPage'

// Mock the API module
vi.mock('../../src/utils/api', () => ({
  analyzeImage: vi.fn(),
}))

// Mock useNavigate
const mockNavigate = vi.fn()
vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal()
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  }
})

import { analyzeImage } from '../../src/utils/api'

const renderUploadPage = () =>
  render(
    <MemoryRouter>
      <UploadPage />
    </MemoryRouter>
  )

describe('UploadPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the heading', () => {
    renderUploadPage()
    expect(screen.getByRole('heading', { level: 1 })).toBeInTheDocument()
    expect(screen.getByText(/Image Quality Detection/i)).toBeInTheDocument()
  })

  it('renders the upload zone', () => {
    renderUploadPage()
    const zone = screen.getByLabelText(/Upload zone/i)
    expect(zone).toBeInTheDocument()
  })

  it('renders feature tiles in idle state', () => {
    renderUploadPage()
    expect(screen.getByText('Quality Score')).toBeInTheDocument()
    expect(screen.getByText('Defect Analysis')).toBeInTheDocument()
    expect(screen.getByText('Batch Mode')).toBeInTheDocument()
  })

  it('shows analyse button after file selection', async () => {
    renderUploadPage()
    const input = screen.getByLabelText(/Select an image file/i)

    const mockFile = new File(['(⌐□_□)'], 'test.jpg', { type: 'image/jpeg' })
    // Mock URL.createObjectURL
    global.URL.createObjectURL = vi.fn().mockReturnValue('blob:test')

    fireEvent.change(input, { target: { files: [mockFile] } })

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Analyse Image/i })).toBeInTheDocument()
    })
  })

  it('shows error state when API fails', async () => {
    analyzeImage.mockRejectedValueOnce(new Error('File too large'))

    renderUploadPage()
    const input = screen.getByLabelText(/Select an image file/i)
    const mockFile = new File(['data'], 'bad.jpg', { type: 'image/jpeg' })
    global.URL.createObjectURL = vi.fn().mockReturnValue('blob:test')

    fireEvent.change(input, { target: { files: [mockFile] } })

    await waitFor(() => {
      const analyseBtn = screen.getByRole('button', { name: /Analyse Image/i })
      fireEvent.click(analyseBtn)
    })

    await waitFor(() => {
      expect(screen.getByText(/Analysis failed/i)).toBeInTheDocument()
    })
  })

  it('navigates to result page on successful analysis', async () => {
    analyzeImage.mockResolvedValueOnce({ id: 42, quality_score: 85, quality_label: 'ACCEPTABLE', issues: [], filename: 'test.jpg', image_stats: {}, model_version: 'v1', created_at: new Date().toISOString() })

    renderUploadPage()
    const input = screen.getByLabelText(/Select an image file/i)
    const mockFile = new File(['data'], 'photo.jpg', { type: 'image/jpeg' })
    global.URL.createObjectURL = vi.fn().mockReturnValue('blob:test')

    fireEvent.change(input, { target: { files: [mockFile] } })

    await waitFor(() => {
      const analyseBtn = screen.getByRole('button', { name: /Analyse Image/i })
      fireEvent.click(analyseBtn)
    })

    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/result/42', expect.any(Object))
    })
  })
})
