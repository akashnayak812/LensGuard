import { useCallback, useState } from 'react'
import { Upload, ImagePlus, X, Loader2 } from 'lucide-react'
import clsx from 'clsx'

/**
 * Drag-and-drop upload zone with live image preview and upload state.
 */
export default function DropZone({ onUpload, isLoading }) {
  const [preview, setPreview] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [file, setFile] = useState(null)

  const processFile = useCallback((f) => {
    if (!f || !f.type.startsWith('image/')) return
    setFile(f)
    const url = URL.createObjectURL(f)
    setPreview(url)
  }, [])

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault()
      setDragOver(false)
      const f = e.dataTransfer.files?.[0]
      if (f) processFile(f)
    },
    [processFile]
  )

  const handleFileInput = (e) => {
    const f = e.target.files?.[0]
    if (f) processFile(f)
    e.target.value = ''
  }

  const clearFile = () => {
    if (preview) URL.revokeObjectURL(preview)
    setPreview(null)
    setFile(null)
  }

  const handleSubmit = () => {
    if (file && !isLoading) onUpload(file)
  }

  return (
    <div className="space-y-4">
      {/* Drop area */}
      <label
        id="upload-zone"
        htmlFor="file-input"
        className={clsx(
          'relative flex min-h-[220px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed transition-all duration-200',
          dragOver
            ? 'border-brand-500 bg-brand-600/10 scale-[1.01]'
            : 'border-white/10 bg-surface-800/40 hover:border-brand-600/50 hover:bg-brand-600/5',
          isLoading && 'pointer-events-none opacity-60'
        )}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        aria-label="Upload zone. Click or drag and drop an image here."
      >
        <input
          id="file-input"
          type="file"
          accept="image/jpeg,image/png,image/webp,image/bmp,image/tiff"
          className="sr-only"
          onChange={handleFileInput}
          aria-label="Select an image file to upload"
        />

        {preview ? (
          /* Image preview */
          <div className="relative w-full h-full p-3">
            <img
              src={preview}
              alt="Preview of selected image"
              className="mx-auto max-h-48 rounded-xl object-contain"
            />
            <button
              type="button"
              onClick={(e) => { e.preventDefault(); clearFile() }}
              className="absolute right-4 top-4 flex h-7 w-7 items-center justify-center rounded-full bg-surface-900/90 text-slate-400 hover:text-slate-100 transition-colors"
              aria-label="Remove selected image"
            >
              <X size={14} />
            </button>
            <p className="mt-2 text-center text-xs text-slate-500 truncate px-4">
              {file?.name}
            </p>
          </div>
        ) : (
          /* Empty state */
          <div className="flex flex-col items-center gap-3 px-6 text-center">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-brand-600/15 text-brand-400">
              <ImagePlus size={26} />
            </div>
            <div>
              <p className="text-sm font-medium text-slate-300">
                Drag & drop an image here
              </p>
              <p className="text-xs text-slate-500 mt-0.5">
                or <span className="text-brand-400 font-medium">browse files</span>
              </p>
            </div>
            <p className="text-xs text-slate-600">
              JPEG · PNG · WebP · BMP · TIFF · up to 20MB
            </p>
          </div>
        )}
      </label>

      {/* Analyse button */}
      {file && (
        <button
          id="analyze-btn"
          type="button"
          onClick={handleSubmit}
          disabled={isLoading}
          className="btn-primary w-full justify-center text-base py-3"
        >
          {isLoading ? (
            <>
              <Loader2 size={18} className="animate-spin" />
              Analysing…
            </>
          ) : (
            <>
              <Upload size={18} />
              Analyse Image
            </>
          )}
        </button>
      )}
    </div>
  )
}
