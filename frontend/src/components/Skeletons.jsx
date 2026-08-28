/** Skeleton loading states */

export function ResultSkeleton() {
  return (
    <div className="animate-pulse space-y-6">
      {/* Gauge skeleton */}
      <div className="glass-card p-6 flex flex-col items-center gap-4">
        <div className="h-[180px] w-[180px] rounded-full skeleton" />
        <div className="h-5 w-28 skeleton rounded-full" />
      </div>

      {/* Issues skeleton */}
      <div className="space-y-3">
        {[1, 2].map((i) => (
          <div key={i} className="glass-card p-4 flex gap-4">
            <div className="h-10 w-10 skeleton rounded-xl shrink-0" />
            <div className="flex-1 space-y-2">
              <div className="h-4 w-32 skeleton rounded" />
              <div className="h-3 w-full skeleton rounded" />
              <div className="h-2 w-full skeleton rounded-full" />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export function HistorySkeleton() {
  return (
    <div className="animate-pulse space-y-3">
      {[1, 2, 3, 4].map((i) => (
        <div key={i} className="glass-card p-4 flex items-center gap-4">
          <div className="h-12 w-12 skeleton rounded-xl shrink-0" />
          <div className="flex-1 space-y-2">
            <div className="h-4 w-48 skeleton rounded" />
            <div className="h-3 w-32 skeleton rounded" />
          </div>
          <div className="h-8 w-16 skeleton rounded-lg" />
        </div>
      ))}
    </div>
  )
}

export function BatchSkeleton() {
  return (
    <div className="animate-pulse grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
      {[1, 2, 3, 4, 5, 6].map((i) => (
        <div key={i} className="glass-card p-3 space-y-2">
          <div className="h-32 skeleton rounded-xl" />
          <div className="h-4 w-full skeleton rounded" />
          <div className="h-3 w-16 skeleton rounded" />
        </div>
      ))}
    </div>
  )
}
