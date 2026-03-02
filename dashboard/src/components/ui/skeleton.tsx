import { cn } from '@/lib/utils'

/**
 * Skeleton component for loading placeholder animations.
 * Displays an animated pulse to indicate content is loading.
 */
function Skeleton({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn('animate-pulse rounded-md bg-primary/10', className)}
      {...props}
    />
  )
}

export { Skeleton }
