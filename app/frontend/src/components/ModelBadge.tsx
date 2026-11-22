import type { ModelInfo } from '../types'

export function ExperimentalBanner() {
  return (
    <p className="experimental">
      <span aria-hidden="true">⚠</span>
      Experimental model — results on real-world audio are unreliable.
    </p>
  )
}

export function ModelBadge({ model }: { model: ModelInfo }) {
  return (
    <details className="model-badge" title={model.description}>
      <summary>Model · {model.name} v{model.version}</summary>
      <p>{model.description}</p>
    </details>
  )
}
