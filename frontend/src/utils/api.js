export async function getApiErrorMessage(response, fallbackMessage) {
  try {
    const data = await response.json()
    if (typeof data?.error === 'string') return data.error
    if (typeof data?.error?.message === 'string') return data.error.message
  } catch {
    // Ignore parsing errors and return fallback below.
  }
  return fallbackMessage
}
