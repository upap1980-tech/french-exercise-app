import React, { useState } from 'react'

const defaultTypes = ['fill_blank', 'multiple_choice', 'translate', 'role_play', 'conjugation', 'matching']

export default function TeacherAssistant({ apiBase = '' }) {
  const [level, setLevel] = useState('A1')
  const [type, setType] = useState('fill_blank')
  const [count, setCount] = useState(1)
  const [topicsText, setTopicsText] = useState('les couleurs')
  const [style, setStyle] = useState('creative')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState([])
  const [accepted, setAccepted] = useState([])
  const [error, setError] = useState(null)

  const buildUrl = (path) => (apiBase ? `${apiBase}${path}` : path)

  async function generate() {
    setLoading(true)
    setError(null)
    setResults([])
    const topics = topicsText.split(',').map(s => s.trim()).filter(Boolean)
    try {
      const res = await fetch(buildUrl('/api/assistant/create-exercise'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ level, type, count, topics, style })
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const body = await res.json()
      if (!body.ok) throw new Error('Respuesta del servidor inválida')
      setResults(body.generated || [])
    } catch (err) {
      console.error(err)
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  function downloadJSON(item) {
    const dataStr = JSON.stringify(item, null, 2)
    const blob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${item.type || 'exercise'}-${item.id || Date.now()}.json`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  function acceptItem(item) {
    setAccepted(prev => [item, ...prev])
  }

  function variantPromptAdjust(item) {
    // Small heuristic: propose a variant by adding contextual hint
    const variant = { ...item }
    variant.prompt = (variant.prompt || '') + ' (Variante con más contexto)'
    if (variant.id) variant.id = `${variant.id}-v1`
    return variant
  }

  function createVariant(item) {
    const v = variantPromptAdjust(item)
    setResults(prev => [v, ...prev])
  }

  return (
    <div style={{ padding: 12, maxWidth: 900 }}>
      <h3>Asistente para docentes</h3>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <label>
          Nivel
          <select value={level} onChange={e => setLevel(e.target.value)}>
            <option>A1</option>
            <option>A2</option>
            <option>B1</option>
            <option>B2</option>
          </select>
        </label>
        <label>
          Tipo
          <select value={type} onChange={e => setType(e.target.value)}>
            {defaultTypes.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
        <label>
          Cantidad
          <input type="number" min="1" value={count} onChange={e => setCount(Math.max(1, Number(e.target.value) || 1))} style={{ width: 72 }} />
        </label>
        <label style={{ flex: 1 }}>
          Temas (separados por coma)
          <input value={topicsText} onChange={e => setTopicsText(e.target.value)} style={{ width: '100%' }} />
        </label>
        <label>
          Estilo
          <select value={style} onChange={e => setStyle(e.target.value)}>
            <option value="creative">Creativo</option>
            <option value="concise">Conciso</option>
          </select>
        </label>
      </div>
      <div style={{ marginTop: 10 }}>
        <button onClick={generate} disabled={loading}>
          {loading ? 'Generando...' : 'Generar ejercicios'}
        </button>
        <button onClick={() => { setResults([]); setError(null); }} style={{ marginLeft: 8 }}>Limpiar</button>
      </div>
      {error && <div style={{ color: 'crimson', marginTop: 8 }}>Error: {error}</div>}

      <div style={{ marginTop: 18 }}>
        <h4>Resultados</h4>
        {results.length === 0 && <div>No hay ejercicios generados.</div>}
        {results.map(item => (
          <div key={item.id || item.prompt} style={{ border: '1px solid #ddd', padding: 8, marginBottom: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <div>
                <strong>{item.type}</strong> • Nivel: {item.level}
              </div>
              <div>
                <button onClick={() => { acceptItem(item); downloadJSON(item); }}>Aceptar y descargar JSON</button>
                <button onClick={() => acceptItem(item)} style={{ marginLeft: 6 }}>Aceptar</button>
                <button onClick={() => createVariant(item)} style={{ marginLeft: 6 }}>Crear variante</button>
              </div>
            </div>
            <div style={{ marginTop: 8 }}>{item.prompt}</div>
            {item.choices && (
              <ul>
                {item.choices.map((c, idx) => <li key={idx}>{c}</li>)}
              </ul>
            )}
            {item.solution && <div style={{ marginTop: 6, color: '#2b7' }}>Solución: {String(item.solution)}</div>}
          </div>
        ))}
      </div>

      <div style={{ marginTop: 18 }}>
        <h4>Aceptados ({accepted.length})</h4>
        {accepted.length === 0 && <div>No has aceptado variantes aún.</div>}
        {accepted.map(a => (
          <div key={a.id || a.prompt} style={{ borderBottom: '1px dashed #eee', padding: 6 }}>
            <div><strong>{a.type}</strong> — {a.prompt}</div>
            <div style={{ marginTop: 6 }}>
              <button onClick={() => downloadJSON(a)}>Descargar</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
