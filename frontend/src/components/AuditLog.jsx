import { useState } from 'react'

function AuditLog() {
  const [logs, setLogs] = useState([])
  const [totalCount, setTotalCount] = useState(null)
  const [limit, setLimit] = useState(20)
  const [offset, setOffset] = useState(0)
  const [fromDate, setFromDate] = useState('')
  const [toDate, setToDate] = useState('')
  const [actionFilter, setActionFilter] = useState('')
  const [loading, setLoading] = useState(false)
  const apiBaseUrl = import.meta.env.VITE_API_URL || ''

  const buildUrl = (extraParams = {}) => {
    let url = `${apiBaseUrl}/api/compliance/audit-log`
    const params = new URLSearchParams()
    if (fromDate) params.append('from', fromDate)
    if (toDate) params.append('to', toDate)
    if (actionFilter) params.append('action', actionFilter)
    if (limit != null) params.append('limit', String(limit))
    if (offset != null) params.append('offset', String(offset))
    Object.entries(extraParams).forEach(([k,v]) => {
      if (v != null && v !== '') params.append(k, v)
    })
    if ([...params].length) url += `?${params.toString()}`
    return url
  }

  const loadLogs = async () => {
    setLoading(true)
    try {
      const response = await fetch(buildUrl())
      if (!response.ok) throw new Error(`Error ${response.status}`)
      const data = await response.json()
      if (data && data.entries !== undefined) {
        setLogs(data.entries)
        setTotalCount(data.total)
      } else {
        setLogs(Array.isArray(data) ? data : [])
        setTotalCount(null)
      }
    } catch (err) {
      setLogs([{ error: err.message || 'Error fetching logs' }])
      setTotalCount(null)
    } finally {
      setLoading(false)
    }
  }

  const exportCsv = () => {
    const url = buildUrl({ format: 'csv' })
    // trigger download
    const a = document.createElement('a')
    a.href = url
    a.download = 'audit_log.csv'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  return (
    <div className="audit-log">
      <h2>Registro de Auditoría</h2>
      <div className="form-group">
        <label>Desde:</label>
        <input
          type="datetime-local"
          value={fromDate}
          onChange={(e) => setFromDate(e.target.value)}
        />
      </div>
      <div className="form-group">
        <label>Hasta:</label>
        <input
          type="datetime-local"
          value={toDate}
          onChange={(e) => setToDate(e.target.value)}
        />
      </div>
      <div className="form-group">
        <label>Acción contiene:</label>
        <input
          type="text"
          value={actionFilter}
          onChange={(e) => setActionFilter(e.target.value)}
        />
      </div>
        <div className="generator-actions">
        <button className="btn-primary" onClick={() => { setOffset(0); loadLogs() }} disabled={loading}>
          {loading ? 'Cargando...' : 'Filtrar'}
        </button>
        <label style={{ marginLeft: '1em' }}>
          Tamaño:
          <select value={limit} onChange={(e) => setLimit(Number(e.target.value))} style={{ marginLeft: '0.5em' }}>
            {[10,20,50,100].map((n) => (<option key={n} value={n}>{n}</option>))}
          </select>
        </label>
        <button
          className="btn-secondary"
          style={{ marginLeft: '1em' }}
          onClick={exportCsv}
        >
          Exportar CSV
        </button>
        <button
          className="btn-secondary"
          style={{ marginLeft: '1em' }}
          onClick={() => {
            const url = buildUrl({ format: 'json' })
            const a = document.createElement('a')
            a.href = url
            a.download = 'audit_log.json'
            document.body.appendChild(a)
            a.click()
            document.body.removeChild(a)
          }}
        >
          Exportar JSON
        </button>
      </div>
      <div className="ia-tool-output" style={{ marginTop: '1em' }}>
        {Array.isArray(logs) && logs.length > 0 && !logs[0].error ? (
          <>
            <table className="audit-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Action</th>
                  <th>Detail</th>
                </tr>
              </thead>
              <tbody>
                {logs.map((row, idx) => (
                  <tr key={idx}>
                    <td>{row.timestamp}</td>
                    <td>{row.action}</td>
                    <td>{row.detail ? JSON.stringify(row.detail) : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {totalCount !== null && (
              <div className="pagination-controls" style={{ marginTop: '0.5em' }}>
                <button
                  className="btn-secondary"
                  onClick={() => {
                    if (offset - limit >= 0) {
                      setOffset(offset - limit)
                      loadLogs()
                    }
                  }}
                  disabled={offset === 0 || loading}
                >
                  Anterior
                </button>
                <span style={{ margin: '0 1em' }}>
                  Página {Math.floor(offset / limit) + 1} de {Math.ceil(totalCount / limit)}
                </span>
                <button
                  className="btn-secondary"
                  onClick={() => {
                    if (offset + limit < totalCount) {
                      setOffset(offset + limit)
                      loadLogs()
                    }
                  }}
                  disabled={offset + limit >= totalCount || loading}
                >
                  Siguiente
                </button>
              </div>
            )}
          </>
        ) : (
          <pre style={{ whiteSpace: 'pre-wrap' }}>
            {JSON.stringify(logs, null, 2)}
          </pre>
        )}
      </div>
    </div>
  )
}

export default AuditLog
