import { useEffect, useState } from 'react'

function Dashboard({ onBackendStatusChange }) {
  const REFRESH_OPTIONS = [5, 15, 30, 60]
  const REFRESH_STORAGE_KEY = 'dashboard_refresh_seconds'
  const DASHBOARD_SNAPSHOT_KEY = 'dashboard_last_snapshot'

  const getInitialRefreshSeconds = () => {
    const savedValue = Number(localStorage.getItem(REFRESH_STORAGE_KEY))
    return REFRESH_OPTIONS.includes(savedValue) ? savedValue : 15
  }

  const [refreshSeconds, setRefreshSeconds] = useState(getInitialRefreshSeconds)
  const [stats, setStats] = useState({
    totalExercises: 0,
    totalDocuments: 0,
    totalExams: 0,
    aiModels: 0,
    lastUpdated: '-'
  })
  const [models, setModels] = useState([])
  const [aiHealth, setAiHealth] = useState({})
  const [loading, setLoading] = useState(false)
  const [backendError, setBackendError] = useState('')
  const [isBackendOnline, setIsBackendOnline] = useState(true)
  const [compliance, setCompliance] = useState(null)
  const [analytics, setAnalytics] = useState(null)
  const [keyHealth, setKeyHealth] = useState(null)
  const [aiTools, setAiTools] = useState([])
  const [toolStates, setToolStates] = useState({})
  const [toolLoading, setToolLoading] = useState({})
  const [repairLoading, setRepairLoading] = useState(false)
  const [repairResult, setRepairResult] = useState(null)
  const [backendHealth, setBackendHealth] = useState(null)
  const [opsMetrics, setOpsMetrics] = useState(null)
  const [workspaceHealth, setWorkspaceHealth] = useState(null)
  const [enterpriseFeatures, setEnterpriseFeatures] = useState([])
  const [backupLoading, setBackupLoading] = useState(false)
  const [lastBackupResult, setLastBackupResult] = useState(null)
  const [restoreLoading, setRestoreLoading] = useState(false)
  const [lastRestoreResult, setLastRestoreResult] = useState(null)
  const [anonymizeText, setAnonymizeText] = useState('Nombre: Juan Pérez, email: juan@test.com')
  const [anonymizeResult, setAnonymizeResult] = useState(null)
  const [auditLogPreview, setAuditLogPreview] = useState([])
  const [refineProvider, setRefineProvider] = useState('openai')
  const [refineText, setRefineText] = useState('Ficha de francés sobre les vêtements para 6º.')
  const [refineResult, setRefineResult] = useState(null)
  const apiBaseUrl = import.meta.env.VITE_API_URL || ''

  const formatDetail = (detail) => {
    if (!detail) return ''
    if (detail === 'reachable') return 'Conectado correctamente'
    if (detail === 'missing_api_key') return 'Falta API key'
    if (detail.startsWith('missing_api_key:')) return `Falta variable ${detail.replace('missing_api_key:', '')}`
    if (detail === 'rate_limit') return 'Límite de peticiones alcanzado (429)'
    if (detail === 'insufficient_credits') return 'Créditos insuficientes en proveedor'
    if (detail === 'installed') return 'Instalado localmente'
    if (detail === 'not_installed') return 'No instalado (opcional)'
    if (detail === 'pending_manual_runtime') return 'Pendiente runtime manual'
    if (detail.startsWith('missing_runtime:')) return `Falta runtime ${detail.replace('missing_runtime:', '')}`
    if (detail === 'gemini_model_not_found') return 'Gemini sin modelo disponible en tu cuenta/proyecto'
    if (detail === 'gemini_auth_error') return 'Gemini API key inválida o sin permisos'
    if (detail === 'unsupported_provider') return 'Proveedor no soportado'
    if (detail.startsWith('http_')) return `Error HTTP ${detail.replace('http_', '')}`
    if (detail.startsWith('network_error:')) return 'Error de red o timeout'
    return detail
  }

  const getProviderGuidance = (provider, detail) => {
    const base = {
      openai: {
        docs: 'https://platform.openai.com/docs/overview',
        keys: 'https://platform.openai.com/api-keys',
        billing: 'https://platform.openai.com/settings/organization/billing/overview'
      },
      gemini: {
        docs: 'https://ai.google.dev/gemini-api/docs',
        keys: 'https://aistudio.google.com/app/apikey',
        billing: 'https://console.cloud.google.com/billing'
      },
      deepseek: {
        docs: 'https://platform.deepseek.com/docs',
        keys: 'https://platform.deepseek.com/api_keys',
        billing: 'https://platform.deepseek.com/'
      },
      perplexity: {
        docs: 'https://docs.perplexity.ai/',
        keys: 'https://www.perplexity.ai/settings/api',
        billing: 'https://www.perplexity.ai/settings/api'
      },
      llama3: {
        docs: 'https://ollama.com/library/llama3',
        keys: 'https://ollama.com/',
        billing: 'https://ollama.com/'
      },
      mistral: {
        docs: 'https://ollama.com/library/mistral',
        keys: 'https://ollama.com/',
        billing: 'https://ollama.com/'
      }
    }

    const links = base[provider] || base.openai
    const checklist = []

    if (provider === 'llama3' || provider === 'mistral') {
      checklist.push('Verifica que Ollama esté activo: `ollama serve`')
      checklist.push(`Confirma que el modelo exista: \`ollama pull ${provider}\``)
      checklist.push('Comprueba conectividad local en `http://localhost:11434`')
      return { checklist, links }
    }

    checklist.push('Revisar que la API key esté cargada en `backend/.env`')
    checklist.push('Reiniciar backend para recargar variables de entorno')

    if (detail === 'missing_api_key') {
      checklist.push('Crear una nueva API key y actualizar `.env`')
    } else if (detail === 'rate_limit') {
      checklist.push('Esperar unos minutos o aumentar límites/cuota del proveedor')
      checklist.push('Reducir frecuencia de llamadas en pruebas manuales')
    } else if (detail === 'insufficient_credits') {
      checklist.push('Recargar saldo o activar método de pago')
    } else if (detail === 'http_404') {
      checklist.push('Validar endpoint/modelo habilitado para tu proyecto')
      checklist.push('Revisar restricciones de API key (referrer/IP/proyecto)')
    } else if (detail && detail.startsWith('http_')) {
      checklist.push('Revisar permisos de cuenta y estado del servicio')
    } else if (detail && detail.startsWith('network_error:')) {
      checklist.push('Comprobar firewall/VPN/proxy y salida HTTPS')
    } else if (detail === 'reachable') {
      checklist.push('Proveedor operativo, no requiere acción')
    }

    return { checklist, links }
  }

  const copyChecklist = async (provider, state, checklist) => {
    const lines = [
      `Proveedor: ${provider}`,
      `Estado: ${state.ok ? 'OPERATIVO' : 'REVISAR'}`,
      `Detalle: ${formatDetail(state.detail)}`,
      '',
      'Checklist:'
    ]
    checklist.forEach((step, idx) => lines.push(`${idx + 1}. ${step}`))
    const content = lines.join('\n')

    try {
      await navigator.clipboard.writeText(content)
      alert(`Checklist copiado para ${provider}`)
    } catch (error) {
      const textarea = document.createElement('textarea')
      textarea.value = content
      document.body.appendChild(textarea)
      textarea.select()
      document.execCommand('copy')
      document.body.removeChild(textarea)
      alert(`Checklist copiado para ${provider}`)
    }
  }

  const copyMissingVars = async () => {
    const missing = keyHealth?.missing_vars || []
    if (missing.length === 0) {
      alert('No hay variables faltantes')
      return
    }
    const content = [
      '# Variables API faltantes',
      ...missing.map((item) => `${item}=`)
    ].join('\n')
    try {
      await navigator.clipboard.writeText(content)
      alert('Variables faltantes copiadas')
    } catch {
      const textarea = document.createElement('textarea')
      textarea.value = content
      document.body.appendChild(textarea)
      textarea.select()
      document.execCommand('copy')
      document.body.removeChild(textarea)
      alert('Variables faltantes copiadas')
    }
  }

  const loadDashboard = async (forceAiTest = false) => {
    setLoading(true)
    setBackendError('')
    try {
      const [exerciseRes, docRes, examRes, modelRes, aiTestRes, complianceRes, analyticsRes, toolsRes, keyHealthRes, healthRes, opsRes, workspaceHealthRes, featureRes] = await Promise.all([
        fetch(`${apiBaseUrl}/api/exercises`),
        fetch(`${apiBaseUrl}/api/documents`),
        fetch(`${apiBaseUrl}/api/exams`),
        fetch(`${apiBaseUrl}/api/ai/models`),
        fetch(`${apiBaseUrl}/api/ai/test${forceAiTest ? '?force=1' : ''}`),
        fetch(`${apiBaseUrl}/api/compliance/status`),
        fetch(`${apiBaseUrl}/api/analytics/learning`),
        fetch(`${apiBaseUrl}/api/ai/tools`),
        fetch(`${apiBaseUrl}/api/health/ai-keys`),
        fetch(`${apiBaseUrl}/api/health`),
        fetch(`${apiBaseUrl}/api/ops/metrics`),
        fetch(`${apiBaseUrl}/api/google/workspace/health`),
        fetch(`${apiBaseUrl}/api/enterprise/features`)
      ])

      const failed = [exerciseRes, docRes, examRes, modelRes].some((res) => !res.ok)
      if (failed) {
        throw new Error('No se pudo obtener datos del backend')
      }

      const exercises = exerciseRes.ok ? await exerciseRes.json() : []
      const documents = docRes.ok ? await docRes.json() : []
      const exams = examRes.ok ? await examRes.json() : []
      const modelData = modelRes.ok ? await modelRes.json() : { local: [], cloud: [] }
      const aiTestData = aiTestRes.ok ? await aiTestRes.json() : { results: {} }
      const complianceData = complianceRes.ok ? await complianceRes.json() : null
      const analyticsData = analyticsRes.ok ? await analyticsRes.json() : null
      const toolsData = toolsRes.ok ? await toolsRes.json() : { items: [] }
      const keyHealthData = keyHealthRes.ok ? await keyHealthRes.json() : null
      const healthData = healthRes.ok ? await healthRes.json() : null
      const opsData = opsRes.ok ? await opsRes.json() : null
      const wsHealthData = workspaceHealthRes.ok ? await workspaceHealthRes.json() : null
      const featureData = featureRes.ok ? await featureRes.json() : null
      const availableModels = [...(modelData.local || []), ...(modelData.cloud || [])]

      setModels(availableModels)
      setAiHealth(aiTestData.results || {})
      setCompliance(complianceData)
      setAnalytics(analyticsData)
      setKeyHealth(keyHealthData)
      setBackendHealth(healthData)
      setOpsMetrics(opsData)
      setWorkspaceHealth(wsHealthData)
      setEnterpriseFeatures(Array.isArray(featureData?.features) ? featureData.features : [])
      setAiTools(Array.isArray(toolsData.items) ? toolsData.items : [])
      const nextStats = {
        totalExercises: exercises.length,
        totalDocuments: documents.length,
        totalExams: exams.length,
        aiModels: availableModels.length,
        lastUpdated: new Date().toLocaleTimeString('es-ES')
      }
      setStats(nextStats)
      localStorage.setItem(
        DASHBOARD_SNAPSHOT_KEY,
        JSON.stringify({
          stats: nextStats,
          models: availableModels,
          aiHealth: aiTestData.results || {}
        })
      )
      setIsBackendOnline(true)
      if (onBackendStatusChange) onBackendStatusChange(true)
    } catch (error) {
      console.error('Error loading dashboard:', error)
      const snapshotRaw = localStorage.getItem(DASHBOARD_SNAPSHOT_KEY)
      if (snapshotRaw) {
        try {
          const snapshot = JSON.parse(snapshotRaw)
          if (snapshot?.stats) setStats(snapshot.stats)
          if (Array.isArray(snapshot?.models)) setModels(snapshot.models)
          if (snapshot?.aiHealth) setAiHealth(snapshot.aiHealth)
          setCompliance(null)
          setAnalytics(null)
          setKeyHealth(null)
          setBackendError('Backend desconectado. Mostrando último estado guardado.')
        } catch {
          setBackendError('Backend desconectado o con error. Verifica que esté iniciado con el entorno virtual.')
        }
      } else {
        setBackendError('Backend desconectado o con error. Verifica que esté iniciado con el entorno virtual.')
      }
      setIsBackendOnline(false)
      if (onBackendStatusChange) onBackendStatusChange(false)
    } finally {
      setLoading(false)
    }
  }

  const callToolEndpoint = async (toolId, action) => {
    const isGet = action === 'diagnostic'
    const response = await fetch(`${apiBaseUrl}/api/ai/tools/${toolId}/${action}`, {
      method: isGet ? 'GET' : 'POST',
      headers: isGet ? {} : { 'Content-Type': 'application/json' },
      body: isGet ? undefined : JSON.stringify({ topic: 'les vêtements', level: 'A2' })
    })
    if (!response.ok) {
      throw new Error(`Error ${response.status}`)
    }
    return response.json()
  }

  const runToolAction = async (toolId, action) => {
    setToolLoading((prev) => ({ ...prev, [`${toolId}_${action}`]: true }))
    try {
      const data = await callToolEndpoint(toolId, action)
      setToolStates((prev) => ({
        ...prev,
        [toolId]: {
          ...(prev[toolId] || {}),
          [action]: data
        }
      }))
      if (action === 'test') {
        await loadDashboard(true)
      }
    } catch (error) {
      setToolStates((prev) => ({
        ...prev,
        [toolId]: {
          ...(prev[toolId] || {}),
          [action]: { error: error.message || 'Error' }
        }
      }))
    } finally {
      setToolLoading((prev) => ({ ...prev, [`${toolId}_${action}`]: false }))
    }
  }

  const runGeminiDeepseekRepair = async () => {
    setRepairLoading(true)
    try {
      const response = await fetch(`${apiBaseUrl}/api/ai/repair/gemini-deepseek`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      if (!response.ok) throw new Error(`Error ${response.status}`)
      const data = await response.json()
      setRepairResult(data)
    } catch (error) {
      setRepairResult({ error: error.message || 'Error' })
    } finally {
      setRepairLoading(false)
    }
  }

  const runBackupNow = async () => {
    setBackupLoading(true)
    try {
      const response = await fetch(`${apiBaseUrl}/api/backups/export`, { method: 'POST' })
      const data = await response.json()
      if (!response.ok) throw new Error(data?.error?.message || data?.error || `Error ${response.status}`)
      setLastBackupResult(data)
    } catch (error) {
      setLastBackupResult({ error: error.message || 'Error' })
    } finally {
      setBackupLoading(false)
    }
  }

  const runRestoreLatest = async () => {
    setRestoreLoading(true)
    try {
      const response = await fetch(`${apiBaseUrl}/api/backups/restore-latest`, { method: 'POST' })
      const data = await response.json()
      if (!response.ok) throw new Error(data?.error?.message || data?.error || `Error ${response.status}`)
      setLastRestoreResult(data)
      await loadDashboard(true)
    } catch (error) {
      setLastRestoreResult({ error: error.message || 'Error' })
    } finally {
      setRestoreLoading(false)
    }
  }

  const runAnonymizePreview = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/compliance/anonymize-preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: anonymizeText })
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data?.error?.message || data?.error || `Error ${response.status}`)
      setAnonymizeResult(data)
    } catch (error) {
      setAnonymizeResult({ error: error.message || 'Error' })
    }
  }

  const loadAuditLog = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/compliance/audit-log`)
      const data = await response.json()
      if (!response.ok) throw new Error(data?.error?.message || data?.error || `Error ${response.status}`)
      setAuditLogPreview(Array.isArray(data) ? data.slice(0, 20) : [])
    } catch (error) {
      setAuditLogPreview([{ action: 'error', detail: { message: error.message || 'Error' }, timestamp: new Date().toISOString() }])
    }
  }

  const runProviderRefine = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/ai/providers/${refineProvider}/refine`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: refineText })
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data?.error?.message || data?.error || `Error ${response.status}`)
      setRefineResult(data)
    } catch (error) {
      setRefineResult({ error: error.message || 'Error' })
    }
  }

  useEffect(() => {
    localStorage.setItem(REFRESH_STORAGE_KEY, String(refreshSeconds))
  }, [refreshSeconds])

  useEffect(() => {
    loadDashboard()
    if (!isBackendOnline) return
    const intervalId = setInterval(loadDashboard, refreshSeconds * 1000)
    return () => clearInterval(intervalId)
  }, [refreshSeconds, isBackendOnline])

  return (
    <div className="dashboard">
      <h2>Panel de Control</h2>
      <div className="dashboard-toolbar">
        <button className="btn-primary" onClick={() => loadDashboard(true)} disabled={loading}>
          {loading ? 'Actualizando...' : 'Actualizar ahora'}
        </button>
        <div className="refresh-control">
          <label htmlFor="refreshSelect">Frecuencia:</label>
          <select
            id="refreshSelect"
            value={refreshSeconds}
            onChange={(e) => setRefreshSeconds(Number(e.target.value))}
          >
            {REFRESH_OPTIONS.map((seconds) => (
              <option key={seconds} value={seconds}>
                {seconds}s
              </option>
            ))}
          </select>
        </div>
        <span className="dashboard-refresh-info">
          {isBackendOnline ? `Auto-refresh cada ${refreshSeconds}s` : 'Auto-refresh pausado (backend offline)'}
        </span>
      </div>
      {backendError && (
        <div className="backend-error-banner">
          {backendError}
        </div>
      )}
      
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-value">{stats.totalExercises}</div>
          <div className="stat-label">Ejercicios Generados</div>
        </div>
        
        <div className="stat-card">
          <div className="stat-value">{stats.totalDocuments}</div>
          <div className="stat-label">Documentos Subidos</div>
        </div>
        
        <div className="stat-card">
          <div className="stat-value">{stats.aiModels}</div>
          <div className="stat-label">Modelos IA Disponibles</div>
        </div>

        <div className="stat-card">
          <div className="stat-value">{stats.totalExams}</div>
          <div className="stat-label">Exámenes Creados</div>
        </div>
        
        <div className="stat-card">
          <div className="stat-value">{stats.lastUpdated}</div>
          <div className="stat-label">Última Actualización</div>
        </div>
      </div>

      <div className="dashboard-content">
        <div className="section">
          <h3>Modelos de IA Disponibles</h3>
          <ul className="model-list">
            {models.length === 0 && <li>No hay modelos IA disponibles</li>}
            {models.map((model) => (
              <li key={model}>{model}</li>
            ))}
          </ul>
        </div>

        <div className="section">
          <h3>Key Health</h3>
          <div className="wizard-card key-health-card">
            <div className="wizard-header">
              <strong>Estado de API Keys</strong>
              <span className={`health-badge ${keyHealth?.summary?.status === 'green' ? 'health-green' : 'health-red'}`}>
                {keyHealth?.summary?.status === 'green' ? 'OK' : 'FALTAN KEYS'}
              </span>
            </div>
            <div className="wizard-detail">
              {keyHealth
                ? `${keyHealth.summary.providers_ok}/${keyHealth.summary.providers_total} proveedores configurados`
                : 'Sin datos de key health'}
            </div>
            {keyHealth?.missing_vars?.length > 0 ? (
              <>
                <ul className="wizard-checklist">
                  {keyHealth.missing_vars.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
                <button className="btn-primary btn-copy-checklist" onClick={copyMissingVars}>
                  Copiar variables faltantes
                </button>
              </>
            ) : (
              <div className="provider-health-detail detail-ok">
                Todas las variables requeridas están configuradas.
              </div>
            )}
          </div>
        </div>

        <div className="section">
          <h3>Estado de Conectividad IA</h3>
          <ul className="model-list">
            {Object.keys(aiHealth).length === 0 && <li>Sin datos de conectividad</li>}
            {Object.entries(aiHealth).map(([provider, state]) => (
              <li key={provider} className="provider-health-row">
                <div className="provider-health-main">
                  <div className="provider-health-top">
                    <span>{provider}</span>
                    <span className={`health-badge ${state.ok ? 'health-green' : 'health-red'}`}>
                      {state.ok ? 'VERDE' : 'ROJO'}
                    </span>
                  </div>
                  {state.detail && (
                    <div className={`provider-health-detail ${state.ok ? 'detail-ok' : 'detail-error'}`}>
                      {formatDetail(state.detail)}
                    </div>
                  )}
                </div>
              </li>
            ))}
          </ul>
        </div>

        <div className="section">
          <h3>Wizard de Diagnóstico IA</h3>
          <div className="wizard-list">
            {Object.entries(aiHealth).map(([provider, state]) => {
              const guidance = getProviderGuidance(provider, state.detail)
              return (
                <div key={`wizard-${provider}`} className="wizard-card">
                  <div className="wizard-header">
                    <strong>{provider}</strong>
                    <span className={`health-badge ${state.ok ? 'health-green' : 'health-red'}`}>
                      {state.ok ? 'OPERATIVO' : 'REVISAR'}
                    </span>
                  </div>
                  <div className="wizard-detail">{formatDetail(state.detail)}</div>
                  <ul className="wizard-checklist">
                    {guidance.checklist.map((step, idx) => (
                      <li key={`${provider}-step-${idx}`}>{step}</li>
                    ))}
                  </ul>
                  <button
                    className="btn-primary btn-copy-checklist"
                    onClick={() => copyChecklist(provider, state, guidance.checklist)}
                  >
                    Copiar checklist
                  </button>
                  <div className="wizard-links">
                    <a href={guidance.links.docs} target="_blank" rel="noreferrer">Docs</a>
                    <a href={guidance.links.keys} target="_blank" rel="noreferrer">API Keys</a>
                    <a href={guidance.links.billing} target="_blank" rel="noreferrer">Billing</a>
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        <div className="section ia-studio-section">
          <h3>IA Studio</h3>
          <div className="ia-repair-card">
            <div className="ia-tool-header">
              <strong>Reparación Asistida Gemini/DeepSeek</strong>
            </div>
            <div className="wizard-detail">
              Detecta automáticamente `GEMINI_MODELS` óptimo y proveedor fallback recomendado por coste/rendimiento.
            </div>
            <button className="btn-primary" onClick={runGeminiDeepseekRepair} disabled={repairLoading}>
              {repairLoading ? 'Analizando...' : 'Reparar Gemini/DeepSeek'}
            </button>
            {repairResult?.error && (
              <div className="provider-health-detail detail-error">
                {repairResult.error}
              </div>
            )}
            {repairResult?.gemini && (
              <div className="ia-tool-output">
                <strong>GEMINI_MODELS recomendado:</strong>
                <pre>{repairResult.gemini.recommended_env_value}</pre>
                <strong>Fallback recomendado:</strong>
                <pre>{repairResult.deepseek_fallback?.provider}</pre>
                {Array.isArray(repairResult.checklist) && (
                  <ul className="wizard-checklist">
                    {repairResult.checklist.map((step, idx) => (
                      <li key={`repair-step-${idx}`}>{step}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </div>
          <div className="ia-studio-grid">
            {aiTools.length === 0 && <p>No hay herramientas IA configuradas todavía.</p>}
            {aiTools.map((tool) => {
              const state = toolStates[tool.id] || {}
              const testResult = state.test
              const sampleResult = state.sample
              const diagnostic = state.diagnostic
              return (
                <div className="ia-tool-card" key={tool.id}>
                  <div className="ia-tool-header">
                    <strong>{tool.name}</strong>
                    <span className={`health-badge ${tool.configured ? 'health-green' : 'health-red'}`}>
                      {tool.configured ? 'CONFIGURADO' : 'SIN KEY'}
                    </span>
                  </div>
                  <div className="wizard-detail">{tool.category}</div>
                  <div className="ia-tool-actions">
                    <button
                      className="btn-secondary"
                      onClick={() => runToolAction(tool.id, 'test')}
                      disabled={!!toolLoading[`${tool.id}_test`]}
                    >
                      {toolLoading[`${tool.id}_test`] ? 'Probando...' : 'Test'}
                    </button>
                    <button
                      className="btn-secondary"
                      onClick={() => runToolAction(tool.id, 'sample')}
                      disabled={!!toolLoading[`${tool.id}_sample`]}
                    >
                      {toolLoading[`${tool.id}_sample`] ? 'Generando...' : 'Generate sample'}
                    </button>
                    <button
                      className="btn-secondary"
                      onClick={() => runToolAction(tool.id, 'diagnostic')}
                      disabled={!!toolLoading[`${tool.id}_diagnostic`]}
                    >
                      {toolLoading[`${tool.id}_diagnostic`] ? 'Cargando...' : 'Diagnóstico'}
                    </button>
                  </div>
                  {testResult && (
                    <div className={`provider-health-detail ${testResult.ok ? 'detail-ok' : 'detail-error'}`}>
                      Test: {testResult.ok ? 'OK' : 'FAIL'} · {formatDetail(testResult.detail)}
                    </div>
                  )}
                  {sampleResult && (
                    <div className="ia-tool-output">
                      <strong>Sample:</strong>
                      <pre>{JSON.stringify(sampleResult, null, 2)}</pre>
                    </div>
                  )}
                  {diagnostic?.checklist && (
                    <ul className="wizard-checklist">
                      {diagnostic.checklist.map((step, idx) => (
                        <li key={`${tool.id}-diag-${idx}`}>{step}</li>
                      ))}
                    </ul>
                  )}
                  {(sampleResult?.configure_url || tool.url) && (
                    <div className="wizard-links">
                      {tool.url && <a href={tool.url} target="_blank" rel="noreferrer">Sitio oficial</a>}
                      {sampleResult?.configure_url && (
                        <a href={sampleResult.configure_url} target="_blank" rel="noreferrer">Configurar</a>
                      )}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>

        <div className="section">
          <h3>Características Principales</h3>
          <ul className="features-list">
            <li>✅ Generación de ejercicios con IA</li>
            <li>✅ Análisis de documentos escaneados</li>
            <li>✅ Generación de exámenes de evaluación</li>
            <li>✅ Interfaz responsive para iPhone</li>
            <li>✅ Soporte PWA para instalación en dispositivos</li>
          </ul>
        </div>

        <div className="section">
          <h3>Cumplimiento IA</h3>
          <ul className="features-list">
            <li>Modo: {compliance?.mode || 'sin datos'}</li>
            <li>Filtro PII: {compliance?.pii_filter ? 'activo' : 'sin datos'}</li>
            <li>Bloqueo evaluación automática: {compliance?.ai_grading_blocked ? 'activo' : 'sin datos'}</li>
          </ul>
        </div>

        <div className="section">
          <h3>Analítica de aprendizaje</h3>
          <ul className="features-list">
            <li>Ejercicios totales: {analytics?.totals?.exercises ?? '-'}</li>
            <li>Documentos totales: {analytics?.totals?.documents ?? '-'}</li>
            <li>Exámenes totales: {analytics?.totals?.exams ?? '-'}</li>
            <li>Reutilización (copias): {analytics?.library_reuse_estimate?.duplicated_items ?? '-'}</li>
          </ul>
        </div>

        <div className="section">
          <h3>Operación Enterprise</h3>
          <ul className="features-list">
            <li>Backend health: {backendHealth?.status || 'sin datos'}</li>
            <li>Workspace health: {workspaceHealth?.ready ? 'ready' : `not ready (${workspaceHealth?.reason || 'sin datos'})`}</li>
            <li>Ops ejercicios 24h: {opsMetrics?.window_24h?.exercises_created ?? '-'}</li>
            <li>Features activas: {enterpriseFeatures.length > 0 ? enterpriseFeatures.join(', ') : 'sin datos'}</li>
          </ul>
          <div className="generator-actions">
            <button className="btn-primary" onClick={runBackupNow} disabled={backupLoading}>
              {backupLoading ? 'Backup...' : 'Backup ahora'}
            </button>
            <button className="btn-primary" onClick={runRestoreLatest} disabled={restoreLoading}>
              {restoreLoading ? 'Restaurando...' : 'Restaurar último backup'}
            </button>
            <button className="btn-primary" onClick={loadAuditLog}>
              Cargar audit log
            </button>
          </div>
          {lastBackupResult && (
            <div className="ia-tool-output">
              <strong>Último backup:</strong>
              <pre>{JSON.stringify(lastBackupResult, null, 2)}</pre>
            </div>
          )}
          {lastRestoreResult && (
            <div className="ia-tool-output">
              <strong>Última restauración:</strong>
              <pre>{JSON.stringify(lastRestoreResult, null, 2)}</pre>
            </div>
          )}
        </div>

        <div className="section">
          <h3>Compliance Tools</h3>
          <div className="form-group">
            <label>Texto para anonimizar:</label>
            <textarea rows="3" value={anonymizeText} onChange={(e) => setAnonymizeText(e.target.value)} />
          </div>
          <div className="generator-actions">
            <button className="btn-primary" onClick={runAnonymizePreview}>Previsualizar anonimización</button>
          </div>
          {anonymizeResult && (
            <div className="ia-tool-output">
              <pre>{JSON.stringify(anonymizeResult, null, 2)}</pre>
            </div>
          )}
          {auditLogPreview.length > 0 && (
            <div className="ia-tool-output">
              <strong>Audit log (20 recientes)</strong>
              <pre>{JSON.stringify(auditLogPreview, null, 2)}</pre>
            </div>
          )}
        </div>

        <div className="section">
          <h3>Refinar por Proveedor</h3>
          <div className="form-group">
            <label>Proveedor:</label>
            <select value={refineProvider} onChange={(e) => setRefineProvider(e.target.value)}>
              {['openai', 'gemini', 'perplexity', 'deepseek', 'llama3', 'mistral', 'groq', 'glm', 'qwen', 'kimi'].map((provider) => (
                <option key={provider} value={provider}>{provider}</option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label>Texto:</label>
            <textarea rows="3" value={refineText} onChange={(e) => setRefineText(e.target.value)} />
          </div>
          <div className="generator-actions">
            <button className="btn-primary" onClick={runProviderRefine}>Refinar texto</button>
          </div>
          {refineResult && (
            <div className="ia-tool-output">
              <pre>{JSON.stringify(refineResult, null, 2)}</pre>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Dashboard
