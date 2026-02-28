import { useEffect, useRef, useState } from 'react'
import { getApiErrorMessage } from '../utils/api'

function ExerciseGenerator({ onGenerate }) {
  const [suggestedTopics, setSuggestedTopics] = useState(['les couleurs', 'au téléphone', 'les vêtements', 'objets du quotidien'])
  const [topic, setTopic] = useState('')
  const [difficulty, setDifficulty] = useState('intermediate')
  const [format, setFormat] = useState('magic_mix')
  const [aiModel, setAiModel] = useState('llama3')
  const [modelOptions, setModelOptions] = useState([{ value: 'llama3', label: 'llama3 (Local)', mode: 'local' }])
  const [localModels, setLocalModels] = useState(['llama3', 'mistral'])
  const [loading, setLoading] = useState(false)
  const [className, setClassName] = useState('6º Primaria')
  const [printPreset, setPrintPreset] = useState('teacher-clean')
  const [dualPrintOrder, setDualPrintOrder] = useState('student-teacher')
  const [generatedExercises, setGeneratedExercises] = useState([])
  const [savedExercises, setSavedExercises] = useState([])
  const [batchQuality, setBatchQuality] = useState(null)
  const [retryingRejected, setRetryingRejected] = useState(false)
  const [retryMeta, setRetryMeta] = useState(null)
  const [activeInteractiveExerciseId, setActiveInteractiveExerciseId] = useState('')
  const [interactiveSession, setInteractiveSession] = useState(null)
  const [interactiveAnswerSelected, setInteractiveAnswerSelected] = useState('')
  const [interactiveAnswerExpected, setInteractiveAnswerExpected] = useState('')
  const [interactiveSubmitResult, setInteractiveSubmitResult] = useState(null)
  const [interactiveScoreResult, setInteractiveScoreResult] = useState(null)
  const dualPrintFlowRef = useRef(null)
  const apiBaseUrl = import.meta.env.VITE_API_URL || ''

  const difficultyToLevel = {
    easy: 'A1',
    intermediate: 'A2',
    hard: 'B1'
  }

  const resolveAiMode = (modelValue) => (localModels.includes(modelValue) ? 'local' : 'cloud')

  const getActivityType = (exercise) => exercise?.content?.activity_type || 'fill_blank'

  const getActivityLabel = (exercise) => {
    const value = getActivityType(exercise)
    const labels = {
      fill_blank: 'Completar',
      matching: 'Relacionar',
      color_match: 'Colores',
      dialogue: 'Diálogo',
      image_choice: 'Elección visual',
      label_image: 'Etiquetar imagen',
      scene_story: 'Secuencia narrativa'
    }
    return labels[value] || value
  }

  const getPrintConfig = (preset) => {
    const [role, mode] = String(preset || 'teacher-clean').split('-')
    return {
      role: role === 'student' ? 'student' : 'teacher',
      mode: ['clean', 'ink', 'lines'].includes(mode) ? mode : 'clean'
    }
  }

  const normalizeWorksheetItems = (exercise) => {
    const items = exercise?.content?.items
    if (!Array.isArray(items)) return []
    return items
      .map((item) => ({
        question: item?.question || '',
        answer: (item?.correct_answer || '').trim(),
        options: Array.isArray(item?.options) ? item.options : [],
        hint: item?.hint || '',
        emoji: item?.emoji || '📝'
      }))
      .filter((item) => item.question && item.answer)
  }

  const answerBoxes = (answer) => {
    const letters = String(answer || '').split('')
    return (
      <div className="answer-boxes">
        {letters.map((char, idx) => (
          <span key={idx} className={`answer-box ${char === ' ' ? 'answer-box-space' : ''}`}>
            {char === ' ' ? '' : ' '}
          </span>
        ))}
      </div>
    )
  }

  const handlePrint = (preset = printPreset) => {
    const config = getPrintConfig(preset)
    document.body.setAttribute('data-print-mode', config.mode)
    document.body.setAttribute('data-print-role', config.role)
    setTimeout(() => window.print(), 30)
  }

  const handlePrintStudentTeacher = () => {
    const current = getPrintConfig(printPreset)
    const mode = current.mode
    const [firstRole, secondRole] = dualPrintOrder === 'teacher-student' ? ['teacher', 'student'] : ['student', 'teacher']

    dualPrintFlowRef.current = {
      mode,
      firstRole,
      secondRole,
      phase: 'first',
      originalPreset: printPreset
    }

    const firstPreset = `${firstRole}-${mode}`
    setPrintPreset(firstPreset)
    setTimeout(() => handlePrint(firstPreset), 60)
  }

  const handlePrintExercise = (exercise) => {
    setGeneratedExercises([exercise])
    setTimeout(() => handlePrint(), 60)
  }

  const handleExportGoogleWorkspace = async (exercise) => {
    if (!exercise?.id) {
      alert('El ejercicio debe estar guardado para exportar')
      return
    }
    try {
      const response = await fetch(`${apiBaseUrl}/api/library/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ item_type: 'exercise', item_id: exercise.id, format: 'google_workspace' })
      })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data?.error?.message || data?.error || 'No se pudo exportar a Google Workspace')
      }
      const notes = Array.isArray(data?.notes) ? `\n- ${data.notes.join('\n- ')}` : ''
      alert(`Exportación Google Workspace lista: ${data.filename}${notes}`)
      if (data?.workspace_links?.drive_upload) {
        window.open(data.workspace_links.drive_upload, '_blank', 'noopener,noreferrer')
      }
    } catch (error) {
      console.error(error)
      alert(error.message || 'Error en exportación Google Workspace')
    }
  }

  const handlePublishGoogleWorkspace = async (exercise) => {
    if (!exercise?.id) {
      alert('El ejercicio debe estar guardado para publicar')
      return
    }
    try {
      const response = await fetch(`${apiBaseUrl}/api/google/workspace/publish`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          item_type: 'exercise',
          item_id: exercise.id,
          class_name: className || 'Clase'
        })
      })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data?.error?.message || data?.error || 'No se pudo publicar en Google Workspace')
      }
      alert(`Publicado en Google Docs: ${data.doc_name}\nClase: ${data.class_name}`)
      if (data.doc_url) {
        window.open(data.doc_url, '_blank', 'noopener,noreferrer')
      }
    } catch (error) {
      console.error(error)
      alert(error.message || 'Error al publicar en Google Workspace')
    }
  }

  const handleExportPdfWorksheet = async (exercise) => {
    if (!exercise?.id) {
      alert('El ejercicio debe estar guardado para exportar PDF')
      return
    }
    try {
      const response = await fetch(`${apiBaseUrl}/api/library/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ item_type: 'exercise', item_id: exercise.id, format: 'pdf' })
      })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data?.error?.message || data?.error || 'No se pudo exportar PDF')
      }
      if (data.download_url) {
        window.open(`${apiBaseUrl}${data.download_url}`, '_blank', 'noopener,noreferrer')
      } else {
        alert(`PDF exportado: ${data.filename || data.path}`)
      }
    } catch (error) {
      alert(error.message || 'Error al exportar PDF')
    }
  }

  const loadSavedExercises = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/exercises`)
      if (!response.ok) return
      const data = await response.json()
      const sorted = [...data].sort((a, b) => (b.id || 0) - (a.id || 0))
      setSavedExercises(sorted)
    } catch (error) {
      console.error('Error loading saved exercises:', error)
    }
  }

  const loadTemplates = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/exercises/templates`)
      if (!response.ok) return
      const data = await response.json()
      if (Array.isArray(data?.themes) && data.themes.length > 0) {
        setSuggestedTopics(data.themes)
      }
    } catch (error) {
      console.error('Error loading templates:', error)
    }
  }

  const loadAiModels = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/ai/models`)
      if (!response.ok) return
      const data = await response.json()
      const local = Array.isArray(data?.local) ? data.local : []
      const cloud = Array.isArray(data?.cloud) ? data.cloud : []
      const options = [
        ...local.map((model) => ({ value: model, label: `${model} (Local)`, mode: 'local' })),
        ...cloud.map((model) => ({ value: model, label: `${model} (Cloud)`, mode: 'cloud' }))
      ]
      if (options.length === 0) {
        setModelOptions([{ value: 'llama3', label: 'llama3 (Local)', mode: 'local' }])
        setLocalModels(['llama3', 'mistral'])
        return
      }

      setModelOptions(options)
      setLocalModels(local)
      if (!options.some((opt) => opt.value === aiModel)) {
        setAiModel(options[0].value)
      }
    } catch (error) {
      console.error('Error loading AI models:', error)
    }
  }

  const handleGenerate = async () => {
    if (!topic.trim()) {
      alert('Por favor ingresa un tema')
      return
    }

    setLoading(true)
    try {
      const response = await fetch(`${apiBaseUrl}/api/exercises/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic,
          level: difficultyToLevel[difficulty] || 'A2',
          exercise_type: format,
          ai_mode: resolveAiMode(aiModel),
          ai_model: aiModel
        })
      })

      if (!response.ok) {
        const errorMessage = await getApiErrorMessage(response, 'Error al generar ejercicios')
        throw new Error(errorMessage)
      }

      const data = await response.json()
      const exercises = [data]
      setGeneratedExercises(exercises)
      onGenerate(exercises)
      await loadSavedExercises()
    } catch (error) {
      console.error('Error:', error)
      alert(error.message || 'Error al generar ejercicios')
    } finally {
      setLoading(false)
    }
  }

  const handleGenerateBatch = async () => {
    await handleGenerateBatchRealtime(5)
  }

  const reasonToLabel = (reason) => {
    const map = {
      low_variety: 'Baja variedad',
      schema_low_coverage: 'Estructura incompleta',
      too_few_items: 'Muy pocos ítems',
      unexpected_activity_type: 'Tipo de actividad no esperado',
      quality_rejected: 'Rechazado por calidad',
      api_error: 'Error de API',
      api_error_retry: 'Error de API en reintento'
    }
    return map[reason] || reason
  }

  const handleGenerateBatchRealtime = async (count) => {
    const pool = topic.trim()
      ? [topic.trim(), ...suggestedTopics]
      : [...suggestedTopics, 'vocabulaire scolaire', 'les sports', 'la maison']
    const requestedCount = Math.max(1, Number(count) || 5)
    const acceptedItems = []
    const rejectedItems = []

    setBatchQuality({
      running: true,
      total: requestedCount,
      processed: 0,
      accepted: 0,
      rejected: 0,
      acceptedItems: [],
      rejectedItems: [],
      startedAt: new Date().toISOString(),
      finishedAt: null
    })
    setLoading(true)
    try {
      setGeneratedExercises([])
      for (let i = 0; i < requestedCount; i += 1) {
        const currentTopic = pool[i % pool.length]
        const response = await fetch(`${apiBaseUrl}/api/exercises/generate-batch`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            default_level: difficultyToLevel[difficulty] || 'A2',
            default_exercise_type: format,
            default_ai_mode: resolveAiMode(aiModel),
            default_ai_model: aiModel,
            reject_low_quality: true,
            items: [{ topic: currentTopic }]
          })
        })
        if (!response.ok) {
          const errorMessage = await getApiErrorMessage(response, 'Error al generar lote de ejercicios')
          rejectedItems.push({
            topic: currentTopic,
            reasons: ['api_error'],
            detail: errorMessage
          })
        } else {
          const payload = await response.json()
          const created = Array.isArray(payload?.created) ? payload.created : []
          const errors = Array.isArray(payload?.errors) ? payload.errors : []

          if (created.length > 0) {
            acceptedItems.push(...created)
            setGeneratedExercises((prev) => [...prev, ...created])
          }
          if (errors.length > 0) {
            errors.forEach((err) => {
              rejectedItems.push({
                topic: err?.topic || currentTopic,
                reasons: Array.isArray(err?.quality?.reasons) && err.quality.reasons.length > 0 ? err.quality.reasons : [err?.error || 'quality_rejected'],
                detail: err?.error || '',
                quality: err?.quality || null
              })
            })
          }
        }

        setBatchQuality({
          running: true,
          total: requestedCount,
          processed: i + 1,
          accepted: acceptedItems.length,
          rejected: rejectedItems.length,
          acceptedItems: [...acceptedItems],
          rejectedItems: [...rejectedItems]
        })
      }

      setBatchQuality((prev) => ({
        ...(prev || {}),
        running: false,
        processed: requestedCount,
        accepted: acceptedItems.length,
        rejected: rejectedItems.length,
        acceptedItems,
        rejectedItems,
        finishedAt: new Date().toISOString()
      }))

      onGenerate(acceptedItems)
      await loadSavedExercises()
    } catch (error) {
      console.error('Error:', error)
      alert(error.message || 'Error al generar ejercicios de prueba')
      setBatchQuality((prev) => ({
        ...(prev || {}),
        running: false,
        error: error.message || 'Error en lote',
        finishedAt: new Date().toISOString()
      }))
    } finally {
      setLoading(false)
    }
  }

  const resolveFallbackModel = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/ai/repair/gemini-deepseek`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      if (response.ok) {
        const data = await response.json()
        const suggested = data?.deepseek_fallback?.provider
        if (suggested && modelOptions.some((opt) => opt.value === suggested) && suggested !== aiModel) {
          return { model: suggested, reason: `fallback recomendado: ${suggested}` }
        }
      }
    } catch (error) {
      console.warn('No se pudo obtener fallback recomendado:', error)
    }

    const cloudAlternative = modelOptions.find((opt) => opt.mode === 'cloud' && opt.value !== aiModel)
    if (cloudAlternative) return { model: cloudAlternative.value, reason: `siguiente modelo cloud: ${cloudAlternative.value}` }
    const anyAlternative = modelOptions.find((opt) => opt.value !== aiModel)
    if (anyAlternative) return { model: anyAlternative.value, reason: `siguiente modelo disponible: ${anyAlternative.value}` }
    return { model: aiModel, reason: 'sin alternativa, se reutiliza modelo actual' }
  }

  const retryRejectedBatch = async () => {
    const rejected = Array.isArray(batchQuality?.rejectedItems) ? batchQuality.rejectedItems : []
    if (rejected.length === 0 || retryingRejected || loading) return

    setRetryingRejected(true)
    try {
      const fallback = await resolveFallbackModel()
      setRetryMeta({ startedAt: new Date().toISOString(), model: fallback.model, reason: fallback.reason })
      const acceptedRetry = []
      const stillRejected = []

      for (let i = 0; i < rejected.length; i += 1) {
        const item = rejected[i]
        const currentTopic = item?.topic || topic || suggestedTopics[0] || 'français'
        const response = await fetch(`${apiBaseUrl}/api/exercises/generate-batch`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            default_level: difficultyToLevel[difficulty] || 'A2',
            default_exercise_type: format,
            default_ai_mode: resolveAiMode(fallback.model),
            default_ai_model: fallback.model,
            reject_low_quality: true,
            items: [{ topic: currentTopic }]
          })
        })

        if (!response.ok) {
          const errorMessage = await getApiErrorMessage(response, 'Error en reintento de rechazado')
          stillRejected.push({
            topic: currentTopic,
            reasons: ['api_error_retry'],
            detail: errorMessage
          })
          continue
        }

        const payload = await response.json()
        const created = Array.isArray(payload?.created) ? payload.created : []
        const errors = Array.isArray(payload?.errors) ? payload.errors : []
        if (created.length > 0) {
          acceptedRetry.push(...created)
          setGeneratedExercises((prev) => [...prev, ...created])
        }
        if (errors.length > 0) {
          errors.forEach((err) => {
            stillRejected.push({
              topic: err?.topic || currentTopic,
              reasons: Array.isArray(err?.quality?.reasons) && err.quality.reasons.length > 0 ? err.quality.reasons : [err?.error || 'quality_rejected'],
              detail: err?.error || '',
              quality: err?.quality || null
            })
          })
        }

        setBatchQuality((prev) => ({
          ...(prev || {}),
          rejected: Math.max(0, (rejected.length - (i + 1)) + stillRejected.length),
          accepted: (prev?.accepted || 0) + acceptedRetry.length,
          rejectedItems: [...stillRejected]
        }))
      }

      if (acceptedRetry.length > 0) {
        onGenerate(acceptedRetry)
        await loadSavedExercises()
      }

      setBatchQuality((prev) => ({
        ...(prev || {}),
        accepted: (prev?.accepted || 0) + acceptedRetry.length,
        rejected: stillRejected.length,
        acceptedItems: [...(prev?.acceptedItems || []), ...acceptedRetry],
        rejectedItems: stillRejected
      }))

      setRetryMeta((prev) => ({
        ...(prev || {}),
        finishedAt: new Date().toISOString(),
        accepted: acceptedRetry.length,
        stillRejected: stillRejected.length
      }))
    } catch (error) {
      alert(error.message || 'Error al reintentar rechazados')
    } finally {
      setRetryingRejected(false)
    }
  }

  const startInteractiveSession = async () => {
    const targetId = Number(activeInteractiveExerciseId)
    if (!targetId) {
      alert('Selecciona un ID de ejercicio válido para iniciar sesión interactiva')
      return
    }
    try {
      const refreshRes = await fetch(`${apiBaseUrl}/api/exercises/${targetId}`)
      if (!refreshRes.ok) {
        const message = await getApiErrorMessage(refreshRes, 'No se pudo cargar ejercicio por ID')
        throw new Error(message)
      }
      const response = await fetch(`${apiBaseUrl}/api/interactive/session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ exercise_id: targetId, student_id: 'ui-student' })
      })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data?.error?.message || data?.error || 'No se pudo iniciar sesión interactiva')
      }
      setInteractiveSession(data)
      setInteractiveSubmitResult(null)
      setInteractiveScoreResult(null)
    } catch (error) {
      alert(error.message || 'Error iniciando sesión interactiva')
    }
  }

  const submitInteractiveSession = async () => {
    if (!interactiveSession?.session_id) {
      alert('Inicia una sesión interactiva primero')
      return
    }
    try {
      const response = await fetch(`${apiBaseUrl}/api/interactive/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: interactiveSession.session_id,
          answers: [{ selected: interactiveAnswerSelected, expected: interactiveAnswerExpected }]
        })
      })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data?.error?.message || data?.error || 'No se pudo enviar sesión')
      }
      setInteractiveSubmitResult(data)
    } catch (error) {
      alert(error.message || 'Error enviando sesión interactiva')
    }
  }

  const scoreInteractiveAnswers = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/interactive/score`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          answers: [{ selected: interactiveAnswerSelected, expected: interactiveAnswerExpected }]
        })
      })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data?.error?.message || data?.error || 'No se pudo puntuar respuesta')
      }
      setInteractiveScoreResult(data)
    } catch (error) {
      alert(error.message || 'Error en score interactivo')
    }
  }

  const handleDeleteExercise = async (exerciseId) => {
    const targetId = Number(exerciseId)
    if (!targetId) return
    const confirmed = window.confirm(`¿Eliminar ejercicio ID ${targetId}? Esta acción no se puede deshacer.`)
    if (!confirmed) return
    try {
      const response = await fetch(`${apiBaseUrl}/api/exercises/${targetId}`, {
        method: 'DELETE'
      })
      if (!response.ok) {
        const errorMessage = await getApiErrorMessage(response, 'No se pudo eliminar ejercicio')
        throw new Error(errorMessage)
      }
      setGeneratedExercises((prev) => prev.filter((item) => Number(item.id) !== targetId))
      setSavedExercises((prev) => prev.filter((item) => Number(item.id) !== targetId))
      if (Number(activeInteractiveExerciseId) === targetId) {
        setActiveInteractiveExerciseId('')
        setInteractiveSession(null)
      }
      await loadSavedExercises()
    } catch (error) {
      alert(error.message || 'Error al eliminar ejercicio')
    }
  }

  const renderExerciseWorksheet = (exercise) => {
    const activityType = getActivityType(exercise)
    const items = Array.isArray(exercise?.content?.items) ? exercise.content.items : []

    if (items.length === 0) {
      return <p>{exercise?.content?.question || exercise?.question || 'Sin contenido'}</p>
    }

    return (
      <div className="worksheet-card">
        <div className="worksheet-header">
          <h4>{exercise?.content?.title || `Ficha: ${exercise.topic}`} · {getActivityLabel(exercise)}</h4>
          <div className="worksheet-header-actions">
            <button className="btn-primary" onClick={() => handlePrint()}>Imprimir ficha</button>
            <button className="btn-primary" onClick={() => handleExportGoogleWorkspace(exercise)}>Google Workspace</button>
            <button className="btn-primary" onClick={() => handlePublishGoogleWorkspace(exercise)}>Publicar Drive/Docs</button>
          </div>
        </div>

        <div className="worksheet-table">
          {activityType === 'matching' && (
            <div className="magic-matching">
              {items.map((item, idx) => (
                <div key={idx} className="magic-row">
                  <span>{item.emoji || '🧩'} {item.left}</span>
                  <span className="magic-arrow">→</span>
                  <span className="solution-cell">{item.right}</span>
                </div>
              ))}
            </div>
          )}

          {activityType === 'color_match' && (
            <div className="magic-colors">
              {items.map((item, idx) => (
                <div key={idx} className="magic-row">
                  <span>{item.word}</span>
                  <span className="magic-arrow">→</span>
                  <span className="solution-cell">{item.color_label}</span>
                  <span className="color-dot" style={{ backgroundColor: item.color_hex || '#999' }} />
                </div>
              ))}
            </div>
          )}

          {activityType === 'dialogue' && (
            <div className="magic-dialogue">
              {items.map((item, idx) => (
                <div key={idx} className="dialogue-line">
                  <strong>{item.emoji || '💬'} {item.speaker || `Personaje ${idx + 1}`}:</strong> {item.line_with_blank}
                  {Array.isArray(item.options) && item.options.length > 0 && (
                    <div className="worksheet-options">Opciones: {item.options.join(' | ')}</div>
                  )}
                </div>
              ))}
            </div>
          )}

          {activityType === 'image_choice' && (
            <div className="image-choice-grid">
              {items.map((item, idx) => (
                <div key={idx} className="image-choice-card">
                  <h5>{idx + 1}. {item.question}</h5>
                  <div className="image-choice-options">
                    {(item.choices || []).map((choice, cIdx) => (
                      <div key={`${idx}-${cIdx}`} className="image-choice-option">
                        <img src={choice.image_url} alt={choice.label} className="image-choice-thumb" />
                        <span>{choice.label}</span>
                      </div>
                    ))}
                  </div>
                  {item.hint && <div className="worksheet-hint">Pista: {item.hint}</div>}
                </div>
              ))}
            </div>
          )}

          {activityType === 'label_image' && (
            <div className="label-image-grid">
              {items.map((item, idx) => (
                <div key={idx} className="label-image-card">
                  <img src={item.image_url} alt={item.label} className="label-image-thumb" />
                  <div className="label-image-meta">
                    <strong>{item.emoji || '🏷️'} {item.label}</strong>
                    <span>{item.definition}</span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {activityType === 'scene_story' && (
            <div className="scene-story-list">
              {items.map((item, idx) => (
                <div key={idx} className="scene-story-row">
                  <img src={item.image_url} alt={`escena ${idx + 1}`} className="scene-story-thumb" />
                  <div className="scene-story-content">
                    <div className="scene-story-sentence">{item.sentence}</div>
                    <div className="scene-story-order">Orden correcto: {item.correct_order}</div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {activityType === 'fill_blank' && (
            <>
              {normalizeWorksheetItems(exercise).map((item, itemIndex) => (
                <div key={itemIndex} className="worksheet-row">
                  <div className="worksheet-emoji">{item.emoji}</div>
                  <div className="worksheet-main">
                    <div className="worksheet-question">{itemIndex + 1}. {item.question}</div>
                    {answerBoxes(item.answer)}
                    {item.options.length > 0 && <div className="worksheet-options">Opciones: {item.options.join(' | ')}</div>}
                    {item.hint && <div className="worksheet-hint">Pista: {item.hint}</div>}
                  </div>
                </div>
              ))}
            </>
          )}
        </div>

        <div className="worksheet-answer-key">
          <h5>Soluciones</h5>
          <ol>
            {items.map((item, idx) => (
              <li key={`answer-${idx}`}>
                {activityType === 'matching' && `${item.left} -> ${item.right}`}
                {activityType === 'color_match' && `${item.word} -> ${item.color_label}`}
                {activityType === 'dialogue' && `${item.speaker || `Personaje ${idx + 1}`}: ${item.correct_answer || '-'}`}
                {activityType === 'fill_blank' && `${item.question || `Pregunta ${idx + 1}`} -> ${item.correct_answer || '-'}`}
                {activityType === 'image_choice' && `${item.question || `Pregunta ${idx + 1}`} -> ${item.correct_answer || '-'}`}
                {activityType === 'label_image' && `${item.label || `Elemento ${idx + 1}`} -> ${item.definition || '-'}`}
                {activityType === 'scene_story' && `${item.sentence || `Escena ${idx + 1}`} -> ${item.correct_order || idx + 1}`}
              </li>
            ))}
          </ol>
        </div>
      </div>
    )
  }

  useEffect(() => {
    loadSavedExercises()
    loadTemplates()
    loadAiModels()
  }, [])

  useEffect(() => {
    const config = getPrintConfig(printPreset)
    document.body.setAttribute('data-print-mode', config.mode)
    document.body.setAttribute('data-print-role', config.role)
  }, [printPreset])

  useEffect(() => {
    const onAfterPrint = () => {
      const flow = dualPrintFlowRef.current
      if (!flow) return
      if (flow.phase === 'first') {
        flow.phase = 'second'
        const secondPreset = `${flow.secondRole}-${flow.mode}`
        setPrintPreset(secondPreset)
        setTimeout(() => handlePrint(secondPreset), 120)
        return
      }
      setPrintPreset(flow.originalPreset)
      dualPrintFlowRef.current = null
    }

    window.addEventListener('afterprint', onAfterPrint)
    return () => window.removeEventListener('afterprint', onAfterPrint)
  }, [printPreset])

  return (
    <div className="exercise-generator">
      <h2>Generador de Ejercicios</h2>

      <div className="form-group">
        <label>Tema:</label>
        <input
          type="text"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="Ej: les couleurs, au téléphone, les vêtements"
        />
        <div className="topic-suggestions">
          {suggestedTopics.map((item) => (
            <button key={item} type="button" className="topic-chip" onClick={() => setTopic(item)}>{item}</button>
          ))}
        </div>
      </div>

      <div className="form-row">
        <div className="form-group">
          <label>Dificultad:</label>
          <select value={difficulty} onChange={(e) => setDifficulty(e.target.value)}>
            <option value="easy">Fácil</option>
            <option value="intermediate">Intermedio</option>
            <option value="hard">Difícil</option>
          </select>
        </div>

        <div className="form-group">
          <label>Modelo IA:</label>
          <select value={aiModel} onChange={(e) => setAiModel(e.target.value)}>
            {modelOptions.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label>Formato creativo:</label>
          <select value={format} onChange={(e) => setFormat(e.target.value)}>
            <option value="magic_mix">Sorpresa (mezcla creativa)</option>
            <option value="fill_blank">Completar</option>
            <option value="matching">Relacionar</option>
            <option value="color_match">Colores</option>
            <option value="dialogue">Diálogo</option>
            <option value="image_choice">Elección visual</option>
            <option value="label_image">Etiquetar imagen</option>
            <option value="scene_story">Secuencia narrativa</option>
          </select>
        </div>
      </div>

      <div className="generator-actions">
        <button onClick={handleGenerate} disabled={loading} className="btn-primary">
          {loading ? 'Generando...' : 'Generar Ejercicios'}
        </button>
        <button onClick={handleGenerateBatch} disabled={loading} className="btn-primary">
          {loading ? 'Generando lote...' : 'Generar 5 ejercicios (calidad)'}
        </button>
        <button onClick={() => handleGenerateBatchRealtime(20)} disabled={loading} className="btn-primary">
          {loading ? 'Generando lote...' : 'Generar 20 ejercicios (calidad)'}
        </button>
      </div>

      {batchQuality && (
        <div className="batch-quality-panel">
          <h3>Calidad del lote</h3>
          <div className="batch-quality-summary">
            <span>Total: {batchQuality.total}</span>
            <span>Procesados: {batchQuality.processed}</span>
            <span className="detail-ok">Aceptados: {batchQuality.accepted}</span>
            <span className="detail-error">Rechazados: {batchQuality.rejected}</span>
            <span>{batchQuality.running ? 'En tiempo real...' : 'Finalizado'}</span>
          </div>
          <div className="batch-quality-progress">
            <div
              className="batch-quality-progress-bar"
              style={{ width: `${Math.round(((batchQuality.processed || 0) / Math.max(1, batchQuality.total || 1)) * 100)}%` }}
            />
          </div>
          {batchQuality.error && (
            <div className="provider-health-detail detail-error">{batchQuality.error}</div>
          )}
          {Array.isArray(batchQuality.rejectedItems) && batchQuality.rejectedItems.length > 0 && (
            <div className="batch-quality-rejected">
              <strong>Rechazados y motivo:</strong>
              <div className="generator-actions">
                <button
                  type="button"
                  className="btn-primary"
                  onClick={retryRejectedBatch}
                  disabled={retryingRejected || loading}
                >
                  {retryingRejected ? 'Reintentando rechazados...' : 'Reintentar solo rechazados (fallback)'}
                </button>
              </div>
              {retryMeta && (
                <div className="provider-health-detail detail-ok">
                  Reintento con modelo: <strong>{retryMeta.model}</strong> · {retryMeta.reason}
                </div>
              )}
              <ul className="wizard-checklist">
                {batchQuality.rejectedItems.map((item, idx) => (
                  <li key={`rejected-${idx}`}>
                    <strong>{item.topic}</strong> · {(item.reasons || []).map((r) => reasonToLabel(r)).join(', ')}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      <div className="form-group">
        <label>Clase destino (Drive/Docs):</label>
        <input
          type="text"
          value={className}
          onChange={(e) => setClassName(e.target.value)}
          placeholder="Ej: 6º Primaria A"
        />
      </div>

      <div className="form-group">
        <label>Interactividad (por ID de ejercicio):</label>
        <input
          type="number"
          value={activeInteractiveExerciseId}
          onChange={(e) => setActiveInteractiveExerciseId(e.target.value)}
          placeholder="Ej: 72"
        />
        <div className="generator-actions">
          <button type="button" className="btn-primary" onClick={startInteractiveSession}>
            Iniciar sesión interactiva
          </button>
        </div>
        <div className="form-row">
          <div className="form-group">
            <label>Respuesta seleccionada</label>
            <input
              type="text"
              value={interactiveAnswerSelected}
              onChange={(e) => setInteractiveAnswerSelected(e.target.value)}
              placeholder="respuesta alumno"
            />
          </div>
          <div className="form-group">
            <label>Respuesta esperada</label>
            <input
              type="text"
              value={interactiveAnswerExpected}
              onChange={(e) => setInteractiveAnswerExpected(e.target.value)}
              placeholder="respuesta correcta"
            />
          </div>
        </div>
        <div className="generator-actions">
          <button type="button" className="btn-primary" onClick={submitInteractiveSession}>
            Enviar sesión
          </button>
          <button type="button" className="btn-primary" onClick={scoreInteractiveAnswers}>
            Puntuar respuesta
          </button>
        </div>
        {interactiveSession && (
          <div className="ia-tool-output">
            <strong>Sesión:</strong>
            <pre>{JSON.stringify(interactiveSession, null, 2)}</pre>
          </div>
        )}
        {interactiveSubmitResult && (
          <div className="ia-tool-output">
            <strong>Resultado submit:</strong>
            <pre>{JSON.stringify(interactiveSubmitResult, null, 2)}</pre>
          </div>
        )}
        {interactiveScoreResult && (
          <div className="ia-tool-output">
            <strong>Resultado score:</strong>
            <pre>{JSON.stringify(interactiveScoreResult, null, 2)}</pre>
          </div>
        )}
      </div>

      <div className="print-mode-toolbar">
        <label htmlFor="printPreset">Versión impresión:</label>
        <select id="printPreset" value={printPreset} onChange={(e) => setPrintPreset(e.target.value)}>
          <option value="teacher-clean">Profesor · A4 limpia (con soluciones)</option>
          <option value="student-clean">Alumno · A4 limpia (sin soluciones)</option>
          <option value="teacher-ink">Profesor · Tinta reducida (con soluciones)</option>
          <option value="student-ink">Alumno · Tinta reducida (sin soluciones)</option>
          <option value="teacher-lines">Profesor · Líneas respuesta (con soluciones)</option>
          <option value="student-lines">Alumno · Líneas respuesta (sin soluciones)</option>
        </select>
        <select id="dualPrintOrder" value={dualPrintOrder} onChange={(e) => setDualPrintOrder(e.target.value)}>
          <option value="student-teacher">Orden: Alumno + Profesor</option>
          <option value="teacher-student">Orden: Profesor + Alumno</option>
        </select>
        <button type="button" className="btn-primary" onClick={handlePrintStudentTeacher}>Imprimir versión doble</button>
      </div>

      {generatedExercises.length > 0 && (
        <div className="exercises-list">
          <h3>Ejercicios Generados:</h3>
          {generatedExercises.map((exercise, index) => (
            <div key={exercise.id || index} className="exercise-card">
              <div className="exercise-number">Ejercicio {index + 1} · {getActivityLabel(exercise)}</div>
              {(exercise.image_url || exercise?.content?.image_url) && (
                <img
                  className="preview-image"
                  src={exercise.image_url || exercise?.content?.image_url}
                  alt={`Ilustracion ejercicio ${index + 1}`}
                />
              )}
              {renderExerciseWorksheet(exercise)}
            </div>
          ))}
        </div>
      )}

      <div className="saved-exercises-section">
        <h3>Historial de ejercicios guardados</h3>
        {savedExercises.length === 0 && <p>No hay ejercicios guardados todavía.</p>}
        {savedExercises.map((exercise) => (
          <div key={exercise.id} className="exercise-card">
            <div className="exercise-number">ID {exercise.id} · {exercise.title} · {getActivityLabel(exercise)}</div>
            {(exercise.image_url || exercise?.content?.image_url) && (
              <img className="preview-image" src={exercise.image_url || exercise?.content?.image_url} alt={`Ilustracion ejercicio ${exercise.id}`} />
            )}
            {renderExerciseWorksheet(exercise)}
            <div className="generator-actions">
              <button className="btn-primary" onClick={() => handlePrintExercise(exercise)}>Imprimir ejercicio</button>
              <button className="btn-primary" onClick={() => handleExportPdfWorksheet(exercise)}>Exportar PDF ficha</button>
              <button className="btn-primary" onClick={() => handleExportGoogleWorkspace(exercise)}>Exportar Google Workspace</button>
              <button className="btn-primary" onClick={() => handlePublishGoogleWorkspace(exercise)}>Publicar Drive/Docs</button>
              <button className="btn-primary" onClick={() => handleDeleteExercise(exercise.id)}>Eliminar</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default ExerciseGenerator
