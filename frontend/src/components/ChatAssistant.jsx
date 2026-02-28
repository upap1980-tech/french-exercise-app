import { useEffect, useMemo, useRef, useState } from 'react'
import { getApiErrorMessage } from '../utils/api'

const PROMPT_PRESETS = [
  { label: 'Ejercicio rápido', goal: 'exercise_gen', text: 'Crea una ficha corta para primaria con 6 preguntas claras y visuales.' },
  { label: 'Examen completo', goal: 'exam_gen', text: 'Crea un examen de 20 minutos con instrucciones simples y rúbrica de puntuación.' },
  { label: 'Actividad oral', goal: 'interactive_gen', text: 'Crea una actividad oral por parejas con turnos y frases modelo.' },
  { label: 'Juego en clase', goal: 'exercise_gen', text: 'Diseña un juego de aula con dinámica, reglas y evaluación rápida.' }
]

const REVISION_ACTIONS = [
  'Hazlo más fácil',
  'Hazlo más difícil',
  'Hazlo más corto',
  'Hazlo más visual',
  'Hazlo más divertido',
  'Añade instrucciones paso a paso'
]

function ChatAssistant({ onExerciseCreated }) {
  const apiBaseUrl = import.meta.env.VITE_API_URL || (typeof window !== 'undefined' ? `http://localhost:5012` : '')
  const [messages, setMessages] = useState([])
  const [requestText, setRequestText] = useState('')
  const [topic, setTopic] = useState('les vêtements')
  const [level, setLevel] = useState('A2')
  const [goal, setGoal] = useState('exercise_gen')
  const [provider, setProvider] = useState('llama3')
  const [model, setModel] = useState('llama3')
  const [modelsByProvider, setModelsByProvider] = useState({})
  const [sending, setSending] = useState(false)
  const [assistantTyping, setAssistantTyping] = useState(false)
  const [streamAvailable, setStreamAvailable] = useState(true)
  const [generatedImageUrl, setGeneratedImageUrl] = useState('')
  const [videoJobInfo, setVideoJobInfo] = useState(null)
  const [generatedAudioUrl, setGeneratedAudioUrl] = useState('')
  const [toastMessage, setToastMessage] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [showMoreOptions, setShowMoreOptions] = useState(false)
  const [gameResults, setGameResults] = useState({})
  const [exerciseByMessage, setExerciseByMessage] = useState({})
  const [cardStatus, setCardStatus] = useState({})
  const abortControllerRef = useRef(null)
  const cancelRequestedRef = useRef(false)

  const providerOptions = useMemo(() => Object.keys(modelsByProvider), [modelsByProvider])

  const loadModels = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/ai/models`)
      if (!response.ok) return
      const data = await response.json()
      const map = {}
      ;(data.local || []).forEach((name) => { map[name] = [name] })
      ;(data.cloud || []).forEach((name) => { map[name] = [name] })
      if (Object.keys(map).length === 0) map.llama3 = ['llama3']
      setModelsByProvider(map)
      const first = Object.keys(map)[0]
      if (first) {
        setProvider(first)
        setModel(map[first][0] || first)
      }
    } catch (error) {
      console.error('Error loading models', error)
    }
  }

  const loadMessages = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/chat/messages?limit=120`)
      if (!response.ok) return
      const data = await response.json()
      setMessages(data)
    } catch (error) {
      console.error('Error loading chat messages', error)
    }
  }

  useEffect(() => {
    loadModels()
    loadMessages()
  }, [])

  useEffect(() => {
    const models = modelsByProvider[provider] || []
    if (models.length > 0) setModel(models[0])
  }, [provider, modelsByProvider])

  useEffect(() => {
    if (!toastMessage) return undefined
    const timer = window.setTimeout(() => setToastMessage(''), 2600)
    return () => window.clearTimeout(timer)
  }, [toastMessage])

  const sendMessage = async (textOverride = null, goalOverride = null) => {
    const message = String(textOverride ?? requestText).trim()
    if (!message || sending) return
    const taskType = goalOverride || goal

    setSending(true)
    setAssistantTyping(true)
    try {
      cancelRequestedRef.current = false
      const supportsStreaming = streamAvailable && taskType === 'chat'
      const controller = supportsStreaming ? new AbortController() : null
      abortControllerRef.current = controller

      const userTemp = {
        id: `temp-user-${Date.now()}`,
        role: 'user',
        content: message,
        task_type: taskType,
        provider,
        model
      }
      const assistantTempId = `temp-assistant-${Date.now()}`
      const assistantTemp = {
        id: assistantTempId,
        role: 'assistant',
        content: '',
        task_type: taskType,
        provider,
        model
      }
      setMessages((prev) => [...prev, userTemp, assistantTemp])
      setRequestText('')

      const payload = {
        message,
        task_type: taskType,
        provider,
        model,
        context: { topic, level }
      }

      const applyDone = (finalMessage) => {
        setMessages((prev) => prev.map((msg) => (msg.id === assistantTempId ? finalMessage : msg)))
      }

      const sendNonStreaming = async () => {
        const fallbackResponse = await fetch(`${apiBaseUrl}/api/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        })
        if (!fallbackResponse.ok) {
          throw new Error(await getApiErrorMessage(fallbackResponse, 'No se pudo enviar el mensaje'))
        }
        const fallbackData = await fallbackResponse.json()
        if (fallbackData?.message) {
          const enriched = {
            ...fallbackData.message,
            preview: fallbackData.preview || null
          }
          applyDone(enriched)
        }
      }

      if (supportsStreaming) {
        const response = await fetch(`${apiBaseUrl}/api/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: controller.signal,
          body: JSON.stringify(payload)
        })
        if (response.status === 404 || response.status === 405) {
          setStreamAvailable(false)
          setToastMessage('Streaming no disponible. Usando modo estándar.')
          await sendNonStreaming()
        } else {
          if (!response.ok) {
            throw new Error(await getApiErrorMessage(response, 'No se pudo enviar el mensaje'))
          }
          const reader = response.body?.getReader()
          if (!reader) throw new Error('No hay streaming disponible en esta sesión')
          const decoder = new TextDecoder('utf-8')
          let buffer = ''
          const applyToken = (token) => {
            setMessages((prev) =>
              prev.map((msg) => (msg.id === assistantTempId ? { ...msg, content: `${msg.content || ''}${token}` } : msg))
            )
          }
          while (true) {
            const { done, value } = await reader.read()
            if (done) break
            buffer += decoder.decode(value, { stream: true })
            const frames = buffer.split('\n\n')
            buffer = frames.pop() || ''
            for (const frame of frames) {
              const lines = frame.split('\n')
              const eventLine = lines.find((line) => line.startsWith('event:'))
              const dataLine = lines.find((line) => line.startsWith('data:'))
              if (!eventLine || !dataLine) continue
              const event = eventLine.replace('event:', '').trim()
              const jsonText = dataLine.replace('data:', '').trim()
              let parsed = null
              try {
                parsed = JSON.parse(jsonText)
              } catch {
                parsed = null
              }
              if (event === 'token' && parsed?.token) applyToken(parsed.token)
              if (event === 'done' && parsed?.message) applyDone(parsed.message)
              if (event === 'error') throw new Error(parsed?.error || 'Error durante el streaming')
            }
          }
        }
      } else {
        await sendNonStreaming()
      }
      await loadMessages()
    } catch (error) {
      if (error?.name === 'AbortError') {
        if (cancelRequestedRef.current) setToastMessage('Generación detenida por el usuario')
        await loadMessages()
      } else {
        console.error(error)
        await loadMessages()
        alert(error.message || 'Error en chat')
      }
    } finally {
      abortControllerRef.current = null
      setAssistantTyping(false)
      setSending(false)
    }
  }

  const stopGeneration = () => {
    if (abortControllerRef.current) {
      cancelRequestedRef.current = true
      abortControllerRef.current.abort()
    }
  }

  const sendRevision = async (revisionText) => {
    const prompt = `Revisa el último resultado. ${revisionText}. Mantén el tema "${topic}" y nivel ${level}.`
    await sendMessage(prompt, goal)
  }

  const applyPreset = async (preset) => {
    setGoal(preset.goal)
    setRequestText(preset.text)
  }

  const convertMessage = async (messageId, target) => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/chat/convert`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chat_message_id: messageId,
          target,
          topic,
          level
        })
      })
      if (!response.ok) throw new Error(await getApiErrorMessage(response, 'No se pudo guardar'))
      const data = await response.json()
      if (target === 'exercise') {
        setExerciseByMessage((prev) => ({ ...prev, [String(messageId)]: data.item.id }))
        if (onExerciseCreated) onExerciseCreated(data.item)
      }
      setToastMessage(target === 'exercise' ? 'Ejercicio guardado' : 'Examen guardado')
    } catch (error) {
      alert(error.message || 'Error al convertir')
    }
  }

  const resolveUrl = (path) => {
    if (!path) return ''
    if (path.startsWith('http://') || path.startsWith('https://') || path.startsWith('data:')) return path
    return `${apiBaseUrl}${path}`
  }

  const setCardFeedback = (messageId, text, tone = 'info') => {
    setCardStatus((prev) => ({ ...prev, [String(messageId)]: { text, tone } }))
  }

  const ensureExerciseForMessage = async (message) => {
    const key = String(message.id)
    const existing = exerciseByMessage[key]
    if (existing) return existing
    if (key.startsWith('temp-')) throw new Error('Espera un segundo a que termine de guardarse el mensaje')

    setCardFeedback(key, 'Guardando ejercicio base...')
    const response = await fetch(`${apiBaseUrl}/api/chat/convert`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chat_message_id: message.id,
        target: 'exercise',
        topic,
        level
      })
    })
    if (!response.ok) throw new Error(await getApiErrorMessage(response, 'No se pudo guardar el ejercicio base'))
    const data = await response.json()
    const exerciseId = data?.item?.id
    if (!exerciseId) throw new Error('El backend no devolvió el id del ejercicio')
    setExerciseByMessage((prev) => ({ ...prev, [key]: exerciseId }))
    if (onExerciseCreated) onExerciseCreated(data.item)
    return exerciseId
  }

  const exportPdfVersion = async (message, worksheetRole) => {
    const key = String(message.id)
    try {
      const itemId = await ensureExerciseForMessage(message)
      setCardFeedback(key, `Generando PDF ${worksheetRole === 'student' ? 'Alumno' : 'Profesor'}...`)
      const response = await fetch(`${apiBaseUrl}/api/library/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          item_type: 'exercise',
          item_id: itemId,
          format: 'pdf',
          options: {
            worksheet_role: worksheetRole,
            include_answers: worksheetRole === 'teacher'
          }
        })
      })
      if (!response.ok) throw new Error(await getApiErrorMessage(response, 'No se pudo generar el PDF'))
      const data = await response.json()
      const downloadUrl = resolveUrl(data.download_url)
      if (downloadUrl) window.open(downloadUrl, '_blank', 'noopener,noreferrer')
      setCardFeedback(key, `PDF ${worksheetRole === 'student' ? 'Alumno' : 'Profesor'} listo`, 'ok')
      setToastMessage(`PDF ${worksheetRole === 'student' ? 'Alumno' : 'Profesor'} generado`)
    } catch (error) {
      setCardFeedback(key, error.message || 'Error al crear PDF', 'error')
      alert(error.message || 'Error al crear PDF')
    }
  }

  const generateGameImageFromMessage = async (message) => {
    const key = String(message.id)
    try {
      const preview = parsePreviewFromMessage(message) || {}
      const prompt = [
        preview.title || `Juego de francés sobre ${topic}`,
        preview.activity_type ? `tipo ${preview.activity_type}` : '',
        `nivel ${level}`,
        'estilo ficha escolar imprimible con iconos claros'
      ]
        .filter(Boolean)
        .join(', ')
      setCardFeedback(key, 'Generando imagen del juego...')
      const response = await fetch(`${apiBaseUrl}/api/media/image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, provider: 'qwen_image' })
      })
      if (!response.ok) throw new Error(await getApiErrorMessage(response, 'No se pudo generar la imagen'))
      const data = await response.json()
      const imageUrl = data.image_url || ''
      setGeneratedImageUrl(imageUrl)
      if (imageUrl) window.open(resolveUrl(imageUrl), '_blank', 'noopener,noreferrer')
      setCardFeedback(key, 'Imagen generada y abierta en pestaña nueva', 'ok')
      setToastMessage('Imagen del juego generada')
    } catch (error) {
      setCardFeedback(key, error.message || 'Error al generar imagen', 'error')
      alert(error.message || 'Error al generar imagen')
    }
  }

  const addInstructionAudioFromMessage = async (message) => {
    const key = String(message.id)
    try {
      const preview = parsePreviewFromMessage(message) || {}
      const steps = Array.isArray(preview?.assistant_guidance?.steps) ? preview.assistant_guidance.steps : []
      const text = steps.length > 0
        ? `Consignes de classe: ${steps.join('. ')}.`
        : `Bonjour la classe. Aujourd'hui nous travaillons ${topic}. Écoutez les consignes et participez.`
      setCardFeedback(key, 'Generando audio de instrucciones...')
      const response = await fetch(`${apiBaseUrl}/api/media/audio`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, voice: 'french_teacher' })
      })
      if (!response.ok) throw new Error(await getApiErrorMessage(response, 'No se pudo generar el audio'))
      const data = await response.json()
      const audioUrl = data.audio_url || ''
      setGeneratedAudioUrl(audioUrl)
      if (audioUrl) window.open(resolveUrl(audioUrl), '_blank', 'noopener,noreferrer')
      setCardFeedback(key, 'Audio de instrucciones listo', 'ok')
      setToastMessage('Audio de instrucciones generado')
    } catch (error) {
      setCardFeedback(key, error.message || 'Error al generar audio', 'error')
      alert(error.message || 'Error al generar audio')
    }
  }

  const publishToWorkspaceFromMessage = async (message) => {
    const key = String(message.id)
    try {
      const itemId = await ensureExerciseForMessage(message)
      setCardFeedback(key, 'Publicando en Drive/Classroom...')
      const response = await fetch(`${apiBaseUrl}/api/google/workspace/publish`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          item_type: 'exercise',
          item_id: itemId,
          class_name: '6º Primaria'
        })
      })
      if (!response.ok) throw new Error(await getApiErrorMessage(response, 'No se pudo publicar en Google Workspace'))
      const data = await response.json()
      if (data?.doc_url) window.open(data.doc_url, '_blank', 'noopener,noreferrer')
      if (data?.folder_url) window.open(data.folder_url, '_blank', 'noopener,noreferrer')
      setCardFeedback(key, 'Publicado en Drive/Docs correctamente', 'ok')
      setToastMessage('Publicado en Drive/Classroom')
    } catch (error) {
      setCardFeedback(key, error.message || 'Error al publicar', 'error')
      alert(error.message || 'Error al publicar')
    }
  }

  const generateImage = async () => {
    const prompt = requestText.trim() || `Ficha escolar de francés: ${topic} (${level})`
    try {
      const response = await fetch(`${apiBaseUrl}/api/media/image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, provider: 'openai' })
      })
      if (!response.ok) throw new Error(await getApiErrorMessage(response, 'No se pudo generar imagen'))
      const data = await response.json()
      setGeneratedImageUrl(data.image_url || '')
      setToastMessage('Imagen generada')
    } catch (error) {
      alert(error.message || 'Error al generar imagen')
    }
  }

  const generateVideo = async () => {
    const prompt = requestText.trim() || `Microvideo escolar de francés: ${topic} (${level})`
    try {
      const response = await fetch(`${apiBaseUrl}/api/media/video`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, provider: 'kling' })
      })
      if (!response.ok) throw new Error(await getApiErrorMessage(response, 'No se pudo solicitar video'))
      const data = await response.json()
      setVideoJobInfo(data)
      setToastMessage(`Video: ${data.status || 'queued'}`)
    } catch (error) {
      alert(error.message || 'Error al solicitar video')
    }
  }

  const generateAudio = async () => {
    const text = requestText.trim() || `Bonjour la classe. Aujourd'hui nous travaillons ${topic}.`
    try {
      const response = await fetch(`${apiBaseUrl}/api/media/audio`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, voice: 'french_teacher' })
      })
      if (!response.ok) throw new Error(await getApiErrorMessage(response, 'No se pudo generar audio'))
      const data = await response.json()
      setGeneratedAudioUrl(data.audio_url || '')
      setToastMessage('Audio generado')
    } catch (error) {
      alert(error.message || 'Error al generar audio')
    }
  }

  const parsePreviewFromMessage = (msg) => {
    if (msg?.preview && typeof msg.preview === 'object') return msg.preview
    if (msg?.role !== 'assistant') return null
    const text = String(msg?.content || '').trim()
    if (!text.startsWith('{') && !text.startsWith('[')) return null
    try {
      const parsed = JSON.parse(text)
      return parsed && typeof parsed === 'object' ? parsed : null
    } catch {
      return null
    }
  }

  const renderAssistantPreview = (msg) => {
    const preview = parsePreviewFromMessage(msg)
    if (!preview || !Array.isArray(preview.items)) return <div className="chat-text">{msg.content}</div>

    const activity = preview.activity_type || 'actividad'
    const items = preview.items.slice(0, 8)
    const spinWheel = () => {
      const segments = Array.isArray(preview?.ui_game?.segments) ? preview.ui_game.segments : []
      if (segments.length === 0) return
      const index = Math.floor(Math.random() * segments.length)
      setGameResults((prev) => ({
        ...prev,
        [msg.id]: {
          index,
          selected: segments[index],
          at: new Date().toISOString()
        }
      }))
    }

    return (
      <div className="chat-preview-card">
        <div className="chat-preview-header">
          <strong>{preview.title || 'Vista previa de actividad'}</strong>
          <span className="chat-preview-type">{activity}</span>
        </div>
        {preview?.ui_game?.type === 'wheel' && (
          <div className="chat-wheel">
            <div className="chat-wheel-title">{preview.ui_game.title || 'Ruleta'}</div>
            <div className="chat-wheel-segments">
              {(preview.ui_game.segments || []).map((segment, idx) => (
                <span key={`seg-${msg.id}-${idx}`} className="chat-wheel-segment">
                  {idx + 1}. {segment}
                </span>
              ))}
            </div>
            <div className="chat-wheel-actions">
              <button type="button" className="btn-primary" onClick={spinWheel}>Girar ruleta</button>
              {gameResults[msg.id]?.selected && (
                <div className="chat-wheel-result">
                  Resultado: <strong>{gameResults[msg.id].selected}</strong>
                </div>
              )}
            </div>
          </div>
        )}
        <div className="chat-preview-items">
          {items.map((item, idx) => (
            <div key={`pv-${msg.id}-${idx}`} className="chat-preview-item">
              <span className="chat-preview-index">{idx + 1}.</span>
              <span>
                {item.question || item.line_with_blank || item.left || item.word || item.label || item.sentence || 'Ítem'}
              </span>
            </div>
          ))}
        </div>
        {preview?.quality && (
          <div className="chat-preview-quality">
            Calidad: {Math.round((preview.quality.score || 0) * 100)}% · {preview.quality.passed ? 'OK' : 'A mejorar'}
          </div>
        )}
        {Array.isArray(preview?.improvement_suggestions) && preview.improvement_suggestions.length > 0 && (
          <div className="chat-preview-suggestions">
            <strong>Mejoras sugeridas:</strong>
            <div className="chat-presets-grid">
              {preview.improvement_suggestions.slice(0, 5).map((suggestion) => (
                <button
                  key={`${msg.id}-${suggestion}`}
                  type="button"
                  className="btn-secondary"
                  onClick={() => sendRevision(suggestion)}
                  disabled={sending}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}
        {preview?.assistant_guidance?.steps && (
          <div className="chat-preview-guidance">
            <strong>Guía del asistente:</strong>
            <ol>
              {preview.assistant_guidance.steps.map((step, idx) => (
                <li key={`${msg.id}-guide-${idx}`}>{step}</li>
              ))}
            </ol>
          </div>
        )}
        <div className="chat-inline-actions">
          <button type="button" className="btn-primary" onClick={() => exportPdfVersion(msg, 'student')} disabled={sending}>
            Crear versión PDF Alumno
          </button>
          <button type="button" className="btn-primary" onClick={() => exportPdfVersion(msg, 'teacher')} disabled={sending}>
            Crear versión PDF Profesor
          </button>
          <button type="button" className="btn-secondary" onClick={() => generateGameImageFromMessage(msg)} disabled={sending}>
            Generar imagen del juego
          </button>
          <button type="button" className="btn-secondary" onClick={() => addInstructionAudioFromMessage(msg)} disabled={sending}>
            Añadir audio de instrucciones
          </button>
          <button type="button" className="btn-secondary" onClick={() => publishToWorkspaceFromMessage(msg)} disabled={sending}>
            Publicar en Drive/Classroom
          </button>
        </div>
        {cardStatus[String(msg.id)]?.text && (
          <div className={`chat-card-status chat-card-status-${cardStatus[String(msg.id)]?.tone || 'info'}`}>
            {cardStatus[String(msg.id)]?.text}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="exercise-generator chat-module chat-simple">
      {toastMessage && (
        <div className="chat-toast" role="status" aria-live="polite">
          <span>{toastMessage}</span>
          <button type="button" className="chat-toast-close" onClick={() => setToastMessage('')} aria-label="Cerrar aviso">×</button>
        </div>
      )}

      <h2>Asistente IA para Crear Material</h2>
      <p className="chat-simple-subtitle">1) Pides algo. 2) La IA lo crea. 3) Indicas cambios con un clic.</p>

      <div className="chat-simple-config">
        <div className="form-group">
          <label>Tema</label>
          <input value={topic} onChange={(e) => setTopic(e.target.value)} />
        </div>
        <div className="form-group">
          <label>Nivel</label>
          <select value={level} onChange={(e) => setLevel(e.target.value)}>
            <option value="A1">A1</option>
            <option value="A2">A2</option>
            <option value="B1">B1</option>
          </select>
        </div>
        <div className="form-group">
          <label>Objetivo</label>
          <select value={goal} onChange={(e) => setGoal(e.target.value)}>
            <option value="exercise_gen">Crear ejercicio</option>
            <option value="exam_gen">Crear examen</option>
            <option value="interactive_gen">Actividad interactiva</option>
            <option value="chat">Ayuda general</option>
          </select>
        </div>
      </div>

      <div className="chat-simple-composer">
        <label>Tu petición</label>
        <textarea
          value={requestText}
          onChange={(e) => setRequestText(e.target.value)}
          placeholder="Ejemplo: Crea una ficha de les couleurs con juego de parejas y corrección final."
        />
        <div className="chat-simple-actions">
          <button className="btn-primary" disabled={sending} onClick={() => sendMessage()}>
            {sending ? 'Generando...' : 'Generar'}
          </button>
          <button className="btn-secondary" disabled={!sending || !(streamAvailable && goal === 'chat')} onClick={stopGeneration}>
            Detener
          </button>
        </div>
      </div>

      <button type="button" className="btn-secondary" onClick={() => setShowAdvanced((v) => !v)}>
        {showAdvanced ? 'Ocultar opciones avanzadas' : 'Mostrar opciones avanzadas'}
      </button>
      {showAdvanced && (
        <div className="form-row">
          <div className="form-group">
            <label>Proveedor</label>
            <select value={provider} onChange={(e) => setProvider(e.target.value)}>
              {providerOptions.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label>Modelo</label>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {(modelsByProvider[provider] || [provider]).map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>
        </div>
      )}

      <button type="button" className="btn-secondary" onClick={() => setShowMoreOptions((v) => !v)}>
        {showMoreOptions ? 'Ocultar más opciones' : 'Más opciones'}
      </button>

      {showMoreOptions && (
        <>
          <div className="chat-presets">
            <h4>Plantillas rápidas</h4>
            <div className="chat-presets-grid">
              {PROMPT_PRESETS.map((preset) => (
                <button key={preset.label} type="button" className="btn-secondary" onClick={() => applyPreset(preset)}>
                  {preset.label}
                </button>
              ))}
            </div>
          </div>

          <div className="chat-revisions">
            <h4>Cambios rápidos (sin escribir)</h4>
            <div className="chat-presets-grid">
              {REVISION_ACTIONS.map((item) => (
                <button key={item} type="button" className="btn-secondary" disabled={sending} onClick={() => sendRevision(item)}>
                  {item}
                </button>
              ))}
            </div>
          </div>

          <div className="chat-revisions">
            <h4>Generación multimedia</h4>
            <div className="chat-presets-grid">
              <button className="btn-secondary" disabled={sending} onClick={generateImage}>Imagen</button>
              <button className="btn-secondary" disabled={sending} onClick={generateVideo}>Video</button>
              <button className="btn-secondary" disabled={sending} onClick={generateAudio}>Audio</button>
            </div>
          </div>
        </>
      )}

      <div className="chat-conversation">
        {messages.length === 0 && <p>No hay mensajes todavía.</p>}
        {messages.map((msg) => (
          <div key={msg.id} className={`chat-bubble ${msg.role === 'user' ? 'chat-user' : 'chat-assistant'}`}>
            <div className="chat-meta">
              <strong>{msg.role === 'user' ? 'Tú' : 'Asistente IA'}</strong>
            </div>
            {msg.role === 'assistant' ? renderAssistantPreview(msg) : <div className="chat-text">{msg.content}</div>}
            {msg.role === 'assistant' && (
              <div className="chat-actions">
                <button className="btn-primary" onClick={() => convertMessage(msg.id, 'exercise')}>
                  Guardar como ejercicio
                </button>
                <button className="btn-primary" onClick={() => convertMessage(msg.id, 'exam')}>
                  Guardar como examen
                </button>
              </div>
            )}
          </div>
        ))}
        {assistantTyping && (
          <div className="typing-indicator" aria-live="polite">
            <span className="typing-dot" />
            <span className="typing-dot" />
            <span className="typing-dot" />
            <span>La IA está escribiendo...</span>
          </div>
        )}
      </div>

      {generatedImageUrl && (
        <div className="chat-media-preview">
          <h4>Imagen generada</h4>
          <img src={generatedImageUrl} alt="Imagen generada" />
        </div>
      )}
      {videoJobInfo && (
        <div className="chat-media-preview">
          <h4>Estado de video</h4>
          <p>{videoJobInfo.status} · {videoJobInfo.provider}</p>
          {videoJobInfo.detail && <p>{videoJobInfo.detail}</p>}
          {videoJobInfo.configure_url && (
            <a href={videoJobInfo.configure_url} target="_blank" rel="noreferrer">Configurar proveedor</a>
          )}
        </div>
      )}
      {generatedAudioUrl && (
        <div className="chat-media-preview">
          <h4>Audio generado</h4>
          <audio controls src={generatedAudioUrl} />
        </div>
      )}
    </div>
  )
}

export default ChatAssistant
