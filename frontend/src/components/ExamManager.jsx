import { useEffect, useState } from 'react'
import { getApiErrorMessage } from '../utils/api'

function ExamManager({ exercises }) {
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [totalScore, setTotalScore] = useState(100)
  const [selectedExerciseIds, setSelectedExerciseIds] = useState([])
  const [exams, setExams] = useState([])
  const [selectedExamDetail, setSelectedExamDetail] = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [loading, setLoading] = useState(false)
  const apiBaseUrl = import.meta.env.VITE_API_URL || ''

  const loadExams = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/exams`)
      if (!response.ok) return
      const data = await response.json()
      setExams(data)
    } catch (error) {
      console.error('Error loading exams:', error)
    }
  }

  useEffect(() => {
    loadExams()
  }, [])

  const toggleExercise = (exerciseId) => {
    setSelectedExerciseIds((prev) =>
      prev.includes(exerciseId) ? prev.filter((id) => id !== exerciseId) : [...prev, exerciseId]
    )
  }

  const handleCreateExam = async () => {
    if (!title.trim()) {
      alert('Ingresa un título para el examen')
      return
    }

    setLoading(true)
    try {
      const selectedExercises = exercises
        .filter((exercise) => selectedExerciseIds.includes(exercise.id))
        .map((exercise) => ({
          id: exercise.id,
          title: exercise.title,
          topic: exercise.topic,
          level: exercise.level,
          content: exercise.content
        }))

      const response = await fetch(`${apiBaseUrl}/api/exams`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title,
          description,
          exercises: selectedExercises,
          total_score: Number(totalScore) || 100
        })
      })

      if (!response.ok) {
        const errorMessage = await getApiErrorMessage(response, 'Error al crear examen')
        throw new Error(errorMessage)
      }

      setTitle('')
      setDescription('')
      setTotalScore(100)
      setSelectedExerciseIds([])
      await loadExams()
    } catch (error) {
      console.error('Error:', error)
      alert(error.message || 'Error al crear examen')
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteExam = async (examId) => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/exams/${examId}`, {
        method: 'DELETE'
      })
      if (!response.ok) throw new Error('No se pudo eliminar examen')
      await loadExams()
    } catch (error) {
      console.error('Error deleting exam:', error)
      alert('Error al eliminar examen')
    }
  }

  const handleViewExamDetail = async (examId) => {
    setDetailLoading(true)
    try {
      const response = await fetch(`${apiBaseUrl}/api/exams/${examId}`)
      if (!response.ok) {
        const errorMessage = await getApiErrorMessage(response, 'No se pudo cargar detalle del examen')
        throw new Error(errorMessage)
      }
      const data = await response.json()
      setSelectedExamDetail(data)
    } catch (error) {
      console.error('Error loading exam detail:', error)
      alert(error.message || 'Error al cargar detalle del examen')
    } finally {
      setDetailLoading(false)
    }
  }

  return (
    <div className="exercise-generator">
      <h2>Gestión de Exámenes</h2>

      <div className="form-group">
        <label>Título del examen:</label>
        <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Ej: Examen Unidad 1" />
      </div>

      <div className="form-group">
        <label>Descripción:</label>
        <textarea value={description} onChange={(e) => setDescription(e.target.value)} rows="3" />
      </div>

      <div className="form-group">
        <label>Puntuación total:</label>
        <input type="number" min="1" value={totalScore} onChange={(e) => setTotalScore(e.target.value)} />
      </div>

      <div className="form-group">
        <label>Seleccionar ejercicios disponibles:</label>
        {exercises.length === 0 && <p>No hay ejercicios en memoria. Genera ejercicios primero.</p>}
        {exercises.map((exercise) => (
          <label key={exercise.id} style={{ display: 'block', marginBottom: '0.5rem' }}>
            <input
              type="checkbox"
              checked={selectedExerciseIds.includes(exercise.id)}
              onChange={() => toggleExercise(exercise.id)}
              style={{ width: 'auto', marginRight: '0.5rem' }}
            />
            {exercise.title} ({exercise.level})
          </label>
        ))}
      </div>

      <button onClick={handleCreateExam} disabled={loading} className="btn-primary">
        {loading ? 'Creando...' : 'Crear Examen'}
      </button>

      <div className="uploaded-files">
        <h3>Exámenes guardados:</h3>
        {selectedExamDetail && (
          <div className="ia-tool-output">
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem', alignItems: 'center' }}>
              <strong>Detalle examen: {selectedExamDetail.title}</strong>
              <button className="btn-primary" onClick={() => setSelectedExamDetail(null)}>Cerrar detalle</button>
            </div>
            <pre>{JSON.stringify(selectedExamDetail, null, 2)}</pre>
          </div>
        )}
        {exams.length === 0 && <p>No hay exámenes creados aún.</p>}
        <div className="files-list">
          {exams.map((exam) => (
            <div key={exam.id} className="file-card">
              <div className="file-info">
                {exam.image_url && (
                  <img className="preview-image" src={exam.image_url} alt={`Portada ${exam.title}`} />
                )}
                <div className="file-name">{exam.title}</div>
                <div className="file-date">Puntaje total: {exam.total_score}</div>
                <div className="file-date">Ejercicios: {exam.exercises?.length || 0}</div>
              </div>
              <button className="btn-primary" onClick={() => handleViewExamDetail(exam.id)} disabled={detailLoading}>
                {detailLoading ? 'Cargando...' : 'Ver detalle'}
              </button>
              <button className="btn-primary" onClick={() => handleDeleteExam(exam.id)}>
                Eliminar
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default ExamManager
