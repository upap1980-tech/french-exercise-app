import { useState } from 'react'

function ExerciseGenerator({ onGenerate }) {
  const [topic, setTopic] = useState('')
  const [difficulty, setDifficulty] = useState('intermediate')
  const [aiModel, setAiModel] = useState('ollama')
  const [loading, setLoading] = useState(false)
  const [generatedExercises, setGeneratedExercises] = useState([])

  const handleGenerate = async () => {
    if (!topic.trim()) {
      alert('Por favor ingresa un tema')
      return
    }

    setLoading(true)
    try {
      const response = await fetch('http://localhost:5007/api/generate-exercises', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic,
          difficulty,
          aiModel,
          quantity: 5
        })
      })

      if (response.ok) {
        const data = await response.json()
        const exercises = data.exercises || []
        setGeneratedExercises(exercises)
        onGenerate(exercises)
      }
    } catch (error) {
      console.error('Error:', error)
      alert('Error al generar ejercicios')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="exercise-generator">
      <h2>Generador de Ejercicios</h2>
      
      <div className="form-group">
        <label>Tema:</label>
        <input
          type="text"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="Ej: Verbos en presente, Vocabulario de animales"
        />
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
            <option value="ollama">Ollama 3 (Local)</option>
            <option value="perplexity">Perplexity</option>
            <option value="gemini">Gemini</option>
            <option value="openai">OpenAI</option>
          </select>
        </div>
      </div>

      <button
        onClick={handleGenerate}
        disabled={loading}
        className="btn-primary"
      >
        {loading ? 'Generando...' : 'Generar Ejercicios'}
      </button>

      {generatedExercises.length > 0 && (
        <div className="exercises-list">
          <h3>Ejercicios Generados:</h3>
          {generatedExercises.map((exercise, index) => (
            <div key={index} className="exercise-card">
              <div className="exercise-number">Ejercicio {index + 1}</div>
              <p>{exercise.question || exercise}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default ExerciseGenerator
