import { useState } from 'react'

function Dashboard({ exercises, documents }) {
  const [stats] = useState({
    totalExercises: exercises.length,
    totalDocuments: documents.length,
    aiModels: 4,
    lastUpdated: new Date().toLocaleDateString('es-ES')
  })

  return (
    <div className="dashboard">
      <h2>Panel de Control</h2>
      
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
          <div className="stat-value">{stats.lastUpdated}</div>
          <div className="stat-label">Última Actualización</div>
        </div>
      </div>

      <div className="dashboard-content">
        <div className="section">
          <h3>Modelos de IA Disponibles</h3>
          <ul className="model-list">
            <li>🔵 Ollama 3 (Local)</li>
            <li>🌐 Perplexity API (Nube)</li>
            <li>🌐 Google Gemini (Nube)</li>
            <li>🌐 OpenAI (Nube)</li>
            <li>🌐 DeepSeek (Nube)</li>
          </ul>
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
      </div>
    </div>
  )
}

export default Dashboard
