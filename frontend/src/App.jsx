import { useState, useRef } from 'react'
import Dashboard from './components/Dashboard'
import ExerciseGenerator from './components/ExerciseGenerator'
import DocumentUploader from './components/DocumentUploader'

function App() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [exercises, setExercises] = useState([])
  const [documents, setDocuments] = useState([])

  const handleGenerateExercises = (newExercises) => {
    setExercises([...exercises, ...newExercises])
  }

  const handleUploadDocument = (doc) => {
    setDocuments([...documents, doc])
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <h1>Generador de Ejercicios de Francés</h1>
          <p>Para Primaria - IA Local y en la Nube</p>
        </div>
      </header>

      <nav className="app-nav">
        <button
          className={`nav-button ${activeTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('dashboard')}
        >
          Panel de Control
        </button>
        <button
          className={`nav-button ${activeTab === 'generator' ? 'active' : ''}`}
          onClick={() => setActiveTab('generator')}
        >
          Generar Ejercicios
        </button>
        <button
          className={`nav-button ${activeTab === 'uploader' ? 'active' : ''}`}
          onClick={() => setActiveTab('uploader')}
        >
          Subir Documentos
        </button>
      </nav>

      <main className="app-main">
        {activeTab === 'dashboard' && (
          <Dashboard exercises={exercises} documents={documents} />
        )}
        {activeTab === 'generator' && (
          <ExerciseGenerator onGenerate={handleGenerateExercises} />
        )}
        {activeTab === 'uploader' && (
          <DocumentUploader onUpload={handleUploadDocument} />
        )}
      </main>
    </div>
  )
}

export default App
