import { useEffect, useState } from 'react'
import Dashboard from './components/Dashboard'
import ExerciseGenerator from './components/ExerciseGenerator'
import DocumentUploader from './components/DocumentUploader'
import ExamManager from './components/ExamManager'
import Library from './components/Library'
import ChatAssistant from './components/ChatAssistant'
import AuditLog from './components/AuditLog'
import TeacherAssistant from './components/TeacherAssistant'

function App() {
  const THEME_KEY = 'french_app_theme'
  const initialTabFromPath = (() => {
    try {
      const p = (typeof window !== 'undefined' && window.location && window.location.pathname) ? window.location.pathname : ''
      if (p.startsWith('/chat')) return 'chat'
      if (p.startsWith('/library')) return 'library'
      if (p.startsWith('/exams')) return 'exams'
      if (p.startsWith('/audit')) return 'audit'
      if (p.startsWith('/teacher-assistant')) return 'teacher-assistant'
      return 'chat'
    } catch {
      return 'chat'
    }
  })()
  const [activeTab, setActiveTab] = useState(initialTabFromPath)
  const [exercises, setExercises] = useState([])
  const [documents, setDocuments] = useState([])
  const [theme, setTheme] = useState(() => localStorage.getItem(THEME_KEY) || 'enterprise')
  const [backendOnline, setBackendOnline] = useState(null)

  useEffect(() => {
    document.body.setAttribute('data-theme', theme)
    localStorage.setItem(THEME_KEY, theme)
  }, [theme])

  const handleGenerateExercises = (newExercises) => {
    setExercises((prev) => [...prev, ...newExercises])
  }

  const handleUploadDocument = (doc) => {
    setDocuments((prev) => [...prev, doc])
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <h1>Generador de Ejercicios de Francés</h1>
          <p>Para Primaria - IA Local y en la Nube</p>
        </div>
        <div className={`backend-status-indicator ${backendOnline === true ? 'online' : backendOnline === false ? 'offline' : 'unknown'}`}>
          Backend: {backendOnline === true ? 'Online' : backendOnline === false ? 'Offline' : 'Verificando'}
        </div>
        <div className="theme-toggle">
          <button
            className={`theme-toggle-btn ${theme === 'academic' ? 'active' : ''}`}
            onClick={() => setTheme('academic')}
          >
            Académico
          </button>
          <button
            className={`theme-toggle-btn ${theme === 'enterprise' ? 'active' : ''}`}
            onClick={() => setTheme('enterprise')}
          >
            Enterprise
          </button>
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
        <button
          className={`nav-button ${activeTab === 'exams' ? 'active' : ''}`}
          onClick={() => setActiveTab('exams')}
        >
          Exámenes
        </button>
        <button
          className={`nav-button ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => { setActiveTab('chat'); if (typeof window !== 'undefined' && window.history && window.history.pushState) window.history.pushState({}, '', '/chat') }}
        >
          Asistente Fácil
        </button>
        <button
          className={`nav-button ${activeTab === 'library' ? 'active' : ''}`}
          onClick={() => setActiveTab('library')}
        >
          Biblioteca
        </button>
        <button
          className={`nav-button ${activeTab === 'audit' ? 'active' : ''}`}
          onClick={() => {
            setActiveTab('audit')
            if (typeof window !== 'undefined' && window.history && window.history.pushState) window.history.pushState({}, '', '/audit')
          }}
        >
          Auditoría
        </button>
        <button
          className={`nav-button ${activeTab === 'teacher-assistant' ? 'active' : ''}`}
          onClick={() => {
            setActiveTab('teacher-assistant')
            if (typeof window !== 'undefined' && window.history && window.history.pushState) window.history.pushState({}, '', '/teacher-assistant')
          }}
        >
          Asistente de Docentes
        </button>
      </nav>

      <main className="app-main">
        {activeTab === 'dashboard' && (
          <Dashboard onBackendStatusChange={setBackendOnline} />
        )}
        {activeTab === 'generator' && (
          <ExerciseGenerator onGenerate={handleGenerateExercises} />
        )}
        {activeTab === 'uploader' && (
          <DocumentUploader onUpload={handleUploadDocument} />
        )}
        {activeTab === 'exams' && (
          <ExamManager exercises={exercises} />
        )}
        {activeTab === 'library' && (
          <Library />
        )}
        {activeTab === 'chat' && (
          <ChatAssistant
            onExerciseCreated={(exercise) => setExercises((prev) => [exercise, ...prev])}
          />
        )}
        {activeTab === 'audit' && (
          <AuditLog />
        )}
        {activeTab === 'teacher-assistant' && (
          <TeacherAssistant onExerciseGenerated={(exercise) => setExercises((prev) => [exercise, ...prev])} />
        )}
      </main>
    </div>
  )
}

export default App
