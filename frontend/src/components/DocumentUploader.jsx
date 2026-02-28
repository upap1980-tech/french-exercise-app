import { useState, useRef } from 'react'
import { getApiErrorMessage } from '../utils/api'

function DocumentUploader({ onUpload }) {
  const [uploadedFile, setUploadedFile] = useState(null)
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [loading, setLoading] = useState(false)
  const [analysisResults, setAnalysisResults] = useState(null)
  const [aiModel, setAiModel] = useState('llama3')
  const fileInputRef = useRef(null)
  const apiBaseUrl = import.meta.env.VITE_API_URL || ''

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      setUploadedFile(file)
    }
  }

  const handleUpload = async () => {
    if (!uploadedFile) {
      alert('Por favor selecciona un archivo')
      return
    }

    setLoading(true)
    const formData = new FormData()
    formData.append('file', uploadedFile)

    try {
      const response = await fetch(`${apiBaseUrl}/api/documents/upload`, {
        method: 'POST',
        body: formData
      })

      if (response.ok) {
        const uploaded = await response.json()
        const analyzeResponse = await fetch(`${apiBaseUrl}/api/documents/${uploaded.id}/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ai_mode: ['llama3', 'mistral'].includes(aiModel) ? 'local' : 'cloud',
            ai_model: aiModel
          })
        })
        const data = analyzeResponse.ok ? await analyzeResponse.json() : uploaded
        const newFile = {
          id: data.id,
          name: data.filename || uploadedFile.name,
          uploadDate: new Date().toLocaleDateString('es-ES'),
          analysis: data.analysis,
          image_url: data.image_url
        }
        setUploadedFiles([...uploadedFiles, newFile])
        setAnalysisResults(data.analysis)
        onUpload(newFile)
        setUploadedFile(null)
        if (fileInputRef.current) fileInputRef.current.value = ''
      } else {
        const errorMessage = await getApiErrorMessage(response, 'Error al subir el documento')
        throw new Error(errorMessage)
      }
    } catch (error) {
      console.error('Error:', error)
      alert(error.message || 'Error al subir el documento')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="document-uploader">
      <h2>Subir y Analizar Documentos</h2>
      
      <div className="upload-section">
        <div className="form-group">
          <label>Modelo IA para análisis:</label>
          <select value={aiModel} onChange={(e) => setAiModel(e.target.value)}>
            <option value="llama3">Llama 3 (Local)</option>
            <option value="mistral">Mistral (Local)</option>
            <option value="perplexity">Perplexity</option>
            <option value="gemini">Gemini</option>
            <option value="openai">OpenAI</option>
            <option value="deepseek">DeepSeek</option>
          </select>
        </div>

        <div className="file-input-wrapper">
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.jpg,.jpeg,.png,.doc,.docx"
            onChange={handleFileChange}
            id="fileInput"
            hidden
          />
          <label htmlFor="fileInput" className="file-input-label">
            📄 {uploadedFile ? uploadedFile.name : 'Selecciona un archivo'}
          </label>
        </div>

        <button
          onClick={handleUpload}
          disabled={loading || !uploadedFile}
          className="btn-primary"
        >
          {loading ? 'Subiendo...' : '📁 Subir y Analizar'}
        </button>
      </div>

      {uploadedFiles.length > 0 && (
        <div className="uploaded-files">
          <h3>Documentos Subidos:</h3>
          <div className="files-list">
            {uploadedFiles.map((file) => (
              <div key={file.id} className="file-card">
                <div className="file-info">
                  {file.image_url && (
                    <img className="preview-image" src={file.image_url} alt={`Vista previa ${file.name}`} />
                  )}
                  <div className="file-name">{file.name}</div>
                  <div className="file-date">Subido: {file.uploadDate}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {analysisResults && (
        <div className="analysis-results">
          <h3>Índice de Análisis:</h3>
          <pre className="analysis-content">{JSON.stringify(analysisResults, null, 2)}</pre>
        </div>
      )}
    </div>
  )
}

export default DocumentUploader
