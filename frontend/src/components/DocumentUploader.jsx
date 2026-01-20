import { useState, useRef } from 'react'

function DocumentUploader({ onUpload }) {
  const [uploadedFile, setUploadedFile] = useState(null)
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [loading, setLoading] = useState(false)
  const [analysisResults, setAnalysisResults] = useState(null)
  const fileInputRef = useRef(null)

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
      const response = await fetch('http://localhost:5007/api/upload-document', {
        method: 'POST',
        body: formData
      })

      if (response.ok) {
        const data = await response.json()
        const newFile = {
          id: Date.now(),
          name: uploadedFile.name,
          uploadDate: new Date().toLocaleDateString('es-ES'),
          analysis: data.analysis
        }
        setUploadedFiles([...uploadedFiles, newFile])
        setAnalysisResults(data.analysis)
        onUpload(newFile)
        setUploadedFile(null)
        if (fileInputRef.current) fileInputRef.current.value = ''
      }
    } catch (error) {
      console.error('Error:', error)
      alert('Error al subir el documento')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="document-uploader">
      <h2>Subir y Analizar Documentos</h2>
      
      <div className="upload-section">
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
          <div className="analysis-content">{analysisResults}</div>
        </div>
      )}
    </div>
  )
}

export default DocumentUploader
