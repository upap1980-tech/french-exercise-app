import { useEffect, useMemo, useState } from 'react'
import { getApiErrorMessage } from '../utils/api'

const LIBRARY_SIMPLE_MODE_KEY = 'library_simple_mode'

function Library() {
  const apiBaseUrl = import.meta.env.VITE_API_URL || ''
  const [items, setItems] = useState([])
  const [filterType, setFilterType] = useState('all')
  const [searchText, setSearchText] = useState('')
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')
  const [selectedId, setSelectedId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [editing, setEditing] = useState(false)
  const [editDraft, setEditDraft] = useState({})
  const [lastExportPath, setLastExportPath] = useState('')
  const [lastDownloadUrl, setLastDownloadUrl] = useState('')
  const [importingTemplates, setImportingTemplates] = useState(false)
  const [importDryRun, setImportDryRun] = useState(true)
  const [workspaceClassName, setWorkspaceClassName] = useState('6º Primaria')
  const [semanticQuery, setSemanticQuery] = useState('')
  const [simpleMode, setSimpleMode] = useState(() => localStorage.getItem(LIBRARY_SIMPLE_MODE_KEY) !== '0')
  const [showMoreActions, setShowMoreActions] = useState(false)

  const loadItems = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${apiBaseUrl}/api/library/items`)
      if (!res.ok) throw new Error(await getApiErrorMessage(res, 'No se pudo cargar la biblioteca'))
      const data = await res.json()
      setItems(data)
      if (!selectedId && data.length > 0) setSelectedId(`${data[0].item_type}:${data[0].id}`)
    } catch (error) {
      console.error(error)
      alert('Error al cargar biblioteca')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadItems()
  }, [])

  useEffect(() => {
    localStorage.setItem(LIBRARY_SIMPLE_MODE_KEY, simpleMode ? '1' : '0')
    if (!simpleMode) {
      setShowMoreActions(true)
    }
  }, [simpleMode])

  const filteredItems = useMemo(() => {
    return items.filter((item) => {
      if (filterType !== 'all' && item.item_type !== filterType) return false
      const haystack = JSON.stringify(item).toLowerCase()
      if (searchText.trim() && !haystack.includes(searchText.trim().toLowerCase())) return false
      if (dateFrom) {
        const from = new Date(`${dateFrom}T00:00:00`)
        const created = new Date(item.created_at || 0)
        if (created < from) return false
      }
      if (dateTo) {
        const to = new Date(`${dateTo}T23:59:59`)
        const created = new Date(item.created_at || 0)
        if (created > to) return false
      }
      return true
    })
  }, [items, filterType, searchText, dateFrom, dateTo])

  const selectedItem = useMemo(() => {
    if (!selectedId) return null
    const [type, id] = selectedId.split(':')
    return items.find((item) => item.item_type === type && String(item.id) === id)
  }, [items, selectedId])

  useEffect(() => {
    if (!selectedItem) return
    setEditDraft({
      title: selectedItem.title || '',
      topic: selectedItem.topic || '',
      description: selectedItem.description || '',
      filename: selectedItem.filename || '',
      content: selectedItem.content || {},
      exercises: selectedItem.exercises || [],
      analysis: selectedItem.analysis || null,
      total_score: selectedItem.total_score || 100
    })
  }, [selectedItem])

  const saveEdit = async () => {
    if (!selectedItem) return
    try {
      const res = await fetch(`${apiBaseUrl}/api/library/items/${selectedItem.item_type}/${selectedItem.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(editDraft)
      })
      if (!res.ok) throw new Error(await getApiErrorMessage(res, 'No se pudo guardar'))
      setEditing(false)
      await loadItems()
    } catch (error) {
      console.error(error)
      alert('Error al guardar cambios')
    }
  }

  const exportItem = async (format) => {
    if (!selectedItem) return
    try {
      const res = await fetch(`${apiBaseUrl}/api/library/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          item_type: selectedItem.item_type,
          item_id: selectedItem.id,
          format
        })
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data?.error?.message || data?.error || 'No se pudo exportar')
      setLastExportPath(data.path)
      setLastDownloadUrl(data.download_url || '')
      if (format === 'google_workspace') {
        const links = data?.workspace_links || {}
        const lines = [
          `Exportado: ${data.filename}`,
          '1) Sube el archivo a Google Drive',
          '2) Ábrelo con Google Docs',
          '3) Compártelo por Classroom'
        ]
        alert(lines.join('\n'))
        if (links.drive_upload) {
          window.open(links.drive_upload, '_blank', 'noopener,noreferrer')
        }
      } else {
        alert(`Exportado: ${data.filename}`)
      }
    } catch (error) {
      console.error(error)
      alert(error.message || 'Error al exportar')
    }
  }

  const exportLmsFormat = async (mode) => {
    if (!selectedItem) return
    const endpoint =
      mode === 'moodle'
        ? '/api/library/export/moodle-xml'
        : mode === 'h5p'
          ? '/api/library/export/h5p-json'
          : '/api/library/export/notebooklm-pack'
    try {
      const res = await fetch(`${apiBaseUrl}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          items: [{ type: selectedItem.item_type, id: selectedItem.id }],
          options: { include_answers: true }
        })
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data?.error?.message || data?.error || 'No se pudo exportar')
      setLastExportPath(data.file_path || '')
      setLastDownloadUrl(data.download_url || '')
      alert(`Exportación completada: ${data.download_name || data.file_path}`)
    } catch (error) {
      console.error(error)
      alert(error.message || 'Error en exportación LMS/Pack')
    }
  }

  const duplicateItem = async () => {
    if (!selectedItem) return
    try {
      const res = await fetch(`${apiBaseUrl}/api/library/duplicate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          item_type: selectedItem.item_type,
          item_id: selectedItem.id
        })
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data?.error?.message || data?.error || 'No se pudo duplicar')
      await loadItems()
      setSelectedId(`${data.item_type}:${data.id}`)
      alert('Item duplicado correctamente')
    } catch (error) {
      console.error(error)
      alert(error.message || 'Error al duplicar')
    }
  }

  const publishItemToWorkspace = async () => {
    if (!selectedItem) return
    try {
      const res = await fetch(`${apiBaseUrl}/api/google/workspace/publish`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          item_type: selectedItem.item_type,
          item_id: selectedItem.id,
          class_name: workspaceClassName || 'Clase'
        })
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data?.error?.message || data?.error || 'No se pudo publicar en Google Workspace')
      alert(`Publicado en Docs: ${data.doc_name}\nClase: ${data.class_name}`)
      if (data.doc_url) {
        window.open(data.doc_url, '_blank', 'noopener,noreferrer')
      }
    } catch (error) {
      console.error(error)
      alert(error.message || 'Error al publicar en Google Workspace')
    }
  }

  const semanticSearch = async () => {
    if (!semanticQuery.trim()) {
      await loadItems()
      return
    }
    try {
      setLoading(true)
      const res = await fetch(`${apiBaseUrl}/api/library/search/semantic?q=${encodeURIComponent(semanticQuery.trim())}&limit=80`)
      const data = await res.json()
      if (!res.ok) throw new Error(data?.error?.message || data?.error || 'No se pudo buscar')
      setItems(Array.isArray(data?.results) ? data.results : [])
      if (data?.results?.length > 0) {
        setSelectedId(`${data.results[0].item_type}:${data.results[0].id}`)
      }
    } catch (error) {
      console.error(error)
      alert(error.message || 'Error en búsqueda semántica')
    } finally {
      setLoading(false)
    }
  }

  const publishRecentBatchToWorkspace = async () => {
    const batch = filteredItems.slice(0, 10).map((item) => ({ item_type: item.item_type, item_id: item.id }))
    if (batch.length === 0) {
      alert('No hay items para publicar')
      return
    }
    try {
      const res = await fetch(`${apiBaseUrl}/api/google/workspace/publish-batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ items: batch, class_name: workspaceClassName || 'Clase' })
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data?.error?.message || data?.error || 'No se pudo publicar batch')
      alert(`Batch publicado: ${data.published_count}/${data.requested} en clase ${data.class_name}`)
    } catch (error) {
      console.error(error)
      alert(error.message || 'Error al publicar batch')
    }
  }

  const repairOldExercises = async () => {
    try {
      const res = await fetch(`${apiBaseUrl}/api/exercises/repair-batch`, {
        method: 'POST'
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data?.error?.message || data?.error || 'No se pudo reparar')
      await loadItems()
      alert(`Ejercicios reparados: ${data.repaired}`)
    } catch (error) {
      console.error(error)
      alert(error.message || 'Error al reparar ejercicios')
    }
  }

  const openInFinder = async () => {
    if (!lastExportPath) {
      alert('Primero exporta un archivo')
      return
    }
    try {
      const res = await fetch(`${apiBaseUrl}/api/library/open`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: lastExportPath })
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data?.error?.message || data?.error || 'No se pudo abrir en Finder')
    } catch (error) {
      console.error(error)
      alert(error.message || 'Error al abrir Finder')
    }
  }

  const downloadLastExport = async () => {
    if (!lastDownloadUrl) {
      alert('Primero exporta un archivo para obtener URL de descarga')
      return
    }
    try {
      const response = await fetch(`${apiBaseUrl}${lastDownloadUrl}`)
      if (!response.ok) {
        const errorMessage = await getApiErrorMessage(response, 'No se pudo descargar el archivo exportado')
        throw new Error(errorMessage)
      }
      const blob = await response.blob()
      const contentDisposition = response.headers.get('content-disposition') || ''
      const filenameMatch = contentDisposition.match(/filename=\"?([^\";]+)\"?/i)
      const filename = filenameMatch?.[1] || 'export.bin'

      const objectUrl = window.URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = objectUrl
      anchor.download = filename
      document.body.appendChild(anchor)
      anchor.click()
      document.body.removeChild(anchor)
      window.URL.revokeObjectURL(objectUrl)
    } catch (error) {
      console.error(error)
      alert(error.message || 'Error al descargar exportación')
    }
  }

  const importFrancais6Templates = async () => {
    setImportingTemplates(true)
    try {
      const res = await fetch(`${apiBaseUrl}/api/library/import-francais6`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dry_run: importDryRun, max_files: 500 })
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data?.error?.message || data?.error || 'No se pudo importar')
      if (!importDryRun) {
        await loadItems()
      }
      if (importDryRun) {
        alert(
          `Previsualización dry-run:\n` +
          `Detectados: ${data.discovered_count}\n` +
          `Se crearían: ${data.created_count}\n` +
          `Omitidos: ${data.skipped_count}`
        )
      } else {
        alert(`Importación completada: ${data.created_count} plantillas creadas, ${data.skipped_count} omitidas.`)
      }
    } catch (error) {
      console.error(error)
      alert(error.message || 'Error al importar materiales')
    } finally {
      setImportingTemplates(false)
    }
  }

  return (
    <div className="library-module">
      <h2>Biblioteca</h2>
      <div className="library-mode-toggle">
        <button
          type="button"
          className={`btn-secondary ${simpleMode ? 'active' : ''}`}
          onClick={() => setSimpleMode(true)}
        >
          Modo simple
        </button>
        <button
          type="button"
          className={`btn-secondary ${!simpleMode ? 'active' : ''}`}
          onClick={() => setSimpleMode(false)}
        >
          Modo completo
        </button>
      </div>
      <div className="library-toolbar library-toolbar-primary">
        <div className="library-toolbar-left">
          <select value={filterType} onChange={(e) => setFilterType(e.target.value)}>
            <option value="all">Todo</option>
            <option value="exercise">Ejercicios</option>
            <option value="exam">Exámenes</option>
            <option value="document">Documentos</option>
          </select>
          <button className="btn-primary" onClick={loadItems} disabled={loading}>
            {loading ? 'Actualizando...' : 'Actualizar biblioteca'}
          </button>
          <button className="btn-primary" onClick={repairOldExercises}>
            Reparar ejercicios antiguos
          </button>
        </div>
        <div className="library-toolbar-right">
          <label className="library-dryrun-toggle">
            <input
              type="checkbox"
              checked={importDryRun}
              onChange={(e) => setImportDryRun(e.target.checked)}
            />
            Dry-run (solo previsualizar)
          </label>
          <button className="btn-primary" onClick={importFrancais6Templates} disabled={importingTemplates}>
            {importingTemplates
              ? (importDryRun ? 'Previsualizando...' : 'Importando...')
              : (importDryRun ? 'Previsualizar importación FRANÇAIS 6º' : 'Importar materiales FRANÇAIS 6º')}
          </button>
        </div>
      </div>

      <div className="library-toolbar">
        <label>Clase Workspace:</label>
        <input
          type="text"
          value={workspaceClassName}
          onChange={(e) => setWorkspaceClassName(e.target.value)}
          placeholder="Ej: 6º Primaria A"
        />
        <button className="btn-primary" onClick={publishRecentBatchToWorkspace}>
          Publicar 10 recientes (batch)
        </button>
      </div>

      <div className="library-toolbar">
        <input
          type="text"
          value={semanticQuery}
          placeholder="Búsqueda semántica: ex. vêtements hiver diálogo"
          onChange={(e) => setSemanticQuery(e.target.value)}
        />
        <button className="btn-primary" onClick={semanticSearch} disabled={loading}>
          Buscar semántico
        </button>
        <button className="btn-primary" onClick={loadItems} disabled={loading}>
          Reset resultados
        </button>
      </div>

      <div className="library-toolbar">
        <input
          type="text"
          value={searchText}
          placeholder="Buscar por texto..."
          onChange={(e) => setSearchText(e.target.value)}
        />
        <label>Desde:</label>
        <input type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} />
        <label>Hasta:</label>
        <input type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)} />
      </div>

      <div className="library-layout">
        <div className="library-list">
          {filteredItems.map((item) => {
            const itemKey = `${item.item_type}:${item.id}`
            return (
              <button
                key={itemKey}
                className={`library-item ${selectedId === itemKey ? 'active' : ''}`}
                onClick={() => setSelectedId(itemKey)}
              >
                <div className="library-item-title">{item.display_title}</div>
                <div className="library-item-meta">{item.item_type} · ID {item.id}</div>
              </button>
            )
          })}
        </div>

        <div className="library-preview">
          {!selectedItem && <p>Selecciona un elemento para ver detalles</p>}
          {selectedItem && (
            <>
              <div className="library-actions">
                <button className="btn-primary" onClick={() => setEditing(!editing)}>
                  {editing ? 'Cancelar edición' : 'Editar'}
                </button>
                <button className="btn-primary" onClick={() => exportItem('pdf')}>Exportar PDF</button>
                <button className="btn-primary" onClick={publishItemToWorkspace}>Publicar Drive/Docs</button>
                <button className="btn-primary" onClick={downloadLastExport}>Descargar último export</button>
                {simpleMode && (
                  <button
                    type="button"
                    className="btn-secondary library-more-btn"
                    onClick={() => setShowMoreActions((prev) => !prev)}
                  >
                    {showMoreActions ? 'Ocultar opciones' : 'Más opciones'}
                  </button>
                )}
                {(!simpleMode || showMoreActions) && (
                  <>
                    <button className="btn-primary" onClick={() => exportItem('image')}>Exportar imagen</button>
                    <button className="btn-primary" onClick={() => exportItem('txt')}>Exportar texto</button>
                    <button className="btn-primary" onClick={() => exportItem('json')}>Exportar JSON</button>
                    <button className="btn-primary" onClick={() => exportItem('google_workspace')}>Exportar Google Workspace</button>
                    <button className="btn-primary" onClick={() => exportLmsFormat('moodle')}>Exportar Moodle XML</button>
                    <button className="btn-primary" onClick={() => exportLmsFormat('h5p')}>Exportar H5P JSON</button>
                    <button className="btn-primary" onClick={() => exportLmsFormat('notebooklm')}>Exportar pack NotebookLM</button>
                    <button className="btn-primary" onClick={duplicateItem}>Duplicar item</button>
                    <button className="btn-primary" onClick={openInFinder}>Abrir en Finder</button>
                  </>
                )}
              </div>

              {(selectedItem.image_url || selectedItem?.content?.image_url) && (
                <img
                  className="preview-image"
                  src={selectedItem.image_url || selectedItem?.content?.image_url}
                  alt="preview"
                />
              )}

              {!editing && (
                <pre className="library-json-preview">
                  {JSON.stringify(selectedItem, null, 2)}
                </pre>
              )}

              {editing && (
                <div className="library-edit-form">
                  {selectedItem.item_type !== 'document' && (
                    <div className="form-group">
                      <label>Título</label>
                      <input
                        value={editDraft.title || ''}
                        onChange={(e) => setEditDraft((prev) => ({ ...prev, title: e.target.value }))}
                      />
                    </div>
                  )}
                  {selectedItem.item_type === 'exercise' && (
                    <div className="form-group">
                      <label>Tema</label>
                      <input
                        value={editDraft.topic || ''}
                        onChange={(e) => setEditDraft((prev) => ({ ...prev, topic: e.target.value }))}
                      />
                    </div>
                  )}
                  {selectedItem.item_type === 'document' && (
                    <div className="form-group">
                      <label>Nombre de archivo</label>
                      <input
                        value={editDraft.filename || ''}
                        onChange={(e) => setEditDraft((prev) => ({ ...prev, filename: e.target.value }))}
                      />
                    </div>
                  )}
                  <div className="form-group">
                    <label>JSON editable (content/analysis/exercises)</label>
                    <textarea
                      rows="12"
                      value={JSON.stringify(
                        selectedItem.item_type === 'exercise'
                          ? editDraft.content
                          : selectedItem.item_type === 'exam'
                            ? editDraft.exercises
                            : editDraft.analysis,
                        null,
                        2
                      )}
                      onChange={(e) => {
                        try {
                          const parsed = JSON.parse(e.target.value)
                          if (selectedItem.item_type === 'exercise') {
                            setEditDraft((prev) => ({ ...prev, content: parsed }))
                          } else if (selectedItem.item_type === 'exam') {
                            setEditDraft((prev) => ({ ...prev, exercises: parsed }))
                          } else {
                            setEditDraft((prev) => ({ ...prev, analysis: parsed }))
                          }
                        } catch {
                          // keep text while invalid JSON; save button will persist last valid object
                        }
                      }}
                    />
                  </div>
                  <button className="btn-primary" onClick={saveEdit}>Guardar cambios</button>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default Library
