# Roadmap Enterprise - Ejecución Técnica (P0 -> P1 -> P2)

## P0 (Core estable)
Estado: IMPLEMENTADO (backend + frontend)

- Batch de generación de ejercicios
  - Endpoint: `POST /api/exercises/generate-batch`
  - Archivo: `/Users/victormfrancisco/Desktop/PROYECTOS/french-exercise-app/backend/app.py`
- Publicación Google Workspace por item y lote
  - Endpoints: `POST /api/google/workspace/publish`, `POST /api/google/workspace/publish-batch`
  - Archivos: backend `app.py`, frontend `ExerciseGenerator.jsx`, `Library.jsx`
- Export Google Workspace (`.gworkspace.html`)
  - Endpoint: `POST /api/library/export` (`format=google_workspace`)
  - Archivo: backend `app.py`
- Health de integración Google
  - Endpoint: `GET /api/google/workspace/health`
  - Archivo: backend `app.py`
- Feature flags enterprise
  - Variable: `ENTERPRISE_FEATURES`
  - Archivos: `backend/.env.example`, `backend/.env.recommended`, `backend/app.py`

## P1 (Docente avanzado)
Estado: IMPLEMENTADO PARCIALMENTE

- Búsqueda semántica en Biblioteca
  - Endpoint: `GET /api/library/search/semantic`
  - UI: input + botón en Biblioteca
  - Archivos: backend `app.py`, frontend `Library.jsx`
- Interactividad de sesión
  - Endpoints: `POST /api/interactive/session`, `POST /api/interactive/submit`, `POST /api/interactive/score`
  - Archivo: backend `app.py`
- Motor creativo extendido con imagen
  - Tipos: `image_choice`, `label_image`, `scene_story` + `magic_mix`
  - Archivos: backend `app.py`, frontend `ExerciseGenerator.jsx`, `App.css`

## P2 (Operación enterprise)
Estado: IMPLEMENTADO PARCIALMENTE

- Métricas operativas
  - Endpoint: `GET /api/ops/metrics`
  - Archivo: backend `app.py`
- Hardening básico
  - Logging estructurado, request_id, manejo homogéneo de errores
  - Archivo: backend `app.py`
- Backup/restore con rotación
  - Endpoints: `POST /api/backups/export`, `POST /api/backups/restore-latest`
  - Archivo: backend `app.py`

## Validación recomendada

1. `cd /Users/victormfrancisco/Desktop/PROYECTOS/french-exercise-app/backend && ./venv/bin/python -m unittest tests/test_api_smoke.py`
2. `cd /Users/victormfrancisco/Desktop/PROYECTOS/french-exercise-app/frontend && npm run build`
3. Health checks:
   - `GET /api/health`
   - `GET /api/google/workspace/health`
   - `GET /api/ops/metrics`

## Siguiente tramo para cierre 100% enterprise

- Auth/RBAC real (admin/profesor/alumno) con permisos por clase.
- Cola de jobs (Celery/RQ) para generación pesada y multimedia.
- Integración Google Classroom completa (crear tareas + sync calificaciones).
- Observabilidad avanzada (trazas, alertas, panel de costes por proveedor IA).
