# Matriz de Paridad Backend ↔ Frontend

Fecha de auditoría: 2026-02-28

## Resumen

- Endpoints backend auditados: **57**
- `implementado`: **57**
- `parcial`: **0**
- `pendiente`: **0**
- Endpoints consumidos en frontend sin backend: **0**

## Matriz endpoint por endpoint

| Método | Endpoint backend | Estado | Reflejo frontend |
|---|---|---|---|
| GET | `/api/health` | implementado | `frontend/src/components/Dashboard.jsx` |
| GET | `/api/health/ai-keys` | implementado | `frontend/src/components/Dashboard.jsx` |
| GET | `/api/exercises` | implementado | `frontend/src/components/Dashboard.jsx`, `frontend/src/components/ExerciseGenerator.jsx` |
| POST | `/api/exercises/generate` | implementado | `frontend/src/components/ExerciseGenerator.jsx` |
| POST | `/api/exercises/generate-batch` | implementado | `frontend/src/components/ExerciseGenerator.jsx` |
| GET | `/api/exercises/<int:id>` | implementado | `frontend/src/components/ExerciseGenerator.jsx` |
| DELETE | `/api/exercises/<int:id>` | implementado | `frontend/src/components/ExerciseGenerator.jsx` |
| GET | `/api/exams` | implementado | `frontend/src/components/Dashboard.jsx`, `frontend/src/components/ExamManager.jsx` |
| POST | `/api/exams` | implementado | `frontend/src/components/ExamManager.jsx` |
| GET | `/api/exams/<int:id>` | implementado | `frontend/src/components/ExamManager.jsx` |
| DELETE | `/api/exams/<int:id>` | implementado | `frontend/src/components/ExamManager.jsx` |
| GET | `/api/documents` | implementado | `frontend/src/components/Dashboard.jsx` |
| POST | `/api/documents/upload` | implementado | `frontend/src/components/DocumentUploader.jsx` |
| POST | `/api/documents/<int:id>/analyze` | implementado | `frontend/src/components/DocumentUploader.jsx` |
| GET | `/api/ai/models` | implementado | `frontend/src/components/Dashboard.jsx`, `frontend/src/components/ExerciseGenerator.jsx`, `frontend/src/components/ChatAssistant.jsx` |
| GET | `/api/ai/tools` | implementado | `frontend/src/components/Dashboard.jsx` |
| POST | `/api/ai/tools/<tool_id>/test` | implementado | `frontend/src/components/Dashboard.jsx` |
| POST | `/api/ai/tools/<tool_id>/sample` | implementado | `frontend/src/components/Dashboard.jsx` |
| GET | `/api/ai/tools/<tool_id>/diagnostic` | implementado | `frontend/src/components/Dashboard.jsx` |
| POST | `/api/ai/repair/gemini-deepseek` | implementado | `frontend/src/components/Dashboard.jsx` |
| POST | `/api/media/image` | implementado | `frontend/src/components/ChatAssistant.jsx` |
| POST | `/api/media/video` | implementado | `frontend/src/components/ChatAssistant.jsx` |
| POST | `/api/interactive/score` | implementado | `frontend/src/components/ExerciseGenerator.jsx` |
| POST | `/api/interactive/session` | implementado | `frontend/src/components/ExerciseGenerator.jsx` |
| POST | `/api/interactive/submit` | implementado | `frontend/src/components/ExerciseGenerator.jsx` |
| GET | `/api/ai/test` | implementado | `frontend/src/components/Dashboard.jsx` |
| POST | `/api/ai/providers/<provider>/refine` | implementado | `frontend/src/components/Dashboard.jsx` |
| GET | `/api/chat/messages` | implementado | `frontend/src/components/ChatAssistant.jsx` |
| POST | `/api/chat` | implementado | `frontend/src/components/ChatAssistant.jsx` |
| POST | `/api/chat/stream` | implementado | `frontend/src/components/ChatAssistant.jsx` |
| POST | `/api/chat/convert` | implementado | `frontend/src/components/ChatAssistant.jsx` |
| POST | `/api/backups/export` | implementado | `frontend/src/components/Dashboard.jsx` |
| POST | `/api/backups/restore-latest` | implementado | `frontend/src/components/Dashboard.jsx` |
| GET | `/api/library/items` | implementado | `frontend/src/components/Library.jsx` |
| GET | `/api/library/search/semantic` | implementado | `frontend/src/components/Library.jsx` |
| POST | `/api/library/repair-exercises` | implementado | Endpoint legacy compatibilizado; recomendado `POST /api/exercises/repair-batch` |
| POST | `/api/library/import-francais6` | implementado | `frontend/src/components/Library.jsx` |
| PUT | `/api/library/items/<item_type>/<int:item_id>` | implementado | `frontend/src/components/Library.jsx` |
| POST | `/api/library/export` | implementado | `frontend/src/components/Library.jsx`, `frontend/src/components/ExerciseGenerator.jsx` |
| GET | `/api/google/workspace/health` | implementado | `frontend/src/components/Dashboard.jsx` |
| POST | `/api/google/workspace/publish` | implementado | `frontend/src/components/Library.jsx`, `frontend/src/components/ExerciseGenerator.jsx` |
| POST | `/api/google/workspace/publish-batch` | implementado | `frontend/src/components/Library.jsx` |
| POST | `/api/library/duplicate` | implementado | `frontend/src/components/Library.jsx` |
| POST | `/api/library/export/moodle-xml` | implementado | `frontend/src/components/Library.jsx` |
| POST | `/api/library/export/h5p-json` | implementado | `frontend/src/components/Library.jsx` |
| POST | `/api/library/export/notebooklm-pack` | implementado | `frontend/src/components/Library.jsx` |
| GET | `/api/library/export/download/<token>` | implementado | `frontend/src/components/Library.jsx` (validación de error + descarga segura) |
| POST | `/api/library/open` | implementado | `frontend/src/components/Library.jsx` |
| GET | `/api/compliance/status` | implementado | `frontend/src/components/Dashboard.jsx` |
| POST | `/api/compliance/anonymize-preview` | implementado | `frontend/src/components/Dashboard.jsx` |
| GET | `/api/compliance/audit-log` | implementado | `frontend/src/components/Dashboard.jsx`, `frontend/src/components/AuditLog.jsx` (filtros `from`, `to`, `action`, paginación `limit`/`offset`, formatos `csv`, `json`) |
| GET | `/api/exercises/templates` | implementado | `frontend/src/components/ExerciseGenerator.jsx` |
| POST | `/api/exercises/repair-batch` | implementado | `frontend/src/components/Library.jsx` |
| GET | `/api/enterprise/features` | implementado | `frontend/src/components/Dashboard.jsx` |
| GET | `/api/analytics/learning` | implementado | `frontend/src/components/Dashboard.jsx` |
| GET | `/api/ops/metrics` | implementado | `frontend/src/components/Dashboard.jsx` |
| POST | `/api/assistant/create-exercise` | implementado | `frontend/src/components/TeacherAssistant.jsx` |

## Correcciones aplicadas por archivo

1. `frontend/src/components/ExerciseGenerator.jsx`
- Acción UI para borrar ejercicio (`DELETE /api/exercises/:id`) desde historial guardado (confirmación + refresh de lista).

2. `frontend/src/components/ExamManager.jsx`
- Vista detalle de examen (`GET /api/exams/:id`) integrada en panel de detalle.

3. `backend/app.py`
- `POST /api/library/repair-exercises` marcado como legacy con cabeceras deprecadas y alias explícito a lógica común de `POST /api/exercises/repair-batch`.

4. `frontend/src/components/Library.jsx`
- Descarga de exportación robusta: validación previa de endpoint, control de errores y descarga por `blob` con nombre de archivo.

5. `backend/tests/test_api_smoke.py`
- Pruebas de contrato añadidas para endpoints críticos de UI:
  - `/api/exercises/<id>` DELETE
  - `/api/exams/<id>` GET
  - `/api/library/export/download/<token>` GET
  - `/api/library/repair-exercises` compatibilidad/alias
  - limpieza de recursos de test para reducir warnings (`tearDown`)

## Observación de calidad

La paridad endpoint↔UI queda cerrada al 100% para este backend Flask actual.
