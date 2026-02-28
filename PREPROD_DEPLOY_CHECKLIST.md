# Pre-Producción Checklist

Fecha de referencia: 2026-02-10

## 1. Entorno y secretos
- [ ] Verificar `backend/.env` con claves válidas (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `PERPLEXITY_API_KEY`).
- [ ] Confirmar `DATABASE_URL` apuntando al archivo SQLite correcto.
- [ ] Definir `LOG_LEVEL=INFO` (o `WARNING` en producción).
- [ ] Confirmar `CANARIAS_COMPLIANCE_MODE=true`.
- [ ] Definir `ALLOWED_ORIGINS` solo con dominios permitidos del frontend.
- [ ] Definir `DEBUG=false` en pre-producción/producción.

## 2. Arranque y puertos
- [ ] Backend levanta en `5012`: `./start_backend.sh`.
- [ ] Frontend levanta en `5191`: `./start_frontend.sh`.
- [ ] Full stack con navegador: `./start_full_stack.command`.
- [ ] Endpoint de salud responde `200`: `GET /api/health`.

## 3. Validación funcional mínima
- [ ] Generar 1 ejercicio desde UI y verificar guardado en Biblioteca.
- [ ] Ejecutar `POST /api/ai/test` y revisar estado por proveedor en Dashboard.
- [ ] Exportar desde Biblioteca: JSON, PDF, imagen, Moodle XML, H5P JSON y NotebookLM pack.
- [ ] Probar “Reparar ejercicios antiguos” sin errores.
- [ ] Verificar Dashboard con auto-refresh (5/15/30/60s).

## 4. Backups y recuperación
- [ ] Ejecutar backup manual: `POST /api/backups/export`.
- [ ] Confirmar backup diario scheduler activo (o log de scheduler deshabilitado si falta APScheduler).
- [ ] Verificar rotación: se conservan solo últimos 30 backups.
- [ ] Simular restore manual: `POST /api/backups/restore-latest`.

## 5. Calidad técnica
- [ ] Tests backend: `cd backend && ./venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v`.
- [ ] Build frontend: `cd frontend && npm run build`.
- [ ] Revisar logs JSON para requests y errores (`request.start`, `request.end`, `request.http_error`, `request.unhandled_exception`).
- [ ] Confirmar cabecera `X-Request-Id` presente en respuestas API.
- [ ] Confirmar rotación de logs en `backend/logs/app.log` (5 MB x 5 archivos).
- [ ] Validar rate-limit con respuesta `429` en endpoints sensibles bajo carga.

## 6. Observabilidad y operación
- [ ] Monitorear `backend` por errores de proveedores cloud (`provider.*.error`).
- [ ] Verificar que errores API retornan formato homogéneo:
  - `ok: false`
  - `error.code`
  - `error.message`
  - `request_id`
- [ ] Documentar contacto de soporte y procedimiento de recuperación rápida.
