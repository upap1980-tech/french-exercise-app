# Matriz Modelo -> Endpoint/Backend -> Boton UI (P0 sin romper)

Fecha: 2026-02-28

Objetivo: integrar 4 proveedores prioritarios manteniendo compatibilidad con flujos actuales (`/api/exercises/generate`, `/api/chat`, `/api/media/image`, `/api/media/video`) y sin regresiones en UI.

## 1) Qwen 3.5 (texto principal para ejercicios/examen/chat)

- Modelo objetivo: `qwen3.5-*` (open weight; via endpoint compatible OpenAI o proveedor propio).
- Estado actual en backend:
  - Ya existe proveedor `qwen` en dispatcher:
    - `call_qwen(...)` en `backend/app.py`
    - inclusión en `generate_with_provider(...)`
  - Endpoints ya consumibles:
    - `POST /api/exercises/generate`
    - `POST /api/exercises/generate-batch`
    - `POST /api/chat`
    - `POST /api/chat/stream` (cuando proceda)
- Cambios mínimos requeridos:
  - `.env`: ajustar
    - `QWEN_API_KEY=`
    - `QWEN_BASE_URL=`
    - `QWEN_MODEL=qwen3.5-...`
- Botones/UI afectados:
  - `frontend/src/components/ExerciseGenerator.jsx`
    - selector `Modelo IA` (ya soporta modelos cloud)
  - `frontend/src/components/ChatAssistant.jsx`
    - selector avanzado `Proveedor/Modelo` (ya existe)
  - `frontend/src/components/Dashboard.jsx`
    - IA Studio/Test conectividad (`/api/ai/test` y `/api/ai/tools/*`)
- Riesgo de ruptura: bajo (usa rutas ya existentes).

## 2) Qwen-Image (imagenes educativas para fichas)

- Modelo objetivo: `Qwen-Image` (HF open model).
- Estado actual en backend:
  - Endpoint existente: `POST /api/media/image`
  - Función actual: `generate_image_asset(...)` con provider principal `openai` + fallback SVG.
- Integracion recomendada sin romper:
  - Mantener `/api/media/image` y extender provider:
    - `provider: "qwen_image"`
  - En backend agregar branch en `generate_image_asset`:
    - si `provider == qwen_image` -> inferencia local/remota
    - si falla -> fallback SVG actual
- Botones/UI afectados:
  - `ChatAssistant`: boton `Imagen` (ya existe)
  - `ExerciseGenerator`: usa `image_url` en render de actividades visuales
- Riesgo de ruptura: bajo si se mantiene fallback actual.

## 3) Wan2.1 / Wan2.6 (video educativo)

- Modelo objetivo: Wan open (recomendado estable hoy: Wan2.1).
- Estado actual en backend:
  - Endpoint existente: `POST /api/media/video`
  - Provider actual principal: `kling` (job simulado/controlado)
- Integracion recomendada sin romper:
  - Mantener `/api/media/video`.
  - Añadir provider nuevo:
    - `provider: "wan"`
  - Respuesta homogenea con shape actual:
    - `{ status, provider, job_id?, detail?, configure_url? }`
  - Si no hay runtime WAN disponible:
    - devolver `pending_manual` en lugar de 500.
- Botones/UI afectados:
  - `ChatAssistant`: boton `Video` (ya existe)
  - `Dashboard` -> IA Studio (`/api/ai/tools`) añadir tool `wan`
- Riesgo de ruptura: bajo-medio (depende de runtime de video).

## 4) GLM-4 (fallback robusto texto)

- Modelo objetivo: `glm-4-*`
- Estado actual en backend:
  - Ya existe proveedor `glm`:
    - `call_glm(...)`
    - test en `/api/ai/test`
    - IA Studio en `/api/ai/tools/*`
- Cambios minimos requeridos:
  - `.env`: validar
    - `GLM_API_KEY=`
    - `GLM_BASE_URL=`
    - `GLM_MODEL=glm-4-flash` (u otro GLM-4 habilitado)
- Botones/UI afectados:
  - mismos selectores de modelo/proveedor en Generador y Chat.
- Riesgo de ruptura: bajo.

---

## Matriz operativa resumida

| Prioridad | Modelo | Endpoint backend (mantener) | Funcion backend clave | Boton/UI | Cambio minimo |
|---|---|---|---|---|---|
| P0 | Qwen 3.5 | `/api/exercises/generate`, `/api/chat` | `call_qwen`, `generate_with_provider` | Selector Modelo IA + Chat avanzado | Ajustar `QWEN_MODEL` y key |
| P0 | GLM-4 | `/api/exercises/generate`, `/api/chat` | `call_glm`, `generate_with_provider` | Selector Modelo IA + Chat avanzado | Ajustar `GLM_MODEL` y key |
| P1 | Qwen-Image | `/api/media/image` | `generate_image_asset` | Boton `Imagen` | Añadir provider `qwen_image` |
| P1 | Wan (2.1/2.6) | `/api/media/video` | `generate_media_video` | Boton `Video` | Añadir provider `wan` |

---

## Contrato de respuesta recomendado (para no romper frontend)

### Imagen (`POST /api/media/image`)
Mantener:
```json
{
  "prompt": "...",
  "provider": "openai|qwen_image|fallback_svg",
  "image_url": "data:image/... or https://..."
}
```

### Video (`POST /api/media/video`)
Mantener:
```json
{
  "status": "queued|ready|pending_manual|error",
  "provider": "kling|wan",
  "job_id": "...",
  "detail": "...",
  "configure_url": "..."
}
```

### Generacion textual (`POST /api/exercises/generate`)
Mantener payload actual + calidad:
```json
{
  "id": 123,
  "content": {
    "activity_type": "...",
    "items": [],
    "quality": {
      "score": 0.81,
      "passed": true,
      "reasons": []
    }
  }
}
```

---

## Archivos a tocar (cuando ejecutes la integracion P1)

- Backend:
  - `backend/app.py`
    - `generate_image_asset` (branch `qwen_image`)
    - `generate_media_video` (branch `wan`)
    - `get_ai_tools_catalog` (agregar `wan`, `qwen_image`)
    - `test_provider` (checks `wan`, `qwen_image`)
  - `backend/.env.example`
    - `QWEN_IMAGE_*`, `WAN_*` placeholders
  - `backend/tests/test_api_smoke.py`
    - smoke de `/api/media/image` con `provider=qwen_image`
    - smoke de `/api/media/video` con `provider=wan`

- Frontend:
  - `frontend/src/components/ChatAssistant.jsx`
    - provider configurable para Imagen/Video (actualmente fijo openai/kling)
  - `frontend/src/components/Dashboard.jsx`
    - IA Studio: tarjetas `qwen_image` y `wan`

---

## Orden de despliegue seguro

1. Activar Qwen 3.5 + GLM-4 por `.env` (sin tocar UI).
2. Validar `/api/ai/test` en Dashboard.
3. Implementar `qwen_image` en `/api/media/image` con fallback.
4. Implementar `wan` en `/api/media/video` con `pending_manual` si no runtime.
5. Añadir smoke tests y pasar build/test.

