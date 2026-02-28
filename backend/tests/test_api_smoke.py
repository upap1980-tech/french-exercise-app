import io
import tempfile
import unittest
import json
from pathlib import Path

import app as backend_app
from app import app, db


class ApiSmokeTest(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        with app.app_context():
            db.create_all()
        backend_app.call_ollama = lambda prompt, model='llama3': '{"question":"Pregunta de prueba","correct_answer":"x"}'

    def tearDown(self):
        with app.app_context():
            db.session.remove()
            db.engine.dispose()

    def test_health(self):
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body['status'], 'ok')

        keys_health = self.client.get('/api/health/ai-keys')
        self.assertEqual(keys_health.status_code, 200)
        keys_body = keys_health.get_json()
        self.assertIn('summary', keys_body)
        self.assertIn('providers', keys_body)

    def test_ai_test_endpoint(self):
        response = self.client.get('/api/ai/test')
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIn('tested_at', body)
        self.assertIn('results', body)
        for provider in ['llama3', 'mistral', 'perplexity', 'openai', 'gemini', 'deepseek', 'qwen_image', 'wan', 'sdxl', 'paddleocr']:
            self.assertIn(provider, body['results'])
            self.assertIn('ok', body['results'][provider])

    def test_svg_conversion_for_pdf_images(self):
        svg_data_uri = backend_app.generate_illustration_data_uri('les couleurs', 'exercise')
        image_bytes = backend_app.extract_image_bytes_for_pdf(svg_data_uri)
        if backend_app.cairosvg is None:
            self.assertIsNone(image_bytes)
            return
        self.assertIsNotNone(image_bytes)
        # PNG header
        self.assertTrue(image_bytes.startswith(b'\x89PNG'))

    def test_backup_export_endpoint(self):
        response = self.client.post('/api/backups/export')
        self.assertEqual(response.status_code, 201)
        body = response.get_json()
        self.assertIn('sqlite_backup', body)
        self.assertIn('json_backup', body)
        self.assertIn('counts', body)

    def test_library_items_and_export(self):
        create = self.client.post(
            '/api/exercises/generate',
            json={
                'topic': 'biblioteca',
                'level': 'A1',
                'exercise_type': 'fill_blank',
                'ai_mode': 'local',
                'ai_model': 'llama3',
            },
        )
        self.assertEqual(create.status_code, 201)
        exercise = create.get_json()

        items_res = self.client.get('/api/library/items')
        self.assertEqual(items_res.status_code, 200)
        items = items_res.get_json()
        self.assertTrue(any(i['id'] == exercise['id'] and i['item_type'] == 'exercise' for i in items))

        export_res = self.client.post(
            '/api/library/export',
            json={'item_type': 'exercise', 'item_id': exercise['id'], 'format': 'json'},
        )
        self.assertEqual(export_res.status_code, 201)
        export_data = export_res.get_json()
        self.assertIn('path', export_data)
        self.assertIn('download_url', export_data)
        download = self.client.get(export_data['download_url'])
        self.assertEqual(download.status_code, 200)
        self.assertIn('attachment', download.headers.get('Content-Disposition', ''))
        download.get_data()
        download.close()

        pdf_export = self.client.post(
            '/api/library/export',
            json={'item_type': 'exercise', 'item_id': exercise['id'], 'format': 'pdf'},
        )
        self.assertEqual(pdf_export.status_code, 201)
        pdf_data = pdf_export.get_json()
        self.assertIn('download_url', pdf_data)
        pdf_download = self.client.get(pdf_data['download_url'])
        self.assertEqual(pdf_download.status_code, 200)
        self.assertEqual(pdf_download.mimetype, 'application/pdf')
        self.assertTrue(pdf_download.get_data().startswith(b'%PDF'))
        pdf_download.close()

        pdf_student_export = self.client.post(
            '/api/library/export',
            json={
                'item_type': 'exercise',
                'item_id': exercise['id'],
                'format': 'pdf',
                'options': {'worksheet_role': 'student', 'include_answers': False}
            },
        )
        self.assertEqual(pdf_student_export.status_code, 201)
        pdf_student_data = pdf_student_export.get_json()
        pdf_student_download = self.client.get(pdf_student_data['download_url'])
        self.assertEqual(pdf_student_download.status_code, 200)
        self.assertEqual(pdf_student_download.mimetype, 'application/pdf')
        self.assertTrue(pdf_student_download.get_data().startswith(b'%PDF'))
        pdf_student_download.close()

        pdf_teacher_export = self.client.post(
            '/api/library/export',
            json={
                'item_type': 'exercise',
                'item_id': exercise['id'],
                'format': 'pdf',
                'options': {'worksheet_role': 'teacher', 'include_answers': True}
            },
        )
        self.assertEqual(pdf_teacher_export.status_code, 201)
        pdf_teacher_data = pdf_teacher_export.get_json()
        pdf_teacher_download = self.client.get(pdf_teacher_data['download_url'])
        self.assertEqual(pdf_teacher_download.status_code, 200)
        self.assertEqual(pdf_teacher_download.mimetype, 'application/pdf')
        self.assertTrue(pdf_teacher_download.get_data().startswith(b'%PDF'))
        pdf_teacher_download.close()

        moodle_export = self.client.post(
            '/api/library/export/moodle-xml',
            json={'items': [{'type': 'exercise', 'id': exercise['id']}], 'options': {'include_answers': True}},
        )
        self.assertEqual(moodle_export.status_code, 201)

        h5p_export = self.client.post(
            '/api/library/export/h5p-json',
            json={'items': [{'type': 'exercise', 'id': exercise['id']}], 'options': {'include_answers': True}},
        )
        self.assertEqual(h5p_export.status_code, 201)

        notebook_pack = self.client.post(
            '/api/library/export/notebooklm-pack',
            json={'items': [{'type': 'exercise', 'id': exercise['id']}]},
        )
        self.assertEqual(notebook_pack.status_code, 201)

        duplicate_res = self.client.post(
            '/api/library/duplicate',
            json={'item_type': 'exercise', 'item_id': exercise['id']},
        )
        self.assertEqual(duplicate_res.status_code, 201)
        duplicate = duplicate_res.get_json()
        self.assertIn('id', duplicate)

        repair_res = self.client.post('/api/library/repair-exercises')
        self.assertEqual(repair_res.status_code, 200)
        repair_data = repair_res.get_json()
        self.assertIn('repaired', repair_data)
        self.assertEqual(repair_res.headers.get('X-Replacement-Endpoint'), '/api/exercises/repair-batch')

        repair_batch_res = self.client.post('/api/exercises/repair-batch')
        self.assertEqual(repair_batch_res.status_code, 200)
        self.assertIn('repaired', repair_batch_res.get_json())

        self.client.delete(f"/api/exercises/{exercise['id']}")
        self.client.delete(f"/api/exercises/{duplicate['id']}")

    def test_library_import_francais6_templates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / 'U.6 DANS MON ARMOIRE').mkdir(parents=True, exist_ok=True)
            (root / 'U.6 DANS MON ARMOIRE' / 'VOC LES VETEMENTS.pdf').write_bytes(b'%PDF-1.4 fake')
            (root / 'U.2 LE CORPS' / 'ECRIS LES PARTIES DU CORPS.odt').parent.mkdir(parents=True, exist_ok=True)
            (root / 'U.2 LE CORPS' / 'ECRIS LES PARTIES DU CORPS.odt').write_text('fake odt', encoding='utf-8')

            import_res = self.client.post(
                '/api/library/import-francais6',
                json={'import_root': str(root), 'dry_run': False},
            )
            self.assertEqual(import_res.status_code, 201)
            payload = import_res.get_json()
            self.assertGreaterEqual(payload.get('created_count', 0), 2)

            items = self.client.get('/api/library/items').get_json()
            imported = []
            for i in items:
                if i.get('item_type') != 'exercise':
                    continue
                source_path = (((i.get('content') or {}).get('import_metadata') or {}).get('source_path'))
                if isinstance(source_path, str) and source_path.startswith(str(root)):
                    imported.append(i)
            self.assertGreaterEqual(len(imported), 2)
            for row in imported:
                self.client.delete(f"/api/exercises/{row['id']}")

    def test_compliance_and_analytics(self):
        compliance = self.client.get('/api/compliance/status')
        self.assertEqual(compliance.status_code, 200)
        self.assertIn('mode', compliance.get_json())

        preview = self.client.post('/api/compliance/anonymize-preview', json={'text': 'Email test@test.com'})
        self.assertEqual(preview.status_code, 200)
        self.assertIn('sanitized', preview.get_json())

        # audit log should contain at least the anonymize preview event
        audit = self.client.get('/api/compliance/audit-log')
        self.assertEqual(audit.status_code, 200)
        audit_body = audit.get_json()
        if isinstance(audit_body, dict) and 'entries' in audit_body:
            audit_entries = audit_body['entries']
        else:
            audit_entries = audit_body
        self.assertIsInstance(audit_entries, list)
        self.assertTrue(any(e.get('action') == 'compliance.anonymize_preview' for e in audit_entries))

        # filtering by action or date should work
        filtered = self.client.get('/api/compliance/audit-log', query_string={'action': 'anonymize'})
        self.assertEqual(filtered.status_code, 200)
        filtered_body = filtered.get_json()
        if isinstance(filtered_body, dict) and 'entries' in filtered_body:
            filtered_entries = filtered_body['entries']
        else:
            filtered_entries = filtered_body
        self.assertTrue(len(filtered_entries) <= len(audit_entries))

        # CSV export should work and contain header row
        csv_res = self.client.get('/api/compliance/audit-log', query_string={'action': 'anonymize', 'format': 'csv'})
        self.assertEqual(csv_res.status_code, 200)
        ctype = csv_res.headers.get('Content-Type') or ''
        self.assertTrue(ctype.startswith('text/csv'))

        # pagination should limit results and return metadata
        page = self.client.get('/api/compliance/audit-log', query_string={'limit': '1', 'offset': '0'})
        self.assertEqual(page.status_code, 200)
        body = page.get_json()
        self.assertIn('total', body)
        self.assertIn('entries', body)
        self.assertLessEqual(len(body['entries']), 1)

        # JSON export returns a file
        json_res = self.client.get('/api/compliance/audit-log', query_string={'action': 'anonymize', 'format': 'json'})
        self.assertEqual(json_res.status_code, 200)
        self.assertTrue(json_res.headers.get('Content-Type', '').startswith('application/json'))
        json_body = json.loads(json_res.get_data(as_text=True))
        self.assertIsInstance(json_body, list)
        text = csv_res.get_data(as_text=True)
        self.assertTrue(text.startswith('timestamp,action,detail'))

        templates = self.client.get('/api/exercises/templates')
        self.assertEqual(templates.status_code, 200)
        self.assertIn('themes', templates.get_json())
        self.assertIn('template_engine', templates.get_json())

        analytics = self.client.get('/api/analytics/learning')
        self.assertEqual(analytics.status_code, 200)
        self.assertIn('totals', analytics.get_json())

    def test_chat_and_convert(self):
        chat = self.client.post(
            '/api/chat',
            json={
                'message': 'Crea una actividad corta sobre les vêtements',
                'task_type': 'exercise_gen',
                'provider': 'llama3',
                'model': 'llama3',
                'context': {'topic': 'les vêtements', 'level': 'A2'}
            },
        )
        self.assertEqual(chat.status_code, 201)
        chat_body = chat.get_json()
        self.assertIn('message', chat_body)
        self.assertIn('preview', chat_body)
        assistant_message = chat_body['message']
        self.assertEqual(assistant_message['role'], 'assistant')

        history = self.client.get('/api/chat/messages?limit=10')
        self.assertEqual(history.status_code, 200)
        self.assertTrue(len(history.get_json()) >= 2)

        convert_ex = self.client.post(
            '/api/chat/convert',
            json={'chat_message_id': assistant_message['id'], 'target': 'exercise', 'topic': 'les vêtements', 'level': 'A2'},
        )
        self.assertEqual(convert_ex.status_code, 201)
        ex_item = convert_ex.get_json()['item']
        self.assertIn('id', ex_item)

        convert_exam = self.client.post(
            '/api/chat/convert',
            json={'chat_message_id': assistant_message['id'], 'target': 'exam'},
        )
        self.assertEqual(convert_exam.status_code, 201)
        exam_item = convert_exam.get_json()['item']
        self.assertIn('id', exam_item)

        self.client.delete(f"/api/exercises/{ex_item['id']}")
        self.client.delete(f"/api/exams/{exam_item['id']}")

        stream = self.client.post(
            '/api/chat/stream',
            json={
                'message': 'Streaming de prueba para generar actividades',
                'task_type': 'chat',
                'provider': 'llama3',
                'model': 'llama3',
                'context': {'topic': 'les couleurs', 'level': 'A2'}
            },
        )
        self.assertEqual(stream.status_code, 200)
        payload_text = stream.get_data(as_text=True)
        self.assertIn('event: done', payload_text)

        image = self.client.post('/api/media/image', json={'prompt': 'une classe de français'})
        self.assertEqual(image.status_code, 201)
        self.assertIn('image_url', image.get_json())
        qwen_image = self.client.post('/api/media/image', json={'prompt': 'les vêtements', 'provider': 'qwen_image'})
        self.assertEqual(qwen_image.status_code, 201)
        self.assertIn('image_url', qwen_image.get_json())

        video = self.client.post('/api/media/video', json={'prompt': 'micro leçon de vocabulaire'})
        self.assertIn(video.status_code, (201, 202))
        wan_video = self.client.post('/api/media/video', json={'prompt': 'micro leçon wan', 'provider': 'wan'})
        self.assertEqual(wan_video.status_code, 202)
        self.assertEqual(wan_video.get_json().get('provider'), 'wan')

        audio = self.client.post('/api/media/audio', json={'text': 'Bonjour les élèves'})
        self.assertEqual(audio.status_code, 201)
        self.assertIn('audio_url', audio.get_json())

        game_chat = self.client.post(
            '/api/chat',
            json={
                'message': 'Crea una ruleta para practicar colores y ropa',
                'task_type': 'exercise_gen',
                'provider': 'llama3',
                'model': 'llama3',
                'context': {'topic': 'les vêtements et les couleurs', 'level': 'A2'}
            },
        )
        self.assertEqual(game_chat.status_code, 201)
        game_body = game_chat.get_json()
        self.assertIn('preview', game_body)
        self.assertIn('ui_game', game_body['preview'])
        self.assertEqual(game_body['preview']['ui_game'].get('type'), 'wheel')

    def test_generate_exercise(self):
        response = self.client.post(
            '/api/exercises/generate',
            json={
                'topic': 'animales',
                'level': 'A1',
                'exercise_type': 'fill_blank',
                'ai_mode': 'local',
            },
        )
        self.assertEqual(response.status_code, 201)
        body = response.get_json()
        self.assertIn('id', body)
        self.assertIn('content', body)
        self.assertIn('quality', body['content'])
        delete = self.client.delete(f"/api/exercises/{body['id']}")
        self.assertEqual(delete.status_code, 200)

    def test_quality_evaluate_batch(self):
        created = self.client.post(
            '/api/exercises/generate',
            json={
                'topic': 'les couleurs',
                'level': 'A2',
                'exercise_type': 'magic_mix',
                'ai_mode': 'local',
            },
        )
        self.assertEqual(created.status_code, 201)
        ex = created.get_json()
        response = self.client.post('/api/exercises/quality/evaluate-batch', json={'exercise_ids': [ex['id']]})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload['evaluated'], 1)
        self.assertEqual(len(payload['reports']), 1)
        self.assertIn('quality', payload['reports'][0])
        self.client.delete(f"/api/exercises/{ex['id']}")

    def test_enterprise_endpoints(self):
        batch = self.client.post(
            '/api/exercises/generate-batch',
            json={
                'reject_low_quality': False,
                'items': [
                    {'topic': 'les vêtements', 'level': 'A2', 'exercise_type': 'image_choice', 'ai_mode': 'local', 'ai_model': 'llama3'},
                    {'topic': 'au téléphone', 'level': 'A2', 'exercise_type': 'scene_story', 'ai_mode': 'local', 'ai_model': 'llama3'}
                ]
            }
        )
        self.assertEqual(batch.status_code, 201)
        batch_body = batch.get_json()
        self.assertEqual(batch_body['created_count'], 2)
        created = batch_body['created']

        search = self.client.get('/api/library/search/semantic?q=vêtements%20hiver&limit=10')
        self.assertEqual(search.status_code, 200)
        self.assertIn('results', search.get_json())

        ops = self.client.get('/api/ops/metrics')
        self.assertEqual(ops.status_code, 200)
        self.assertIn('totals', ops.get_json())

        session = self.client.post('/api/interactive/session', json={'exercise_id': created[0]['id'], 'student_id': 'smoke-student'})
        self.assertEqual(session.status_code, 201)
        sess_body = session.get_json()
        self.assertIn('session_id', sess_body)

        submit = self.client.post('/api/interactive/submit', json={
            'session_id': sess_body['session_id'],
            'answers': [{'selected': 'a', 'expected': 'a'}, {'selected': 'b', 'expected': 'c'}]
        })
        self.assertEqual(submit.status_code, 200)
        self.assertIn('percent', submit.get_json())

        publish_batch = self.client.post('/api/google/workspace/publish-batch', json={
            'class_name': '6º Primaria A',
            'items': [{'item_type': 'exercise', 'item_id': created[0]['id']}]
        })
        self.assertEqual(publish_batch.status_code, 201)
        self.assertIn('published_count', publish_batch.get_json())

        for row in created:
            self.client.delete(f"/api/exercises/{row['id']}")

    def test_exam_crud(self):
        create = self.client.post(
            '/api/exams',
            json={
                'title': 'Examen Smoke',
                'description': 'Prueba rápida',
                'exercises': [],
                'total_score': 100,
            },
        )
        self.assertEqual(create.status_code, 201)
        exam = create.get_json()

        list_response = self.client.get('/api/exams')
        self.assertEqual(list_response.status_code, 200)
        exams = list_response.get_json()
        self.assertTrue(any(item['id'] == exam['id'] for item in exams))

        get_by_id = self.client.get(f"/api/exams/{exam['id']}")
        self.assertEqual(get_by_id.status_code, 200)
        self.assertEqual(get_by_id.get_json()['id'], exam['id'])

        delete = self.client.delete(f"/api/exams/{exam['id']}")
        self.assertEqual(delete.status_code, 200)

    def test_document_upload_and_analyze(self):
        upload = self.client.post(
            '/api/documents/upload',
            data={'file': (io.BytesIO(b'bonjour'), 'smoke.txt')},
            content_type='multipart/form-data',
        )
        self.assertEqual(upload.status_code, 201)
        document = upload.get_json()

        analyze = self.client.post(
            f"/api/documents/{document['id']}/analyze",
            json={'ai_mode': 'local'},
        )
        self.assertEqual(analyze.status_code, 200)
        analyzed_doc = analyze.get_json()
        self.assertEqual(analyzed_doc['id'], document['id'])
        with app.app_context():
            doc = db.session.get(backend_app.Document, document['id'])
            if doc:
                db.session.delete(doc)
                db.session.commit()


if __name__ == '__main__':
    unittest.main()
