import unittest
import json

from app import app, db


class AssistantApiTest(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        with app.app_context():
            db.create_all()

    def tearDown(self):
        with app.app_context():
            db.session.remove()
            db.engine.dispose()

    def test_create_default_exercise(self):
        res = self.client.post('/api/assistant/create-exercise', json={})
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertTrue(body.get('ok'))
        self.assertIn('generated', body)
        self.assertIsInstance(body['generated'], list)
        self.assertGreaterEqual(len(body['generated']), 1)
        self.assertIn('suggestions', body)

    def test_create_multiple_specific_type(self):
        payload = {'type': 'multiple_choice', 'count': 3, 'topics': ['animales', 'objetos'], 'level': 'A2'}
        res = self.client.post('/api/assistant/create-exercise', json=payload)
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertTrue(body.get('ok'))
        generated = body.get('generated')
        self.assertIsInstance(generated, list)
        self.assertEqual(len(generated), 3)
        for item in generated:
            self.assertIn('id', item)
            self.assertIn('type', item)
            # type may be honored or substituted; ensure keys exist
            self.assertIn('prompt', item)


if __name__ == '__main__':
    unittest.main()
