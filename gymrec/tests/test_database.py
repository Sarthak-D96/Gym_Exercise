from django.test import SimpleTestCase

from gymrec.database import database_from_url


class DatabaseUrlTests(SimpleTestCase):
    def test_neon_pooled_url(self):
        db = database_from_url("postgresql://app_user:p%40ss@ep-cool-1234-pooler.eu-central-1.aws.neon.tech/neondb"
                               "?sslmode=require&channel_binding=require")
        self.assertEqual((db["NAME"], db["USER"], db["PASSWORD"], db["PORT"]), ("neondb", "app_user", "p@ss", "5432"))
        self.assertEqual(db["HOST"], "ep-cool-1234-pooler.eu-central-1.aws.neon.tech")
        self.assertEqual(db["OPTIONS"], {"sslmode": "require", "channel_binding": "require"})
        self.assertTrue(db["DISABLE_SERVER_SIDE_CURSORS"])
        self.assertTrue(db["CONN_HEALTH_CHECKS"])

    def test_direct_url_keeps_server_side_cursors(self):
        db = database_from_url("postgres://u:p@ep-cool-1234.eu-central-1.aws.neon.tech:5433/gym?sslmode=require")
        self.assertNotIn("DISABLE_SERVER_SIDE_CURSORS", db)
        self.assertEqual(db["PORT"], "5433")

    def test_rejects_other_schemes(self):
        with self.assertRaises(ValueError):
            database_from_url("mysql://u:p@h/db")
