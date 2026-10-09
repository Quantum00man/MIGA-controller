import asyncio
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request

import archive_main


def dashboard_request(origin=None, fetch_site=None, client='192.168.1.20'):
    headers = [(b'host', b'archive.example:8765')]
    if origin is not None:
        headers.append((b'origin', origin.encode()))
    if fetch_site is not None:
        headers.append((b'sec-fetch-site', fetch_site.encode()))
    return Request({'type': 'http', 'scheme': 'http', 'path': '/',
                    'root_path': '', 'query_string': b'', 'headers': headers,
                    'client': (client, 12345), 'server': ('archive.example', 8765)})


class ArchiveRemoteUpdateTests(unittest.TestCase):
    def test_remote_and_local_dashboard_requests_are_allowed(self):
        for client in ('192.168.1.20', '127.0.0.1', '::1'):
            for origin in (None, 'http://archive.example:8765'):
                archive_main.require_same_origin_update(
                    dashboard_request(origin, 'same-origin', client))

    def test_cross_origin_requests_are_rejected(self):
        for origin, site in [('http://other.example:8765', None),
                             ('http://archive.example:9999', None),
                             ('https://archive.example:8765', None),
                             ('null', None), (None, 'cross-site'), (None, 'same-site')]:
            with self.subTest(origin=origin, site=site):
                with self.assertRaises(HTTPException) as error:
                    archive_main.require_same_origin_update(dashboard_request(origin, site))
                self.assertEqual(error.exception.status_code, 403)

    def test_remote_fetch_apply_and_reload_reach_existing_operations(self):
        request = dashboard_request('http://archive.example:8765', 'same-origin')
        payload = archive_main.ArchiveUpdateRequest(branch='main')
        with patch.object(archive_main.updater, 'run', return_value={'ok': True}) as update:
            for operation in ('fetch', 'apply'):
                result = asyncio.run(archive_main.archive_update(operation, payload, request))
                self.assertEqual(result, {'ok': True})
                update.assert_called_with('main', operation == 'apply', None)
        with patch.object(archive_main.reload_controller, 'schedule', return_value={'id': 'test'}) as reload:
            self.assertEqual(asyncio.run(archive_main.reload_server(request)), {'id': 'test'})
            reload.assert_called_once_with()
        with patch.object(archive_main.configuration, 'load', return_value={}), \
                patch.object(archive_main.configuration, '_save') as save:
            result = asyncio.run(archive_main.reload_preference(
                archive_main.ReloadPreference(enabled=True), request))
            self.assertEqual(result, {'auto_reload_after_update': True})
            save.assert_called_once_with({'auto_reload_after_update': True})

    def test_cross_origin_update_and_reload_do_not_run(self):
        request = dashboard_request('http://other.example')
        with patch.object(archive_main.updater, 'run') as update, \
                patch.object(archive_main.reload_controller, 'schedule') as reload, \
                patch.object(archive_main.configuration, '_save') as save:
            operations = [
                archive_main.archive_update('apply', archive_main.ArchiveUpdateRequest(branch='main'), request),
                archive_main.reload_server(request),
                archive_main.reload_preference(archive_main.ReloadPreference(enabled=True), request),
            ]
            for operation in operations:
                with self.assertRaises(HTTPException):
                    asyncio.run(operation)
            update.assert_not_called()
            reload.assert_not_called()
            save.assert_not_called()
