"""The per-user / per-key binding saves on /settings/api-keys check profile
names against the running cfg. A profile delete (/settings/overrides/state)
and a rename keep the outgoing name in cfg until their hot-apply, so a binding
save inside that window used to pass and land after the in-use guard or the
cascade, leaving the binding naming a profile the apply then removed (its
layer and locks dropped, fail-open). The binding routes now wait for the
profile writer and then meet the normal unknown-profile 400."""

import asyncio
import threading

from fastapi import HTTPException

from faster_whisper_backend.admin import api_keys_routes, overrides_routes
from faster_whisper_backend.auth import api_keys_store
from faster_whisper_backend.settings import config_store
from tests.conftest import bearer, fake_request

PERMS = "/settings/api-keys/api/users"
OV = "/settings/overrides"
_OTHER = {"BEAM_SIZE": 8}


def _setup(client, make_user_key):
    _, raw = make_user_key("admin", is_admin=True)
    h = bearer(raw)
    r = client.post(f"{OV}/state", headers=h, json={"OVERRIDE_PROFILES": {
        "clinic-de": {"DEFAULT_LANGUAGE": "de", "locks": ["DEFAULT_LANGUAGE"]},
        "other": _OTHER}})
    assert r.status_code == 200, r.text
    uid, _ = make_user_key("alice", is_admin=False)
    kid = client.get(f"{PERMS}/{uid}/keys", headers=h).json()["keys"][0]["id"]
    return uid, kid


async def _bind_during(writer, entered, release, bind):
    """Start `writer`, run the sync `bind` in a thread once `entered` fires,
    give it a second to land inside the writer's window, then let the writer
    finish. Returns (writer result, bind result) with exceptions as values."""
    task = asyncio.create_task(writer)
    while not entered.is_set():
        await asyncio.sleep(0.01)
    binding = asyncio.create_task(asyncio.to_thread(bind))
    await asyncio.wait({binding}, timeout=1.0)
    release.set()
    return await asyncio.gather(task, binding, return_exceptions=True)


def test_key_binding_save_waits_for_an_in_flight_rename(client, make_user_key, monkeypatch):
    uid, kid = _setup(client, make_user_key)
    entered, release = threading.Event(), threading.Event()
    real_cascade = api_keys_store.rename_profile_refs

    def _cascade_then_hold(old, new):
        n = real_cascade(old, new)
        entered.set()       # cascade committed; cfg still aliases `old`
        release.wait(5)
        return n

    monkeypatch.setattr(api_keys_store, "rename_profile_refs", _cascade_then_hold)
    renamed, bound = asyncio.run(_bind_during(
        overrides_routes.rename_profile(
            overrides_routes._RenameProfileIn(old="clinic-de", new="clinic-deutsch"),
            fake_request()),
        entered, release,
        lambda: api_keys_routes.patch_key_config_api(
            uid, kid, api_keys_routes.ConfigBindingIn(profiles=["clinic-de"]))))
    assert renamed.status_code == 200
    assert isinstance(bound, HTTPException) and bound.status_code == 400
    assert "clinic-de" not in (api_keys_store.get_key_config(kid) or {}).get("profiles", [])


def test_user_binding_save_waits_for_an_in_flight_delete(client, make_user_key, monkeypatch):
    uid, _ = _setup(client, make_user_key)
    entered, release = threading.Event(), threading.Event()
    real_save = config_store.save_overrides

    def _hold_then_save(payload):
        entered.set()       # past the in-use guard; cfg still has the profile
        release.wait(5)
        return real_save(payload)

    monkeypatch.setattr(config_store, "save_overrides", _hold_then_save)
    deleted, bound = asyncio.run(_bind_during(
        overrides_routes.post_state({"OVERRIDE_PROFILES": {"other": _OTHER}},
                                    fake_request()),
        entered, release,
        lambda: api_keys_routes.patch_user_permissions_api(
            uid, api_keys_routes.PatchPermissionsIn(
                config=api_keys_routes.ConfigBindingIn(profiles=["clinic-de"])))))
    assert deleted.status_code == 200
    assert isinstance(bound, HTTPException) and bound.status_code == 400
    assert "clinic-de" not in (api_keys_store.get_user_config(uid) or {}).get("profiles", [])
