"""
User management and authentication module for HalgOM Screener.
Provides password-based authentication, user storage, session persistence, and per-user data directory creation.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import datetime as dt
import hashlib
import hmac
import logging
import re
import secrets
from pathlib import Path

from screener.data import store
from screener.userfiles import load_user_file, save_user_file

log = logging.getLogger(__name__)


@dataclass
class User:
    id: str
    display_name: str
    salt: str
    password_hash: str
    created_at: str


@dataclass
class Session:
    token: str
    user_id: str
    created_at: str
    expires_at: str


def _get_users_path() -> Path:
    return store.DATA_DIR / "users.json"


def _get_sessions_path() -> Path:
    return store.DATA_DIR / "sessions.json"


def _generate_salt() -> str:
    return secrets.token_hex(16)


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    ).hex()


def _ensure_initialized() -> None:
    path = _get_users_path()
    if not path.exists():
        def init_fn(items: dict[str, User]) -> None:
            if not items:
                now_str = dt.datetime.now(dt.timezone.utc).isoformat()

                salt_ori = _generate_salt()
                hash_ori = _hash_password("halgom123", salt_ori)
                items["ori"] = User(
                    id="ori",
                    display_name="אורי",
                    salt=salt_ori,
                    password_hash=hash_ori,
                    created_at=now_str,
                )

                salt_haim = _generate_salt()
                hash_haim = _hash_password("halgom123", salt_haim)
                items["haim"] = User(
                    id="haim",
                    display_name="חיים",
                    salt=salt_haim,
                    password_hash=hash_haim,
                    created_at=now_str,
                )

        save_user_file(path, User, init_fn)


def _generate_user_id(display_name: str, existing_ids: set[str]) -> str:
    cleaned = re.sub(r"[^\w]+", "_", display_name.strip().lower()).strip("_")
    base_id = cleaned if cleaned else "user"
    candidate = base_id
    counter = 1
    while candidate in existing_ids:
        candidate = f"{base_id}_{counter}"
        counter += 1
    return candidate


def list_users() -> list[User]:
    _ensure_initialized()
    items = load_user_file(_get_users_path(), User)
    return list(items.values())


def get_user(user_id: str) -> User | None:
    _ensure_initialized()
    items = load_user_file(_get_users_path(), User)
    return items.get(user_id)


def create_user(display_name: str, password: str) -> User:
    _ensure_initialized()
    cleaned_name = display_name.strip()
    if not cleaned_name:
        raise ValueError("Display name cannot be empty")
    if not password:
        raise ValueError("Password cannot be empty")

    path = _get_users_path()
    created_user: list[User] = []

    def modify_fn(items: dict[str, User]) -> None:
        user_id = _generate_user_id(cleaned_name, set(items.keys()))
        salt = _generate_salt()
        p_hash = _hash_password(password, salt)
        created_at = dt.datetime.now(dt.timezone.utc).isoformat()

        user = User(
            id=user_id,
            display_name=cleaned_name,
            salt=salt,
            password_hash=p_hash,
            created_at=created_at,
        )
        items[user_id] = user
        created_user.append(user)

    save_user_file(path, User, modify_fn)
    user = created_user[0]
    get_user_data_dir(user.id)
    return user


def verify_password(user_id: str, password: str) -> bool:
    user = get_user(user_id)
    if user is None:
        return False
    computed_hash = _hash_password(password, user.salt)
    return hmac.compare_digest(computed_hash, user.password_hash)


def change_password(user_id: str, old_password: str, new_password: str) -> bool:
    if not verify_password(user_id, old_password):
        return False

    path = _get_users_path()
    new_salt = _generate_salt()
    new_hash = _hash_password(new_password, new_salt)

    def modify_fn(items: dict[str, User]) -> None:
        if user_id in items:
            u = items[user_id]
            items[user_id] = User(
                id=u.id,
                display_name=u.display_name,
                salt=new_salt,
                password_hash=new_hash,
                created_at=u.created_at,
            )

    save_user_file(path, User, modify_fn)
    return True


def get_user_data_dir(user_id: str) -> Path:
    user_dir = store.DATA_DIR / "users" / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def create_session(user_id: str, days: int = 30) -> str:
    token = secrets.token_urlsafe(32)
    now = dt.datetime.now(dt.timezone.utc)
    expires = now + dt.timedelta(days=days)

    path = _get_sessions_path()

    def modify_fn(items: dict[str, Session]) -> None:
        now_iso = now.isoformat()
        expired = [k for k, s in items.items() if s.expires_at < now_iso]
        for k in expired:
            items.pop(k, None)

        items[token] = Session(
            token=token,
            user_id=user_id,
            created_at=now.isoformat(),
            expires_at=expires.isoformat(),
        )

    save_user_file(path, Session, modify_fn)
    return token


def validate_session(token: str) -> str | None:
    if not token or not isinstance(token, str):
        return None
    path = _get_sessions_path()
    if not path.exists():
        return None
    sessions = load_user_file(path, Session)
    session = sessions.get(token)
    if not session:
        return None
    now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    if session.expires_at < now_iso:
        revoke_session(token)
        return None
    user = get_user(session.user_id)
    if not user:
        revoke_session(token)
        return None
    return session.user_id


def revoke_session(token: str) -> None:
    if not token:
        return
    path = _get_sessions_path()
    if not path.exists():
        return

    def modify_fn(items: dict[str, Session]) -> None:
        items.pop(token, None)

    save_user_file(path, Session, modify_fn)
