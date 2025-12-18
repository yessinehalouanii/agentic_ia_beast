from urllib.parse import urljoin
import requests

def login_and_get_token(base_url: str, email: str, password: str) -> str:
    login_url = urljoin(base_url.rstrip("/") + "/", "api/v1/auth/login")
    payloads = [{"email": email, "password": password},
                {"username": email, "password": password}]
    last_resp = None
    for body in payloads:
        try:
            resp = requests.post(login_url, json=body, timeout=30)
            last_resp = resp
            if resp.status_code == 200:
                data = resp.json()
                token = data.get("access_token") or data.get("token")
                if not token:
                    for k in ["jwt","id_token","bearer","session_token"]:
                        if isinstance(data.get(k), str) and data[k]:
                            token = data[k]
                            break
                if not token:
                    raise RuntimeError("Login succeeded but no token-like field was found.")
                return token
        except requests.RequestException as e:
            raise RuntimeError(f"Login request failed: {e}")
    if last_resp is not None:
        raise RuntimeError(f"Login failed: {last_resp.status_code} {last_resp.text}")
    raise RuntimeError("Login failed: no response")
