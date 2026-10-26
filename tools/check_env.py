"""
Live Brain - Environment Variables Status Checker
Loads .env and reports the status of required API keys and configuration values.
"""

import sys
from pathlib import Path
from dotenv import dotenv_values, find_dotenv

# Primary required keys for $0 default pipeline (Deepgram + Gemini + ElevenLabs + Simli)
PRIMARY_REQUIRED_KEYS = [
    "DEEPGRAM_API_KEY",
    "GEMINI_API_KEY",
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_VOICE_ID",
    "SIMLI_API_KEY",
    "SIMLI_FACE_ID",
]

# Secondary / Optional keys
SECONDARY_OPTIONAL_KEYS = [
    ("LLM_PROVIDER", "Default: 'gemini'"),
    ("GEMINI_MODEL", "Default: 'gemini-3.5-flash-lite'"),
    ("DAILY_API_KEY", "Optional: Daily WebRTC room transport"),
]


def check_env():
    # Look for .env in current working dir, project root, or live_brain_files/
    env_path = find_dotenv(usecwd=True)
    if not env_path:
        root_env = Path(".env")
        live_brain_env = Path(".env")
        if root_env.exists():
            env_path = str(root_env.resolve())
        elif live_brain_env.exists():
            env_path = str(live_brain_env.resolve())

    print("=" * 65)
    print(" Live Brain -- Environment Configuration Status")
    print("=" * 65)

    if not env_path or not Path(env_path).exists():
        print("[-] .env file: NOT FOUND")
        print("-" * 65)
        env_dict = {}
    else:
        print(f"[+] .env file: {env_path}")
        print("-" * 65)
        env_dict = dotenv_values(env_path)

    has_missing_required = False

    print("\n[PRIMARY REQUIRED KEYS -- $0 Free Tier Live Brain]")
    header = f"{'KEY':<25} | {'STATUS':<10} | {'NOTE'}"
    print(header)
    print("-" * 65)

    for key in PRIMARY_REQUIRED_KEYS:
        if key not in env_dict:
            status = "MISSING"
            note = "Not found in .env"
            has_missing_required = True
        else:
            val = env_dict[key]
            if val is None or str(val).strip() == "":
                status = "EMPTY"
                note = "Key present but blank"
                has_missing_required = True
            else:
                status = "SET"
                note = f"Configured (len: {len(str(val))})"

        print(f"{key:<25} | {status:<10} | {note}")

    print("\n[SECONDARY / OPTIONAL KEYS]")
    print(header)
    print("-" * 65)

    for key, desc in SECONDARY_OPTIONAL_KEYS:
        if key not in env_dict:
            status = "NOT SET"
            note = f"{desc} (unset)"
        else:
            val = env_dict[key]
            if val is None or str(val).strip() == "":
                status = "EMPTY"
                note = f"{desc} (blank)"
            else:
                status = "SET"
                note = f"{desc} (len: {len(str(val))})"

        print(f"{key:<25} | {status:<10} | {note}")

    print("=" * 65)
    if has_missing_required:
        print("[-] STATUS: Some required keys are MISSING or EMPTY.")
        return False
    else:
        print("[+] STATUS: All primary required keys are SET. Ready for smoke test.")
        return True


if __name__ == "__main__":
    ready = check_env()
    sys.exit(0 if ready else 1)
