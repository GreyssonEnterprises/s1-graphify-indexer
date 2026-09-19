from pkg.config import parse_config


def load_user(user_id):
    cfg = parse_config("users.toml")
    return {"id": user_id, "cfg": cfg}
