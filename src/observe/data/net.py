import requests


def session():
    s = requests.Session(); s.trust_env = False
    s.proxies.update({"http": None, "https": None}); return s
