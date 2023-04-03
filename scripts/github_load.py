import base64
import json
import os
import re
import shutil
from argparse import ArgumentParser
from urllib.request import (
    HTTPBasicAuthHandler,
    HTTPPasswordMgrWithDefaultRealm,
    build_opener,
    install_opener,
    urlopen,
    urlretrieve,
)


class HTTPForcedBasicAuthHandler(HTTPBasicAuthHandler):
    """Forced basic auth.

    Instead of waiting for a 403 to then retry with the credentials,
    send the credentials if the url is handled by the password manager.
    Note: please use realm=None when calling add_password."""

    def https_request(self, req):
        url = req.get_full_url()
        user, pw = self.passwd.find_user_password(None, url)
        if pw:
            raw = "%s:%s" % (user, pw)
            auth = "Basic %s" % base64.b64encode(raw.encode("utf-8")).strip()
            req.add_unredirected_header(self.auth_header, auth)
        return req


def get_release_by_tag(repo, tag):
    release = urlopen(
        "https://api.github.com/repos/" + repo + "/releases/tags/" + tag
    ).read()
    return json.loads(release)


def get_latest_release(repo):
    release = urlopen(
        "https://api.github.com/repos/" + repo + "/releases/latest"
    ).read()
    return json.loads(release)


parser = ArgumentParser(
    prog="ProgramName",
    description="What the program does",
    epilog="Text at the bottom of help",
)

parser.add_argument("repo")
parser.add_argument("-u", "--github_user", nargs="?", const=None, type=str)
parser.add_argument("-p", "--github_password", nargs="?", const=None, type=str)
parser.add_argument("-t", "--tag", default="latest", type=str)
parser.add_argument("-d", "--destination", default=None, type=str)
parser.add_argument("-g", "--grep", default=None, type=str)
parser.add_argument("-c", "--cache_dir", default=None, type=str)

args = parser.parse_args()

if args.github_user is not None:
    passman = HTTPPasswordMgrWithDefaultRealm()
    passman.add_password(
        None, "https://api.github.com/", args.github_user, args.github_password
    )
    authhandler = HTTPForcedBasicAuthHandler(passman)
    opener = build_opener(authhandler)
    install_opener(opener)

if args.tag == "latest":
    release = get_latest_release(args.repo)
else:
    release = get_release_by_tag(args.repo, args.tag)

for item in release["assets"]:
    if args.grep is not None:
        if not re.match(args.grep, item["name"]):
            continue
    if args.destination is None:
        print(item["browser_download_url"])
        continue

    load_dir = os.path.realpath(args.destination)
    load_target = os.path.join(load_dir, item["name"])
    dir = load_dir
    target = load_target

    if args.cache_dir is not None:
        load_dir = os.path.join(args.cache_dir, args.repo, release["tag_name"])
        load_target = os.path.join(load_dir, item["name"])

    print(f"Loading {item['name']}...", end="", flush=True)
    if os.path.isfile(target):
        print("Exists")
        continue
    if os.path.isfile(load_target):
        print("Cached")
    else:
        os.makedirs(load_dir, exist_ok=True)
        urlretrieve(item["browser_download_url"], load_target)
        print("Done")

    if args.cache_dir is not None:
        os.makedirs(dir, exist_ok=True)
        shutil.copy(load_target, target)
