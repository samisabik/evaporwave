import argparse
import logging

import uvicorn

parser = argparse.ArgumentParser(prog="evap")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
uvicorn.run("evap.app:app", host=args.host, port=args.port)
