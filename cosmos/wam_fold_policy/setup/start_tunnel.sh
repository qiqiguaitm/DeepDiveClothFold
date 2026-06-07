#!/usr/bin/env bash
# forward tunnel: this container's localhost:29333 -> gf's localhost:29290 (aurora-slim proxy)
exec ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=20 -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes -p 55555 -L 127.0.0.1:29333:localhost:29290 -N tim@14.103.44.161
