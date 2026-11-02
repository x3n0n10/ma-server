#!/bin/sh
set -e
# keep the signed-in account on the add-on's persistent storage; lite sees it as /data
mkdir -p /data/account /app/rootfs/data
mountpoint -q /app/rootfs/data || mount --bind /data/account /app/rootfs/data
mkdir -p /app/rootfs/data/data/com.apple.android.music/files
cd /app

opt() { jq -r --arg k "$1" '.[$k] // ""' /data/options.json; }

if [ ! -f /app/rootfs/data/data/com.apple.android.music/files/mpl_db/kvs.sqlitedb ]; then
  USERNAME=$(opt username); PASSWORD=$(opt password); CODE=$(opt code)
  if [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
    echo "Set your Apple ID and password in the add-on configuration, then start it again." >&2
    exit 1
  fi
  [ -n "$CODE" ] && echo "$CODE" > /app/rootfs/data/2fa.txt
  echo "Signing in to Apple Music..."
  ./wrapper-lite --login "$USERNAME:$PASSWORD" --code-from-file --base-dir /data
  rm -f /app/rootfs/data/2fa.txt
fi

# 127.0.0.1 on the host network: reachable by Music Assistant, not by the LAN
exec ./wrapper-lite --base-dir /data --host 127.0.0.1 --port 12340
