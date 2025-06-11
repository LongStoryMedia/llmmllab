#!/bin/bash

cd "$(dirname "$0")" || exit 1

cat .env.prod >.env

# Source the .env file to load environment variables
if [[ -f .env ]]; then
  echo "Loading environment variables from .env.prod file"
  set -a # Automatically export all variables
  source .env
  set +a # Turn off auto-export
fi

npm run build
# npm run test
# npm run lint
ssh root@longstorymedia.com "for i in \$(ls /var/www/ai.longstorymedia.com); do rm -rf /var/www/ai.longstorymedia.com/\$i; done" || true
rsync -avzru --delete dist/ root@longstorymedia.com:/var/www/ai.longstorymedia.com || true
ssh root@longstorymedia.com "touch /etc/nginx/sites-available/ai.longstorymedia.com" || true
rsync -avzu --delete nginx.conf root@longstorymedia.com:/etc/nginx/sites-available/ai.longstorymedia.com || true
ssh root@longstorymedia.com "ln -s /etc/nginx/sites-available/ai.longstorymedia.com /etc/nginx/sites-enabled/ai.longstorymedia.com" || true
ssh root@longstorymedia.com "chown -R www-data:www-data /var/www/ai.longstorymedia.com" || true
ssh root@longstorymedia.com "systemctl restart nginx" || true

cat .env.local >.env
# Source the .env file to load environment variables
if [[ -f .env ]]; then
  echo "Loading environment variables from .env.local file"
  set -a # Automatically export all variables
  source .env
  set +a # Turn off auto-export
fi

cd || exit 1
