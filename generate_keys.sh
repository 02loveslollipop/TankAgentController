#!/bin/bash

# Script to generate and update JWT keys for services using Heroku CLI
# Update this script as more services are added

# Services to update
SERVICES=("auth-service" "backend" "rest-api-service")  # Add more as needed

# JWT Algorithm (set via env or default)
JWT_ALGORITHM=${JWT_ALGORITHM:-HS256}

if [ "$JWT_ALGORITHM" == "HS256" ]; then
    # Generate a random secret for HS256
    JWT_SECRET=$(openssl rand -hex 32)
    echo "Generated JWT_SECRET: $JWT_SECRET"

    # Update Heroku config for each service
    for SERVICE in "${SERVICES[@]}"; do
        heroku config:set JWT_ALGORITHM=$JWT_ALGORITHM JWT_SECRET=$JWT_SECRET --app $SERVICE
        echo "Updated $SERVICE with HS256 secret"
    done
else
    # For RSA-based algorithms (e.g., RS256)
    # Generate private key
    openssl genrsa -out jwt_private.pem 2048
    # Extract public key
    openssl rsa -in jwt_private.pem -pubout -out jwt_public.pem

    # Read keys
    JWT_KEY=$(cat jwt_private.pem | base64 -w 0)
    JWT_CERTIFICATE=$(cat jwt_public.pem | base64 -w 0)

    echo "Generated JWT_KEY and JWT_CERTIFICATE"

    # Update Heroku config for each service
    for SERVICE in "${SERVICES[@]}"; do
        heroku config:set JWT_ALGORITHM=$JWT_ALGORITHM JWT_KEY=$JWT_KEY JWT_CERTIFICATE=$JWT_CERTIFICATE --app $SERVICE
        echo "Updated $SERVICE with RSA keys"
    done

    # Clean up local files
    rm jwt_private.pem jwt_public.pem
fi

echo "Key generation and update complete. Update this script when adding new services."