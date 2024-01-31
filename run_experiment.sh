#!/bin/bash

echo "Make sure you have WANDB_API_KEY set in .env"
source ".env"

vlmrm train "$(cat config.yaml)"
