#!/bin/bash

curl -X 'POST' \
  'http://127.0.0.1:8080/api/v1/classify' \
  -H 'Content-Type: application/json'\
  -H 'accept:application/json'\
  -d '{"text": "This is a real departure from other movies in the Rocky-verse. The absence of Stallone is felt throughout the films style, writing, direction and lack of any sentimentality. It simply does not feel like a Rocky/Creed movie, it is darker, more angry, more miserable and a lot less fun. But worst of all you just do not care about any of the characters...and that means no hairs on the back of your neck as you approach fight night."
    }'