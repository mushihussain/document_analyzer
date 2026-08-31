#!/bin/bash
set -e
echo "== login =="
LOGIN=$(curl -s -X POST https://floorsmasters.co.uk/docs/api/auth/login -k \
  -H 'Content-Type: application/json' \
  -d '{"username":"claude_test_user","password":"TestPass123!"}')
echo "$LOGIN"
TOKEN=$(python3 -c "import sys,json; print(json.load(sys.stdin)['token'])" <<< "$LOGIN")

echo "== /me =="
curl -s https://floorsmasters.co.uk/docs/api/auth/me -k -H "Authorization: Bearer $TOKEN"
echo

echo "== chat (expect 409, no docs indexed for this fresh user) =="
curl -s -X POST https://floorsmasters.co.uk/docs/api/chat -k \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"question":"hello","conversation_id":null}'
echo
