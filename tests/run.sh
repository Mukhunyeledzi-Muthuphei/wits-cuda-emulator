#!/bin/bash
# Runs every example and test case, then checks the expectations in tests/expected.txt.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
pass=0
fail=0

trim() { sed -E 's/^ +//; s/ +$//'; }

for f in "$ROOT"/examples/*.cu "$ROOT"/tests/cases/*.cu; do
    name="$(basename "$f" .cu)"
    (cd "$WORK" && WCU_TRACE_OUT="$WORK/$name.json" NO_COLOR=1 "$ROOT/bin/wcu" run "$f" >"$name.out" 2>"$name.err")
    status=$?
    problems=()
    want_exit=0
    checks=0

    while IFS='|' read -r prog check value; do
        prog="$(echo "$prog" | trim)"
        [ "$prog" = "$name" ] || continue
        check="$(echo "$check" | trim)"
        value="$(echo "$value" | trim)"
        checks=$((checks + 1))
        case "$check" in
            diag)   grep -qF -- "$value" "$WORK/$name.err" || problems+=("stderr lacks: $value") ;;
            output) grep -qF -- "$value" "$WORK/$name.out" || problems+=("stdout lacks: $value") ;;
            exit)   want_exit="$value" ;;
            clean)  ! grep -qE "wcu (error|warning)|error:" "$WORK/$name.err" || problems+=("expected a clean run") ;;
        esac
    done < <(grep -vE '^(#|$)' "$ROOT/tests/expected.txt")

    [ "$status" = "$want_exit" ] || problems+=("exit status $status, expected $want_exit")
    [ "$checks" -gt 0 ] || problems+=("no expectations in tests/expected.txt")

    if [ ${#problems[@]} -eq 0 ]; then
        pass=$((pass + 1))
        echo "  ok    $name"
    else
        fail=$((fail + 1))
        echo "  FAIL  $name"
        for p in "${problems[@]}"; do echo "          $p"; done
        sed 's/^/          | /' "$WORK/$name.err" | head -60
    fi
done

if command -v node >/dev/null; then
    if node "$ROOT/tests/viewer_check.js" "$ROOT/viewer/viewer.html" "$WORK"/*.json; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
    fi
else
    echo "  skip  viewer checks (node not installed)"
fi

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
