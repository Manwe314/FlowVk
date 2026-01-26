#!/usr/bin/env bash


DEFAULT_VULKAN_SETUP="${HOME}/vulkan/1.4.321.1/setup-env.sh"

_is_sourced=0
if [ -n "${ZSH_EVAL_CONTEXT:-}" ]; then
  case $ZSH_EVAL_CONTEXT in *:file) _is_sourced=1;; esac
elif [ -n "${BASH_SOURCE:-}" ] && [ "${BASH_SOURCE[0]}" != "$0" ]; then
  _is_sourced=1
fi

_log() { printf "[vulkan-env] %s\n" "$*"; }
_err() { printf "[vulkan-env] ERROR: %s\n" "$*" >&2; }

_die() {
  _err "$*"
  if [ "$_is_sourced" -eq 1 ]; then return 1; else exit 1; fi
}

# ---------- helpers ----------
_exists_file() { [ -n "$1" ] && [ -f "$1" ]; }
_exists_dir()  { [ -n "$1" ] && [ -d "$1" ]; }

_have_glslc()  { command -v glslc >/dev/null 2>&1; }

_valid_sdk_dir() {
  _exists_dir "$1" && [ -d "$1/bin" ] && [ -d "$1/include" ]
}

_already_ok() {
  if [ -n "${VULKAN_SDK:-}" ] && _valid_sdk_dir "$VULKAN_SDK"; then
    _log "Detected existing VULKAN_SDK=${VULKAN_SDK}"
    return 0
  fi
  if _have_glslc; then
    _log "Detected glslc at $(command -v glslc); env already usable."
    return 0
  fi
  return 1
}

_find_setup_auto() {
  candidates=""

  for base in "${HOME}/vulkan" "${HOME}/sdk" ; do
    [ -d "$base" ] || continue
    for f in "$base"/*/setup-env.sh; do
      [ -f "$f" ] && candidates="${candidates}
$f"
    done
  done

  for base in /opt /usr/local/vulkan /usr/local/sdk ; do
    [ -d "$base" ] || continue
    for f in "$base"/*/setup-env.sh; do
      [ -f "$f" ] && candidates="${candidates}
$f"
    done
  done

  if [ -n "${VULKAN_SDK:-}" ] && _exists_dir "$VULKAN_SDK"; then
    [ -f "${VULKAN_SDK}/setup-env.sh" ] && candidates="${candidates}
${VULKAN_SDK}/setup-env.sh"
  fi

  if [ -n "$candidates" ]; then
    printf "%s\n" "$candidates" | sed '/^$/d' | sort -V | tail -n1
  fi
}

_source_setup() {
  local setup="$1"
  if ! _exists_file "$setup"; then
    _die "setup script not found: $setup"
  fi
  _log "Sourcing: $setup"
  . "$setup"

  if [ -n "${VULKAN_SDK:-}" ]; then
    _log "VULKAN_SDK=${VULKAN_SDK}"
  fi
  if _have_glslc; then
    _log "glslc: $(command -v glslc)"
  else
    _log "WARNING: glslc still not found on PATH."
  fi
}

if _already_ok; then
  if [ "$_is_sourced" -eq 1 ]; then
    return 0
  else
    exit 0
  fi
fi

if [ $# -gt 0 ] && [ -n "$1" ]; then
  _source_setup "$1"
  if [ "$_is_sourced" -eq 1 ]; then return 0; else exit 0; fi
fi

if [ -n "${VULKAN_SETUP:-}" ]; then
  _source_setup "$VULKAN_SETUP"
  if [ "$_is_sourced" -eq 1 ]; then return 0; else exit 0; fi
fi

auto_setup="$(_find_setup_auto || true)"
if [ -n "$auto_setup" ]; then
  _source_setup "$auto_setup"
  if [ "$_is_sourced" -eq 1 ]; then return 0; else exit 0; fi
fi

if _exists_file "$DEFAULT_VULKAN_SETUP"; then
  _source_setup "$DEFAULT_VULKAN_SETUP"
  if [ "$_is_sourced" -eq 1 ]; then return 0; else exit 0; fi
fi

_die "Could not find a Vulkan setup-env.sh.
- Pass a path:  source ./env-vulkan.sh /path/to/setup-env.sh
- Or set VULKAN_SETUP:  export VULKAN_SETUP=/path/to/setup-env.sh; source ./env-vulkan.sh
- Or edit DEFAULT_VULKAN_SETUP at the top of this script."
