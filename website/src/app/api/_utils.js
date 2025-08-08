import path from "path";

export function repoPath(rel) {
  return path.resolve(process.cwd(), "..", rel);
}

export function resolveDir(envVar, fallback) {
  const p = process.env[envVar] || fallback;
  return path.resolve(p);
}


