export function isEvent(obj) {
  if (!obj || typeof obj !== "object") return false;
  if (typeof obj.timestamp_ms !== "number") return false;
  if (typeof obj.event_type !== "string") return false;
  if (obj.details && typeof obj.details !== "object") return false;
  return true;
}

export function isEventArray(arr) {
  return Array.isArray(arr) && arr.every(isEvent);
}

export function isMaskJson(obj) {
  if (!obj || typeof obj !== "object") return false;
  for (const k of Object.keys(obj)) {
    const poly = obj[k];
    if (!Array.isArray(poly)) return false;
    for (const pt of poly) {
      if (!Array.isArray(pt) || pt.length < 2) return false;
      if (typeof pt[0] !== "number" || typeof pt[1] !== "number") return false;
    }
  }
  return true;
}



