const CROP_W = 335;
const CROP_H = 354;

export function getScale(width, height) {
  const sx = width / CROP_W;
  const sy = height / CROP_H;
  return { sx, sy };
}

export function scalePoint(p, width, height) {
  const { sx, sy } = getScale(width, height);
  return { x: p.x * sx, y: p.y * sy };
}

export function scalePolygon(poly, width, height) {
  const { sx, sy } = getScale(width, height);
  return poly.map(([x, y]) => [x * sx, y * sy]);
}

export function pointInPolygon(point, polygon) {
  const x = point.x;
  const y = point.y;
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0], yi = polygon[i][1];
    const xj = polygon[j][0], yj = polygon[j][1];
    const intersect = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 0.0000001) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

export { CROP_W, CROP_H };



