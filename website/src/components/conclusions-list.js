import React from "react";

export default function ConclusionsList({ data }) {
  if (!data || !data.length) return <div className="text-sm text-gray-500">No strategic conclusions yet.</div>;
  return (
    <ul className="list-disc pl-5 space-y-1">
      {data.map((c, i) => (
        <li key={i} className="text-sm">{typeof c === "string" ? c : (c.text || JSON.stringify(c))}</li>
      ))}
    </ul>
  );
}


