import React from "react";

export default function RoundSummary({ data }) {
  if (!data || !data.length) return <div className="text-sm text-gray-500">No round summaries generated yet.</div>;
  return (
    <div className="space-y-2">
      {data.map((s, i) => (
        <div key={i} className="p-2 bg-bg rounded text-sm">{typeof s === "string" ? s : JSON.stringify(s)}</div>
      ))}
    </div>
  );
}


